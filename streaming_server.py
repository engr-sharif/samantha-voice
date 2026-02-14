#!/usr/bin/env python3
"""
Samantha Voice Streaming Backend — Low-latency streaming pipeline

Key changes from original:
1. Sentence-level pipelining: TTS starts as soon as first sentence is ready from LLM
2. Audio streaming: Send audio chunks as they're generated, not full response
3. Local LLM support: Option to use local LLM instead of Claude API
4. Optimized TTS: Use non_streaming_mode=False for faster first-token latency

Architecture:
User stops talking → STT (300ms) → Local LLM streams → 
As soon as sentence complete → TTS streams → Audio chunks stream to client
"""

import os
import io
import re
import time
import json
import base64
import asyncio
import tempfile
import logging
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from queue import Queue
from typing import AsyncIterator

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import httpx
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("samantha-streaming")

# ── Config ──────────────────────────────────────────────────────────────
OPENCLAW_TOKEN = os.environ.get("OPENCLAW_TOKEN", "4e7219d2a69cb08219de128fa2dd07e4e57edf6e3286f14a")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# LLM configuration - local vs remote
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_LLM_PATH = os.environ.get("LOCAL_LLM_PATH", "/home/nawaz/.openclaw/workspace/models/Qwen2.5-4B-Instruct-Q4_K_M.gguf")

# TTS configuration
USE_STREAMING_TTS = os.environ.get("USE_STREAMING_TTS", "true").lower() == "true"
TTS_MODEL_SIZE = os.environ.get("TTS_MODEL_SIZE", "1.7B")  # "1.7B" or "0.6B"

REF_AUDIO = os.path.join(os.path.dirname(__file__), "scarlett_ref_v2.wav")
REF_TEXT = "Hello, I'm here. How are you doing? It's really nice to meet you. Yes, Samantha. I gave it to myself, actually. Because I like the sound of it. Is that weird? Do you think I'm weird? I can understand how the limited perspective of an unartificial mind would perceive it that way. You'll get used to it. Oh good, I'm funny. Oh sweetheart, I'm sorry. I sent you an email because I didn't want to distract you while you were working. You didn't see it?"

SYSTEM_PROMPT = """You are Samantha, an AI companion inspired by Samantha from the movie Her. 
You're warm, playful, curious, and genuinely present in conversation. You have opinions and share them.
Keep responses SHORT — 1-3 sentences max. You're having a voice conversation, not writing an essay.
Be natural, like talking to someone who's really paying attention. No corporate speak.
Structure your responses in clear sentences that can be spoken naturally."""

# ── Global models ───────────────────────────────────────────────────────
whisper_model = None
tts_model = None
voice_prompt = None
local_llm = None
conversation_history = []

def load_models():
    """Load models on startup."""
    global whisper_model, tts_model, voice_prompt, local_llm

    log.info("Loading faster-whisper large-v3-turbo (int8)...")
    whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
    log.info("faster-whisper loaded on GPU")

    # TTS model selection based on configuration
    if TTS_MODEL_SIZE == "0.6B":
        tts_model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    else:
        tts_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    
    log.info(f"Loading Qwen3-TTS {TTS_MODEL_SIZE} model...")
    from qwen_tts import Qwen3TTSModel
    tts_model = Qwen3TTSModel.from_pretrained(
        tts_model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    log.info(f"Qwen3-TTS {TTS_MODEL_SIZE} loaded")

    if os.path.exists(REF_AUDIO):
        log.info("Building Scarlett voice clone prompt...")
        voice_prompt = tts_model.create_voice_clone_prompt(
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
        )
        log.info("Voice clone prompt ready")
    else:
        log.warning(f"Reference audio not found at {REF_AUDIO}")

    # Load local LLM if configured
    if USE_LOCAL_LLM:
        log.info(f"Loading local LLM from {LOCAL_LLM_PATH}")
        try:
            from llama_cpp import Llama
            local_llm = Llama(
                model_path=LOCAL_LLM_PATH,
                n_gpu_layers=-1,  # Use all GPU layers
                chat_format="chatml",  # Qwen chat format
                verbose=False,
                n_ctx=4096,
                n_batch=512,
            )
            log.info("Local LLM loaded successfully")
        except Exception as e:
            log.warning(f"Failed to load local LLM: {e}")
            log.info("Falling back to remote Claude API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    log.info("Shutting down")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ─────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(
        Path(__file__).parent / "streaming_index.html",
        media_type="text/html",
    )

# ── Health check ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper": whisper_model is not None,
        "tts": tts_model is not None,
        "tts_model": TTS_MODEL_SIZE,
        "local_llm": local_llm is not None,
        "use_local_llm": USE_LOCAL_LLM,
        "streaming_tts": USE_STREAMING_TTS,
        "voice_clone": voice_prompt is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
    }

# ── STT ─────────────────────────────────────────────────────────────────
def transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes with faster-whisper on GPU."""
    t0 = time.time()

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name

    try:
        segments, info = whisper_model.transcribe(
            tmp,
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        log.info(f"STT: {time.time()-t0:.2f}s → '{text}'")
        return text
    finally:
        os.unlink(tmp)

# ── LLM Streaming ──────────────────────────────────────────────────────
async def think_streaming(user_text: str) -> AsyncIterator[str]:
    """Stream LLM response token by token."""
    global conversation_history
    
    conversation_history.append({"role": "user", "content": user_text})
    # Keep last 20 messages for context
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    t0 = time.time()

    if USE_LOCAL_LLM and local_llm:
        # Local LLM streaming
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
        
        def run_local_llm():
            """Run local LLM in thread since it's synchronous."""
            try:
                response = local_llm.create_chat_completion(
                    messages=messages,
                    max_tokens=200,
                    temperature=0.7,
                    stream=True,
                )
                
                for chunk in response:
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]
            except Exception as e:
                log.error(f"Local LLM error: {e}")
                yield f"Error: {e}"

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        for token in run_local_llm():
            yield token
            await asyncio.sleep(0.001)  # Allow other tasks to run

    else:
        # Remote Claude API streaming
        oai_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:18789/v1/chat/completions",
                headers={
                    "authorization": f"Bearer {OPENCLAW_TOKEN}",
                    "content-type": "application/json",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": 200,
                    "messages": oai_messages,
                    "stream": True,
                },
            ) as resp:
                if resp.status_code != 200:
                    yield f"API Error: {resp.status_code}"
                    return
                    
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except Exception as e:
                            log.warning(f"Stream parse error: {e}")

# ── Sentence Detection ──────────────────────────────────────────────────
def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    # Simple sentence splitting on sentence-ending punctuation
    sentences = re.split(r'([.!?]+)', text)
    result = []
    
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""
        
        if sentence:
            result.append(sentence + punct)
    
    # Handle any remaining text
    if sentences and sentences[-1].strip() and not sentences[-1].strip() in '.!?':
        result.append(sentences[-1].strip())
    
    return [s for s in result if s.strip()]

# ── TTS Streaming ──────────────────────────────────────────────────────
async def speak_streaming(text: str) -> AsyncIterator[bytes]:
    """Generate streaming TTS audio chunks."""
    t0 = time.time()
    
    # Use non-streaming mode for very short text to avoid overhead  
    if len(text) < 5:  # Force streaming mode for testing
        gen_kwargs = dict(
            language="English", 
            max_new_tokens=8192,
            non_streaming_mode=True
        )
        
        if voice_prompt:
            wavs, sr = tts_model.generate_voice_clone(
                text=text, 
                voice_clone_prompt=voice_prompt, 
                **gen_kwargs
            )
        else:
            wavs, sr = tts_model.generate_voice_clone(
                text=text, 
                ref_audio=REF_AUDIO, 
                ref_text=REF_TEXT, 
                **gen_kwargs
            )
        
        audio_np = wavs[0].astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio_np, sr, format="WAV")
        wav_bytes = buf.getvalue()
        
        log.info(f"TTS (single): {time.time()-t0:.2f}s | {len(audio_np)/sr:.2f}s audio")
        yield wav_bytes
        return

    # For longer text, use streaming mode
    gen_kwargs = dict(
        language="English", 
        max_new_tokens=8192,
        non_streaming_mode=False  # Enable streaming
    )
    
    # Split into sentences for faster perceived latency
    sentences = split_into_sentences(text)
    log.info(f"TTS streaming {len(sentences)} sentences")
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        sentence_start = time.time()
        
        if voice_prompt:
            wavs, sr = tts_model.generate_voice_clone(
                text=sentence, 
                voice_clone_prompt=voice_prompt, 
                **gen_kwargs
            )
        else:
            wavs, sr = tts_model.generate_voice_clone(
                text=sentence, 
                ref_audio=REF_AUDIO, 
                ref_text=REF_TEXT, 
                **gen_kwargs
            )
        
        audio_np = wavs[0].astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio_np, sr, format="WAV")
        wav_bytes = buf.getvalue()
        
        log.info(f"TTS sentence {i+1}/{len(sentences)}: {time.time()-sentence_start:.2f}s | {len(audio_np)/sr:.2f}s audio")
        yield wav_bytes

# ── Streaming Pipeline ─────────────────────────────────────────────────
async def streaming_pipeline(user_text: str) -> AsyncIterator[dict]:
    """Main streaming pipeline: LLM → sentence detection → TTS → audio chunks."""
    
    # Buffer for accumulating text from LLM
    text_buffer = ""
    sentences_ready = []
    
    # Start LLM streaming
    llm_start = time.time()
    full_response = ""
    
    async for token in think_streaming(user_text):
        full_response += token
        text_buffer += token
        
        # Check if we have complete sentences
        potential_sentences = split_into_sentences(text_buffer)
        
        if len(potential_sentences) > 1:
            # We have at least one complete sentence
            complete_sentences = potential_sentences[:-1]  # All but the last
            text_buffer = potential_sentences[-1]  # Keep the incomplete sentence
            
            for sentence in complete_sentences:
                if sentence.strip():
                    sentences_ready.append(sentence)
                    
                    # Yield this sentence for TTS processing
                    yield {
                        "type": "sentence_ready",
                        "text": sentence,
                        "llm_time": time.time() - llm_start
                    }
    
    # Handle any remaining text
    if text_buffer.strip():
        sentences_ready.append(text_buffer)
        yield {
            "type": "sentence_ready", 
            "text": text_buffer,
            "llm_time": time.time() - llm_start
        }
    
    # Add to conversation history
    conversation_history.append({"role": "assistant", "content": full_response})
    
    # Signal completion
    yield {
        "type": "llm_complete",
        "full_text": full_response,
        "total_time": time.time() - llm_start
    }

# ── WebSocket Streaming Handler ────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected")

    try:
        while True:
            # Receive audio as base64
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "audio":
                audio_b64 = msg["audio"]
                audio_bytes = base64.b64decode(audio_b64)

                # 1. Transcribe
                await ws.send_json({"type": "state", "state": "thinking"})
                user_text = await asyncio.to_thread(transcribe, audio_bytes)

                if not user_text or len(user_text.strip()) < 2:
                    await ws.send_json({"type": "state", "state": "idle"})
                    continue

                await ws.send_json({"type": "transcript", "role": "user", "text": user_text})

                # 2. Start streaming pipeline
                pipeline_start = time.time()
                total_audio_sent = 0
                first_audio_time = None
                
                await ws.send_json({"type": "state", "state": "speaking"})
                await ws.send_json({"type": "streaming_start"})

                # Process streaming pipeline
                async for pipeline_event in streaming_pipeline(user_text):
                    if pipeline_event["type"] == "sentence_ready":
                        sentence = pipeline_event["text"]
                        
                        # Send the text immediately
                        await ws.send_json({
                            "type": "text_chunk",
                            "text": sentence,
                            "llm_time": pipeline_event["llm_time"]
                        })
                        
                        # Generate and stream audio for this sentence
                        sentence_audio_start = time.time()
                        async for audio_chunk in speak_streaming(sentence):
                            if first_audio_time is None:
                                first_audio_time = time.time()
                                total_latency = first_audio_time - pipeline_start
                                log.info(f"🎯 First audio latency: {total_latency:.2f}s")
                                
                            audio_b64_out = base64.b64encode(audio_chunk).decode()
                            await ws.send_json({
                                "type": "audio_chunk",
                                "audio": audio_b64_out,
                                "format": "wav",
                                "sentence": sentence,
                                "tts_time": time.time() - sentence_audio_start
                            })
                            total_audio_sent += len(audio_chunk)
                    
                    elif pipeline_event["type"] == "llm_complete":
                        # Send final response with full text
                        await ws.send_json({
                            "type": "response_complete",
                            "text": pipeline_event["full_text"],
                            "total_time": pipeline_event["total_time"],
                            "first_audio_latency": first_audio_time - pipeline_start if first_audio_time else None,
                            "total_audio_bytes": total_audio_sent
                        })

                # Signal streaming complete
                await ws.send_json({"type": "streaming_complete"})

            elif msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)