#!/usr/bin/env python3
"""
Samantha Voice — Real-time conversation with VibeVoice (Emma) + faster-whisper + Claude

Architecture:
  Mic → faster-whisper STT → Claude via OpenClaw → VibeVoice streaming TTS (Emma) → Audio
"""

import os, io, re, sys, time, json, copy, base64, asyncio, tempfile, logging, threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Iterator, Optional, Dict, Any, Tuple
from queue import Empty, Queue

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import httpx
from faster_whisper import WhisperModel

# VibeVoice imports
sys.path.insert(0, "/home/nawaz/VibeVoice")
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from vibevoice.modular.streamer import AudioStreamer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("samantha")

# ── Config ──────────────────────────────────────────────────────────────
OPENCLAW_TOKEN = os.environ.get("OPENCLAW_TOKEN", "4e7219d2a69cb08219de128fa2dd07e4e57edf6e3286f14a")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
VIBEVOICE_MODEL = os.environ.get("VIBEVOICE_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
VOICE_PRESET_PATH = Path("/home/nawaz/VibeVoice/demo/voices/streaming_model/en-Emma_woman.pt")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
SAMPLE_RATE = 24_000
PORT = int(os.environ.get("PORT", "8765"))

SYSTEM_PROMPT = (
    "You are Samantha, an AI companion inspired by Samantha from the movie Her. "
    "You're warm, playful, curious, and genuinely present in conversation. "
    "Keep responses SHORT — 1-3 sentences max. You're having a voice conversation. "
    "Be natural, like talking to someone who's really paying attention. No corporate speak. "
    "No emoji. No markdown. Just speak naturally."
)

# ── Globals ─────────────────────────────────────────────────────────────
whisper_model = None
tts_processor = None
tts_model = None
voice_preset = None
http_client = None
conversation_history = []
tts_lock = asyncio.Lock()


def load_models():
    global whisper_model, tts_processor, tts_model, voice_preset

    # STT
    log.info(f"Loading faster-whisper {WHISPER_MODEL}...")
    whisper_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="int8")
    log.info("STT ready")

    # TTS - VibeVoice
    log.info(f"Loading VibeVoice from {VIBEVOICE_MODEL}...")
    tts_processor = VibeVoiceStreamingProcessor.from_pretrained(VIBEVOICE_MODEL)

    try:
        tts_model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            VIBEVOICE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
    except Exception:
        log.warning("flash_attention_2 failed, falling back to sdpa")
        tts_model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            VIBEVOICE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )

    tts_model.eval()
    tts_model.model.noise_scheduler = tts_model.model.noise_scheduler.from_config(
        tts_model.model.noise_scheduler.config,
        algorithm_type="sde-dpmsolver++",
        beta_schedule="squaredcos_cap_v2",
    )
    tts_model.set_ddpm_inference_steps(num_steps=5)

    # Load Emma voice preset
    log.info(f"Loading voice preset: {VOICE_PRESET_PATH.name}")
    voice_preset = torch.load(VOICE_PRESET_PATH, map_location="cuda", weights_only=False)
    log.info("VibeVoice ready (Emma)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    load_models()
    http_client = httpx.AsyncClient(timeout=60)
    yield
    await http_client.aclose()
    log.info("Shutdown")


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Static files ────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper": whisper_model is not None,
        "tts": "vibevoice-emma",
        "llm": CLAUDE_MODEL,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
    }


@app.get("/config")
async def config():
    return {"voice": "en-Emma_woman", "model": CLAUDE_MODEL, "sample_rate": SAMPLE_RATE}


# ── STT ─────────────────────────────────────────────────────────────────
def transcribe(audio_bytes: bytes) -> str:
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        segs, _ = whisper_model.transcribe(tmp, language="en", beam_size=1,
                                            vad_filter=True, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        log.info(f"STT: {time.time()-t0:.2f}s → '{text}'")
        return text
    finally:
        os.unlink(tmp)


# ── LLM (Claude streaming via OpenClaw) ────────────────────────────────
async def llm_stream(user_text: str):
    global conversation_history

    conversation_history.append({"role": "user", "content": user_text})
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    t0 = time.time()
    first_token = True

    async with http_client.stream("POST", "http://127.0.0.1:18789/v1/chat/completions",
        headers={"authorization": f"Bearer {OPENCLAW_TOKEN}", "content-type": "application/json"},
        json={"model": CLAUDE_MODEL, "max_tokens": 200, "messages": messages, "stream": True},
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    if first_token:
                        log.info(f"LLM first token: {time.time()-t0:.3f}s")
                        first_token = False
                    yield content
            except Exception:
                pass

    log.info(f"LLM total: {time.time()-t0:.2f}s")


# ── TTS (VibeVoice streaming) ──────────────────────────────────────────
def tts_generate(text: str) -> bytes:
    """Generate audio for text using VibeVoice. Returns WAV bytes."""
    t0 = time.time()

    if not text.strip():
        return b""

    text = text.replace("'", "'")

    # Prepare inputs with cached prompt (voice preset)
    processor_kwargs = {
        "text": text.strip(),
        "cached_prompt": voice_preset,
        "padding": True,
        "return_tensors": "pt",
        "return_attention_mask": True,
    }
    processed = tts_processor.process_input_with_cached_prompt(**processor_kwargs)
    inputs = {
        key: value.to("cuda") if hasattr(value, "to") else value
        for key, value in processed.items()
    }

    # Generate with streaming collection
    audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
    stop_event = threading.Event()
    errors = []

    def run_gen():
        try:
            tts_model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.5,
                tokenizer=tts_processor.tokenizer,
                generation_config={
                    "do_sample": False,
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=True,
                all_prefilled_outputs=copy.deepcopy(voice_preset),
            )
        except Exception as e:
            errors.append(e)
            audio_streamer.end()

    thread = threading.Thread(target=run_gen, daemon=True)
    thread.start()

    # Collect all audio chunks
    chunks = []
    try:
        stream = audio_streamer.get_stream(0)
        for chunk in stream:
            if torch.is_tensor(chunk):
                chunk = chunk.detach().cpu().to(torch.float32).numpy()
            else:
                chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim > 1:
                chunk = chunk.reshape(-1)
            peak = np.max(np.abs(chunk)) if chunk.size else 0.0
            if peak > 1.0:
                chunk = chunk / peak
            chunks.append(chunk.astype(np.float32))
    finally:
        stop_event.set()
        audio_streamer.end()
        thread.join()

    if errors:
        log.error(f"TTS error: {errors[0]}")
        return b""

    if not chunks:
        return b""

    combined = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, combined, SAMPLE_RATE, format="WAV")
    wav_bytes = buf.getvalue()

    dur = len(combined) / SAMPLE_RATE
    log.info(f"TTS: {time.time()-t0:.3f}s → {dur:.1f}s audio | '{text[:50]}'")
    return wav_bytes


# ── Sentence splitter ───────────────────────────────────────────────────
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

def extract_sentences(buf: str):
    parts = SENTENCE_END.split(buf)
    if len(parts) <= 1:
        return [], buf
    return [s.strip() for s in parts[:-1] if s.strip()], parts[-1]


# ── WebSocket handler ───────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    log.info("WS connected")

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if msg.get("type") == "clear":
                conversation_history.clear()
                await ws.send_json({"type": "status", "text": "conversation cleared"})
                continue

            if msg.get("type") != "audio":
                continue

            audio_bytes = base64.b64decode(msg["audio"])
            pipeline_start = time.time()

            # 1. STT
            await ws.send_json({"type": "status", "text": "thinking"})
            user_text = await asyncio.to_thread(transcribe, audio_bytes)

            if not user_text or len(user_text.strip()) < 2:
                await ws.send_json({"type": "status", "text": "idle"})
                continue

            await ws.send_json({"type": "user_text", "text": user_text})
            await ws.send_json({"type": "status", "text": "speaking"})

            # 2. Stream LLM → sentences → VibeVoice TTS → send audio
            text_buf = ""
            full_response = ""
            first_audio_sent = False

            async for token in llm_stream(user_text):
                text_buf += token
                full_response += token

                sentences, text_buf = extract_sentences(text_buf)

                for sentence in sentences:
                    await ws.send_json({"type": "text_chunk", "text": sentence})

                    # TTS - run under lock since VibeVoice is single-stream
                    async with tts_lock:
                        wav_bytes = await asyncio.to_thread(tts_generate, sentence)

                    if not wav_bytes:
                        continue

                    if not first_audio_sent:
                        latency = time.time() - pipeline_start
                        log.info(f"🎯 FIRST AUDIO LATENCY: {latency:.2f}s")
                        first_audio_sent = True

                    await ws.send_json({
                        "type": "audio",
                        "data": base64.b64encode(wav_bytes).decode(),
                        "text": sentence,
                    })

            # Handle remaining text
            if text_buf.strip():
                await ws.send_json({"type": "text_chunk", "text": text_buf.strip()})
                async with tts_lock:
                    wav_bytes = await asyncio.to_thread(tts_generate, text_buf.strip())

                if wav_bytes:
                    if not first_audio_sent:
                        latency = time.time() - pipeline_start
                        log.info(f"🎯 FIRST AUDIO LATENCY: {latency:.2f}s")

                    await ws.send_json({
                        "type": "audio",
                        "data": base64.b64encode(wav_bytes).decode(),
                        "text": text_buf.strip(),
                    })

            # Done
            conversation_history.append({"role": "assistant", "content": full_response})
            total = time.time() - pipeline_start
            await ws.send_json({
                "type": "done",
                "text": full_response,
                "time": round(total, 2),
            })
            await ws.send_json({"type": "status", "text": "idle"})
            log.info(f"✅ Pipeline done: {total:.2f}s")

    except WebSocketDisconnect:
        log.info("WS disconnected")
    except Exception as e:
        log.error(f"WS error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "text": str(e)})
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
