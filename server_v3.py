#!/usr/bin/env python3
"""
Samantha Voice v3 — Low-latency streaming pipeline

Architecture:
  Mic → faster-whisper STT (~700ms)
  → Claude via OpenClaw streaming (~1.5s to first sentence)
  → Kokoro TTS per sentence (~100ms)
  → Audio streams to client as sentences complete

Target: ~2.3s from user stops talking to first audio plays
"""

import os, io, re, time, json, base64, asyncio, tempfile, logging
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("samantha-v3")

# ── Config ──────────────────────────────────────────────────────────────
OPENCLAW_TOKEN = os.environ.get("OPENCLAW_TOKEN", "4e7219d2a69cb08219de128fa2dd07e4e57edf6e3286f14a")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")

SYSTEM_PROMPT = (
    "You are Samantha, an AI companion inspired by Samantha from the movie Her. "
    "You're warm, playful, curious, and genuinely present in conversation. "
    "Keep responses SHORT — 1-3 sentences max. You're having a voice conversation. "
    "Be natural, like talking to someone who's really paying attention. No corporate speak. "
    "No emoji. No markdown. Just speak naturally."
)

# ── Globals ─────────────────────────────────────────────────────────────
whisper_model = None
kokoro_pipe = None
conversation_history = []
http_client = None


def load_models():
    global whisper_model, kokoro_pipe

    log.info("Loading faster-whisper large-v3-turbo (int8)...")
    whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
    log.info("STT ready")

    log.info("Loading Kokoro TTS (82M)...")
    os.environ["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
    from kokoro import KPipeline
    kokoro_pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    # Warmup — first call is slow
    log.info("Warming up Kokoro...")
    for _, _, _ in kokoro_pipe("Hello.", voice=KOKORO_VOICE, speed=1.0):
        pass
    log.info(f"Kokoro ready (voice: {KOKORO_VOICE})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    load_models()
    http_client = httpx.AsyncClient(timeout=30)
    yield
    await http_client.aclose()
    log.info("Shutdown")


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper": whisper_model is not None,
        "tts": "kokoro-82M",
        "tts_voice": KOKORO_VOICE,
        "llm": "claude-via-openclaw",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
    }


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
    """Stream tokens from Claude. Yields token strings."""
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


# ── TTS (Kokoro) ───────────────────────────────────────────────────────
def tts_sentence(text: str) -> bytes:
    """Generate audio for a sentence with Kokoro. Returns WAV bytes."""
    t0 = time.time()

    audio_chunks = []
    for _, _, audio in kokoro_pipe(text, voice=KOKORO_VOICE, speed=1.0):
        audio_chunks.append(audio.numpy() if hasattr(audio, 'numpy') else np.array(audio))

    if not audio_chunks:
        return b""

    combined = np.concatenate(audio_chunks).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, combined, 24000, format="WAV")
    wav_bytes = buf.getvalue()

    dur = len(combined) / 24000
    log.info(f"TTS: {time.time()-t0:.3f}s → {dur:.1f}s audio | '{text[:50]}'")
    return wav_bytes


# ── Sentence splitter ───────────────────────────────────────────────────
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

def extract_sentences(buf: str):
    """Extract complete sentences from buffer. Returns (sentences, remaining)."""
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

            if msg.get("type") != "audio":
                continue

            audio_bytes = base64.b64decode(msg["audio"])
            pipeline_start = time.time()

            # 1. STT
            await ws.send_json({"type": "state", "state": "thinking"})
            user_text = await asyncio.to_thread(transcribe, audio_bytes)

            if not user_text or len(user_text.strip()) < 2:
                await ws.send_json({"type": "state", "state": "idle"})
                continue

            await ws.send_json({"type": "transcript", "role": "user", "text": user_text})
            await ws.send_json({"type": "state", "state": "speaking"})

            # 2. Stream LLM → detect sentences → Kokoro TTS each → send audio
            text_buf = ""
            full_response = ""
            first_audio_sent = False
            sentences_generated = 0

            async for token in llm_stream(user_text):
                text_buf += token
                full_response += token

                sentences, text_buf = extract_sentences(text_buf)

                for sentence in sentences:
                    sentences_generated += 1
                    await ws.send_json({"type": "text_chunk", "text": sentence})

                    wav_bytes = await asyncio.to_thread(tts_sentence, sentence)
                    if not wav_bytes:
                        continue

                    if not first_audio_sent:
                        latency = time.time() - pipeline_start
                        log.info(f"🎯 FIRST AUDIO LATENCY: {latency:.2f}s")
                        first_audio_sent = True

                    await ws.send_json({
                        "type": "audio_chunk",
                        "audio": base64.b64encode(wav_bytes).decode(),
                        "format": "wav",
                    })

            # Handle remaining text
            if text_buf.strip():
                sentences_generated += 1
                await ws.send_json({"type": "text_chunk", "text": text_buf.strip()})
                wav_bytes = await asyncio.to_thread(tts_sentence, text_buf.strip())

                if wav_bytes:
                    if not first_audio_sent:
                        latency = time.time() - pipeline_start
                        log.info(f"🎯 FIRST AUDIO LATENCY: {latency:.2f}s")
                        first_audio_sent = True

                    await ws.send_json({
                        "type": "audio_chunk",
                        "audio": base64.b64encode(wav_bytes).decode(),
                        "format": "wav",
                    })

            # Done
            conversation_history.append({"role": "assistant", "content": full_response})
            total = time.time() - pipeline_start
            await ws.send_json({"type": "transcript", "role": "assistant", "text": full_response})
            await ws.send_json({
                "type": "response_complete",
                "total_time": round(total, 2),
                "sentences": sentences_generated,
            })
            await ws.send_json({"type": "state", "state": "idle"})
            log.info(f"✅ Pipeline done: {total:.2f}s, {sentences_generated} sentences")

    except WebSocketDisconnect:
        log.info("WS disconnected")
    except Exception as e:
        log.error(f"WS error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
