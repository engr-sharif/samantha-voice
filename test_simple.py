#!/usr/bin/env python3
"""
Simple test for streaming pipeline - use actual voice recording
"""

import asyncio
import json
import base64
import time
import wave
import numpy as np

async def test_with_text_synthesis():
    """Test by synthesizing speech from text and then processing it."""
    print("🧪 Testing streaming pipeline with synthesized speech")
    
    # Create a more speech-like signal (mix of frequencies)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a more speech-like signal with multiple harmonics and modulation
    fundamental = 200  # Base frequency
    signal = (np.sin(2 * np.pi * fundamental * t) * 0.3 +
              np.sin(2 * np.pi * fundamental * 2 * t) * 0.2 +
              np.sin(2 * np.pi * fundamental * 3 * t) * 0.1)
    
    # Add some amplitude modulation to make it more speech-like
    modulation = 1 + 0.5 * np.sin(2 * np.pi * 5 * t)
    signal = signal * modulation
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.05, len(signal))
    signal = signal + noise
    
    # Clip and normalize
    signal = np.clip(signal, -1.0, 1.0)
    
    # Convert to 16-bit PCM
    audio_data = (signal * 32767).astype(np.int16)
    
    # Create WAV file in memory
    import io
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    wav_bytes = wav_buffer.getvalue()
    audio_b64 = base64.b64encode(wav_bytes).decode()
    
    print("📡 Testing with synthesized speech-like audio...")
    await test_pipeline(audio_b64)

async def test_with_silence():
    """Test with just silence to trigger a quick response."""
    print("🧪 Testing streaming pipeline with silence (should get ignored)")
    
    # Create 0.5 seconds of silence
    sample_rate = 16000
    duration = 0.5
    silence = np.zeros(int(sample_rate * duration), dtype=np.int16)
    
    # Create WAV file
    import io
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence.tobytes())
    
    wav_bytes = wav_buffer.getvalue()
    audio_b64 = base64.b64encode(wav_bytes).decode()
    
    print("📡 Testing with silence...")
    await test_pipeline(audio_b64)

async def test_pipeline(audio_b64):
    """Test the pipeline with given audio."""
    import websockets
    
    try:
        uri = "ws://localhost:8765/ws"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Send the audio
            message = {
                "type": "audio",
                "audio": audio_b64
            }
            
            start_time = time.time()
            await websocket.send(json.dumps(message))
            print(f"📤 Sent audio at {time.time()}")
            
            # Listen for responses
            timeout_count = 0
            max_timeout = 10
            
            while timeout_count < max_timeout:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    msg = json.loads(message)
                    
                    elapsed = (time.time() - start_time) * 1000
                    
                    print(f"[{elapsed:6.0f}ms] {msg['type']}: {msg}")
                    
                    # Stop on completion or error
                    if msg['type'] in ['streaming_complete', 'error', 'state'] and msg.get('state') == 'idle':
                        if msg['type'] == 'state' and msg.get('state') == 'idle':
                            print("🔄 Returned to idle state - probably no speech detected")
                        break
                        
                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(f"⏰ Timeout {timeout_count}/{max_timeout}")
                    
            if timeout_count >= max_timeout:
                print("❌ Test timed out")
            else:
                print("✅ Test completed")
                
    except Exception as e:
        print(f"❌ Error: {e}")

async def test_health():
    """Test health endpoint."""
    import httpx
    
    print("🏥 Checking health...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8765/health")
            data = response.json()
            
            print("Health Status:")
            for k, v in data.items():
                status = "✅" if v is True else "❌" if v is False else ""
                print(f"  {k}: {v} {status}")
            
            return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

async def main():
    print("🧪 Simple Streaming Pipeline Test")
    print("=" * 40)
    
    # Health check first
    if not await test_health():
        print("❌ Health check failed")
        return
    
    print()
    
    # Test with silence (should be quick)
    await test_with_silence()
    
    print("\n" + "=" * 40)
    
    # Test with synthesized speech
    await test_with_text_synthesis()

if __name__ == "__main__":
    asyncio.run(main())