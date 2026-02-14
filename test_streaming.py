#!/usr/bin/env python3
"""
Test client for the streaming voice pipeline.
Tests the key latency metrics and streaming functionality.
"""

import asyncio
import json
import base64
import time
import websockets
from pathlib import Path
import wave
import numpy as np

async def test_streaming_pipeline():
    """Test the streaming voice pipeline end-to-end."""
    
    # Create a simple test audio file (1 second of 440Hz tone)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV file temporarily
    test_audio_path = "/tmp/test_audio.wav"
    with wave.open(test_audio_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    # Read the test audio file
    with open(test_audio_path, 'rb') as f:
        test_audio_bytes = f.read()
    
    # Encode to base64
    audio_b64 = base64.b64encode(test_audio_bytes).decode()
    
    print("🎤 Connecting to streaming server...")
    
    try:
        async with websockets.connect('ws://localhost:8765/ws') as websocket:
            print("✅ Connected to WebSocket")
            
            # Send test audio
            message = {
                "type": "audio",
                "audio": audio_b64
            }
            
            print("📡 Sending test audio...")
            start_time = time.time()
            await websocket.send(json.dumps(message))
            
            # Track metrics
            first_audio_time = None
            first_text_time = None
            total_audio_chunks = 0
            total_text_chunks = 0
            streaming_started = False
            
            print("🔍 Listening for responses...")
            
            async for message in websocket:
                msg = json.loads(message)
                current_time = time.time()
                elapsed = (current_time - start_time) * 1000
                
                if msg['type'] == 'state':
                    print(f"  [{elapsed:6.0f}ms] State: {msg['state']}")
                
                elif msg['type'] == 'transcript':
                    print(f"  [{elapsed:6.0f}ms] 📝 Transcript: {msg['text']}")
                
                elif msg['type'] == 'streaming_start':
                    print(f"  [{elapsed:6.0f}ms] 🚀 Streaming started")
                    streaming_started = True
                
                elif msg['type'] == 'text_chunk':
                    if first_text_time is None:
                        first_text_time = current_time
                        print(f"  [{elapsed:6.0f}ms] 📜 First text chunk: '{msg['text']}' (LLM time: {msg.get('llm_time', 0)*1000:.0f}ms)")
                    else:
                        print(f"  [{elapsed:6.0f}ms] 📜 Text chunk: '{msg['text']}'")
                    total_text_chunks += 1
                
                elif msg['type'] == 'audio_chunk':
                    if first_audio_time is None:
                        first_audio_time = current_time
                        first_audio_latency = (first_audio_time - start_time) * 1000
                        print(f"  [{elapsed:6.0f}ms] 🔊 First audio chunk received! (TTS time: {msg.get('tts_time', 0)*1000:.0f}ms)")
                        print(f"  🎯 FIRST AUDIO LATENCY: {first_audio_latency:.0f}ms")
                    else:
                        print(f"  [{elapsed:6.0f}ms] 🔊 Audio chunk (sentence: '{msg.get('sentence', '')[:50]}...')")
                    total_audio_chunks += 1
                
                elif msg['type'] == 'response_complete':
                    print(f"  [{elapsed:6.0f}ms] ✅ Response complete")
                    print(f"      Full text: '{msg['text']}'")
                    print(f"      Total LLM time: {msg['total_time']*1000:.0f}ms")
                    if msg.get('first_audio_latency'):
                        print(f"      Server-measured first audio latency: {msg['first_audio_latency']*1000:.0f}ms")
                
                elif msg['type'] == 'streaming_complete':
                    total_time = (current_time - start_time) * 1000
                    print(f"  [{elapsed:6.0f}ms] 🏁 Streaming complete")
                    print(f"\n📊 FINAL METRICS:")
                    print(f"   Total time: {total_time:.0f}ms")
                    if first_audio_time:
                        print(f"   Time to first audio: {(first_audio_time - start_time)*1000:.0f}ms")
                    if first_text_time:
                        print(f"   Time to first text: {(first_text_time - start_time)*1000:.0f}ms")
                    print(f"   Text chunks: {total_text_chunks}")
                    print(f"   Audio chunks: {total_audio_chunks}")
                    
                    # Performance evaluation
                    if first_audio_time:
                        latency = (first_audio_time - start_time) * 1000
                        if latency < 1500:
                            print(f"   🎯 SUCCESS: First audio latency {latency:.0f}ms < 1500ms target!")
                        else:
                            print(f"   ❌ MISS: First audio latency {latency:.0f}ms > 1500ms target")
                    
                    break
                
                elif msg['type'] == 'error':
                    print(f"  [{elapsed:6.0f}ms] ❌ Error: {msg['message']}")
                    break
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    return True

async def test_health_check():
    """Test the health check endpoint."""
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8765/health")
            health_data = response.json()
            
            print("🏥 Health Check:")
            for key, value in health_data.items():
                if isinstance(value, bool):
                    status = "✅" if value else "❌"
                    print(f"   {key}: {status}")
                else:
                    print(f"   {key}: {value}")
            
            return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Testing Samantha Voice Streaming Pipeline")
    print("=" * 50)
    
    # Test health first
    health_ok = await test_health_check()
    if not health_ok:
        print("❌ Health check failed, aborting tests")
        return
    
    print()
    
    # Test streaming pipeline
    success = await test_streaming_pipeline()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Tests failed")

if __name__ == "__main__":
    asyncio.run(main())