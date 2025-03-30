import os
import base64
import json
import asyncio
import torch
import torchaudio
import numpy as np
import io
import whisperx
from io import BytesIO
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generator import load_csm_1b, Segment
import uvicorn
import time
import gc
from collections import deque

# Select device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# Initialize the model
generator = load_csm_1b(device=device)

# Initialize WhisperX for ASR
print("Loading WhisperX model...")
# Use a smaller model for faster response times
asr_model = whisperx.load_model("medium", device, compute_type="float16")
print("WhisperX model loaded!")

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Mount a static files directory if you have any static assets like CSS or JS
static_dir = os.path.join(base_dir, "static")
os.makedirs(static_dir, exist_ok=True)  # Create the directory if it doesn't exist
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Define route to serve index.html as the main page
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open(os.path.join(base_dir, "index.html"), "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<html><body><h1>Error: index.html not found</h1></body></html>")

# Add a favicon endpoint (optional, but good to have)
@app.get("/favicon.ico")
async def get_favicon():
    favicon_path = os.path.join(static_dir, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    else:
        return HTMLResponse(status_code=204)  # No content

# Connection manager to handle multiple clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

# Silence detection parameters
SILENCE_THRESHOLD = 0.01  # Adjust based on your audio normalization
SILENCE_DURATION_SEC = 1.0  # How long silence must persist to be considered "stopped talking"

# Helper function to convert audio data
async def decode_audio_data(audio_data: str) -> torch.Tensor:
    """Decode base64 audio data to a torch tensor"""
    try:
        # Decode base64 audio data
        binary_data = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
        
        # Save to a temporary WAV file first
        temp_file = BytesIO(binary_data)
        
        # Load audio from binary data, explicitly specifying the format
        audio_tensor, sample_rate = torchaudio.load(temp_file, format="wav")
        
        # Resample if needed
        if sample_rate != generator.sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), 
                orig_freq=sample_rate, 
                new_freq=generator.sample_rate
            )
        else:
            audio_tensor = audio_tensor.squeeze(0)
            
        return audio_tensor
    except Exception as e:
        print(f"Error decoding audio: {str(e)}")
        # Return a small silent audio segment as fallback
        return torch.zeros(generator.sample_rate // 2)  # 0.5 seconds of silence


async def encode_audio_data(audio_tensor: torch.Tensor) -> str:
    """Encode torch tensor audio to base64 string"""
    buf = BytesIO()
    torchaudio.save(buf, audio_tensor.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
    buf.seek(0)
    audio_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:audio/wav;base64,{audio_base64}"


async def transcribe_audio(audio_tensor: torch.Tensor) -> str:
    """Transcribe audio using WhisperX"""
    try:
        # Save the tensor to a temporary file
        temp_file = BytesIO()
        torchaudio.save(temp_file, audio_tensor.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
        temp_file.seek(0)
        
        # Create a temporary file on disk (WhisperX requires a file path)
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(temp_file.read())
        
        # Load and transcribe the audio
        audio = whisperx.load_audio(temp_path)
        result = asr_model.transcribe(audio, batch_size=16)
        
        # Clean up
        os.remove(temp_path)
        
        # Get the transcription text
        if result["segments"] and len(result["segments"]) > 0:
            # Combine all segments
            transcription = " ".join([segment["text"] for segment in result["segments"]])
            print(f"Transcription: {transcription}")
            return transcription.strip()
        else:
            return ""
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return ""


async def generate_response(text: str, conversation_history: List[Segment]) -> str:
    """Generate a contextual response based on the transcribed text"""
    # Simple response logic - can be replaced with a more sophisticated LLM in the future
    responses = {
        "hello": "Hello there! How are you doing today?",
        "how are you": "I'm doing well, thanks for asking! How about you?",
        "what is your name": "I'm Sesame, your voice assistant. How can I help you?",
        "bye": "Goodbye! It was nice chatting with you.",
        "thank you": "You're welcome! Is there anything else I can help with?",
        "weather": "I don't have real-time weather data, but I hope it's nice where you are!",
        "help": "I can chat with you using natural voice. Just speak normally and I'll respond.",
    }
    
    text_lower = text.lower()
    
    # Check for matching keywords
    for key, response in responses.items():
        if key in text_lower:
            return response
    
    # Default responses based on text length
    if not text:
        return "I didn't catch that. Could you please repeat?"
    elif len(text) < 10:
        return "Thanks for your message. Could you elaborate a bit more?"
    else:
        return f"I understand you said '{text}'. That's interesting! Can you tell me more about that?"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    context_segments = []  # Store conversation context
    streaming_buffer = []  # Buffer for streaming audio chunks
    is_streaming = False
    
    # Variables for silence detection
    last_active_time = time.time()
    is_silence = False
    energy_window = deque(maxlen=10)  # For tracking recent audio energy
    
    try:
        while True:
            # Receive JSON data from client
            data = await websocket.receive_text()
            request = json.loads(data)
            
            action = request.get("action")
            
            if action == "generate":
                try:
                    text = request.get("text", "")
                    speaker_id = request.get("speaker", 0)
                    
                    # Generate audio response
                    print(f"Generating audio for: '{text}' with speaker {speaker_id}")
                    audio_tensor = generator.generate(
                        text=text,
                        speaker=speaker_id,
                        context=context_segments,
                        max_audio_length_ms=10_000,
                    )
                    
                    # Add to conversation context
                    context_segments.append(Segment(text=text, speaker=speaker_id, audio=audio_tensor))
                    
                    # Convert audio to base64 and send back to client
                    audio_base64 = await encode_audio_data(audio_tensor)
                    await websocket.send_json({
                        "type": "audio_response",
                        "audio": audio_base64
                    })
                except Exception as e:
                    print(f"Error generating audio: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error generating audio: {str(e)}"
                    })
                
            elif action == "add_to_context":
                try:
                    text = request.get("text", "")
                    speaker_id = request.get("speaker", 0)
                    audio_data = request.get("audio", "")
                    
                    # Convert received audio to tensor
                    audio_tensor = await decode_audio_data(audio_data)
                    
                    # Add to conversation context
                    context_segments.append(Segment(text=text, speaker=speaker_id, audio=audio_tensor))
                    
                    await websocket.send_json({
                        "type": "context_updated",
                        "message": "Audio added to context"
                    })
                except Exception as e:
                    print(f"Error adding to context: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing audio: {str(e)}"
                    })
                
            elif action == "clear_context":
                context_segments = []
                await websocket.send_json({
                    "type": "context_updated",
                    "message": "Context cleared"
                })
            
            elif action == "stream_audio":
                try:
                    speaker_id = request.get("speaker", 0)
                    audio_data = request.get("audio", "")
                    
                    # Convert received audio to tensor
                    audio_chunk = await decode_audio_data(audio_data)
                    
                    # Start streaming mode if not already started
                    if not is_streaming:
                        is_streaming = True
                        streaming_buffer = []
                        energy_window.clear()
                        is_silence = False
                        last_active_time = time.time()
                        print(f"Streaming started with speaker ID: {speaker_id}")
                        await websocket.send_json({
                            "type": "streaming_status",
                            "status": "started"
                        })
                    
                    # Calculate audio energy for silence detection
                    chunk_energy = torch.mean(torch.abs(audio_chunk)).item()
                    energy_window.append(chunk_energy)
                    avg_energy = sum(energy_window) / len(energy_window)
                    
                    # Debug audio levels
                    if len(energy_window) >= 5:  # Only start printing after we have enough samples
                        if avg_energy > SILENCE_THRESHOLD:
                            print(f"[AUDIO] Active sound detected - Energy: {avg_energy:.6f} (threshold: {SILENCE_THRESHOLD})")
                        else:
                            print(f"[AUDIO] Silence detected - Energy: {avg_energy:.6f} (threshold: {SILENCE_THRESHOLD})")
                    
                    # Check if audio is silent
                    current_silence = avg_energy < SILENCE_THRESHOLD
                    
                    # Track silence transition
                    if not is_silence and current_silence:
                        # Transition to silence
                        is_silence = True
                        last_active_time = time.time()
                        print("[STREAM] Transition to silence detected")
                    elif is_silence and not current_silence:
                        # User started talking again
                        is_silence = False
                        print("[STREAM] User resumed speaking")
                    
                    # Add chunk to buffer regardless of silence state
                    streaming_buffer.append(audio_chunk)
                    
                    # Debug buffer size periodically
                    if len(streaming_buffer) % 10 == 0:
                        print(f"[BUFFER] Current size: {len(streaming_buffer)} chunks, ~{len(streaming_buffer)/5:.1f} seconds")
                        
                    # Check if silence has persisted long enough to consider "stopped talking"
                    silence_elapsed = time.time() - last_active_time
                    
                    if is_silence and silence_elapsed >= SILENCE_DURATION_SEC and len(streaming_buffer) > 0:
                        # User has stopped talking - process the collected audio
                        print(f"[STREAM] Processing audio after {silence_elapsed:.2f}s of silence")
                        print(f"[STREAM] Processing {len(streaming_buffer)} audio chunks (~{len(streaming_buffer)/5:.1f} seconds)")
                        
                        full_audio = torch.cat(streaming_buffer, dim=0)
                        
                        # Log audio statistics
                        audio_duration = len(full_audio) / generator.sample_rate
                        audio_min = torch.min(full_audio).item()
                        audio_max = torch.max(full_audio).item()
                        audio_mean = torch.mean(full_audio).item()
                        print(f"[AUDIO] Processed audio - Duration: {audio_duration:.2f}s, Min: {audio_min:.4f}, Max: {audio_max:.4f}, Mean: {audio_mean:.4f}")
                        
                        # Process with WhisperX speech-to-text
                        print("[ASR] Starting transcription with WhisperX...")
                        transcribed_text = await transcribe_audio(full_audio)
                        
                        # Log the transcription
                        print(f"[ASR] Transcribed text: '{transcribed_text}'")
                        
                        # Add to conversation context
                        if transcribed_text:
                            print(f"[DIALOG] Adding user utterance to context: '{transcribed_text}'")
                            user_segment = Segment(text=transcribed_text, speaker=speaker_id, audio=full_audio)
                            context_segments.append(user_segment)
                            
                            # Generate a contextual response
                            print("[DIALOG] Generating response...")
                            response_text = await generate_response(transcribed_text, context_segments)
                            print(f"[DIALOG] Response text: '{response_text}'")
                            
                            # Send the transcribed text to client
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcribed_text
                            })
                            
                            # Generate audio for the response
                            print("[TTS] Generating speech for response...")
                            audio_tensor = generator.generate(
                                text=response_text,
                                speaker=1 if speaker_id == 0 else 0,  # Use opposite speaker
                                context=context_segments,
                                max_audio_length_ms=10_000,
                            )
                            print(f"[TTS] Generated audio length: {len(audio_tensor)/generator.sample_rate:.2f}s")
                            
                            # Add response to context
                            ai_segment = Segment(
                                text=response_text, 
                                speaker=1 if speaker_id == 0 else 0, 
                                audio=audio_tensor
                            )
                            context_segments.append(ai_segment)
                            print(f"[DIALOG] Context now has {len(context_segments)} segments")
                            
                            # Convert audio to base64 and send back to client
                            audio_base64 = await encode_audio_data(audio_tensor)
                            print("[STREAM] Sending audio response to client")
                            await websocket.send_json({
                                "type": "audio_response",
                                "text": response_text,
                                "audio": audio_base64
                            })
                        else:
                            print("[ASR] Transcription failed or returned empty text")
                            # If transcription failed, send a generic response
                            await websocket.send_json({
                                "type": "error",
                                "message": "Sorry, I couldn't understand what you said. Could you try again?"
                            })
                        
                        # Clear buffer and reset silence detection
                        streaming_buffer = []
                        energy_window.clear()
                        is_silence = False
                        last_active_time = time.time()
                        print("[STREAM] Buffer cleared, ready for next utterance")
                    
                    # If buffer gets too large without silence, process it anyway
                    # This prevents memory issues with very long streams
                    elif len(streaming_buffer) >= 30:  # ~6 seconds of audio at 5 chunks/sec
                        print("[BUFFER] Maximum buffer size reached, processing audio")
                        full_audio = torch.cat(streaming_buffer, dim=0)
                        
                        # Process with WhisperX speech-to-text
                        print("[ASR] Starting forced transcription of long audio...")
                        transcribed_text = await transcribe_audio(full_audio)
                        
                        if transcribed_text:
                            print(f"[ASR] Transcribed long audio: '{transcribed_text}'")
                            context_segments.append(Segment(text=transcribed_text, speaker=speaker_id, audio=full_audio))
                            
                            # Send the transcribed text to client
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcribed_text + " (processing continued speech...)"
                            })
                        else:
                            print("[ASR] No transcription from long audio")
                        
                        streaming_buffer = []
                        print("[BUFFER] Buffer cleared due to size limit")
                        
                except Exception as e:
                    print(f"[ERROR] Processing streaming audio: {str(e)}")
                    # Print traceback for more detailed error information
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing streaming audio: {str(e)}"
                    })
            
            elif action == "stop_streaming":
                is_streaming = False
                if streaming_buffer and len(streaming_buffer) > 5:  # Only process if there's meaningful audio
                    # Process any remaining audio in the buffer
                    full_audio = torch.cat(streaming_buffer, dim=0)
                    
                    # Process with WhisperX speech-to-text
                    transcribed_text = await transcribe_audio(full_audio)
                    
                    if transcribed_text:
                        context_segments.append(Segment(text=transcribed_text, speaker=request.get("speaker", 0), audio=full_audio))
                        
                        # Send the transcribed text to client
                        await websocket.send_json({
                            "type": "transcription",
                            "text": transcribed_text
                        })
                
                streaming_buffer = []
                await websocket.send_json({
                    "type": "streaming_status",
                    "status": "stopped"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
        manager.disconnect(websocket)

# Update the __main__ block with a comprehensive server startup message
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🔊 Sesame AI Voice Chat Server")
    print(f"{'='*60}")
    print(f"📡 Server Information:")
    print(f"   - Local URL: http://localhost:8000")
    print(f"   - Network URL: http://<your-ip-address>:8000")
    print(f"   - WebSocket: ws://<your-ip-address>:8000/ws")
    print(f"{'='*60}")
    print(f"💡 To make this server public:")
    print(f"   1. Ensure port 8000 is open in your firewall")
    print(f"   2. Set up port forwarding on your router to port 8000")
    print(f"   3. Or use a service like ngrok with: ngrok http 8000")
    print(f"{'='*60}")
    print(f"🌐 Device: {device.upper()}")
    print(f"🧠 Models loaded: Sesame CSM + WhisperX ({asr_model.device})")
    print(f"🔧 Serving from: {os.path.join(base_dir, 'index.html')}")
    print(f"{'='*60}")
    print(f"Ready to receive connections! Press Ctrl+C to stop the server.\n")
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)