import os
import base64
import json
import asyncio
import torch
import torchaudio
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generator import load_csm_1b, Segment
import uvicorn
import time
from collections import deque

# Select device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# Initialize the model
generator = load_csm_1b(device=device)

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                        await websocket.send_json({
                            "type": "streaming_status",
                            "status": "started"
                        })
                    
                    # Calculate audio energy for silence detection
                    chunk_energy = torch.mean(torch.abs(audio_chunk)).item()
                    energy_window.append(chunk_energy)
                    avg_energy = sum(energy_window) / len(energy_window)
                    
                    # Check if audio is silent
                    current_silence = avg_energy < SILENCE_THRESHOLD
                    
                    # Track silence transition
                    if not is_silence and current_silence:
                        # Transition to silence
                        is_silence = True
                        last_active_time = time.time()
                    elif is_silence and not current_silence:
                        # User started talking again
                        is_silence = False
                    
                    # Add chunk to buffer regardless of silence state
                    streaming_buffer.append(audio_chunk)
                    
                    # Check if silence has persisted long enough to consider "stopped talking"
                    silence_elapsed = time.time() - last_active_time
                    
                    if is_silence and silence_elapsed >= SILENCE_DURATION_SEC and len(streaming_buffer) > 0:
                        # User has stopped talking - process the collected audio
                        full_audio = torch.cat(streaming_buffer, dim=0)
                        
                        # Process with speech-to-text (you would need to implement this)
                        # For now, just use a placeholder text
                        text = f"User audio from speaker {speaker_id}"
                        
                        print(f"Detected end of speech, processing {len(streaming_buffer)} chunks")
                        
                        # Add to conversation context
                        context_segments.append(Segment(text=text, speaker=speaker_id, audio=full_audio))
                        
                        # Generate response
                        response_text = "This is a response to what you just said"
                        audio_tensor = generator.generate(
                            text=response_text,
                            speaker=1 if speaker_id == 0 else 0,  # Use opposite speaker
                            context=context_segments,
                            max_audio_length_ms=10_000,
                        )
                        
                        # Convert audio to base64 and send back to client
                        audio_base64 = await encode_audio_data(audio_tensor)
                        await websocket.send_json({
                            "type": "audio_response",
                            "audio": audio_base64
                        })
                        
                        # Clear buffer and reset silence detection
                        streaming_buffer = []
                        energy_window.clear()
                        is_silence = False
                        last_active_time = time.time()
                    
                    # If buffer gets too large without silence, process it anyway
                    # This prevents memory issues with very long streams
                    elif len(streaming_buffer) >= 30:  # ~6 seconds of audio at 5 chunks/sec
                        print("Buffer limit reached, processing audio")
                        full_audio = torch.cat(streaming_buffer, dim=0)
                        text = f"Continued speech from speaker {speaker_id}"
                        context_segments.append(Segment(text=text, speaker=speaker_id, audio=full_audio))
                        streaming_buffer = []
                        
                except Exception as e:
                    print(f"Error processing streaming audio: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing streaming audio: {str(e)}"
                    })
            
            elif action == "stop_streaming":
                is_streaming = False
                if streaming_buffer:
                    # Process any remaining audio in the buffer
                    full_audio = torch.cat(streaming_buffer, dim=0)
                    text = f"Final streaming audio from speaker {request.get('speaker', 0)}"
                    context_segments.append(Segment(text=text, speaker=request.get("speaker", 0), audio=full_audio))
                
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
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)