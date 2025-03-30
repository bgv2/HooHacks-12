import os
import base64
import json
import torch
import torchaudio
import numpy as np
import whisperx
from io import BytesIO
from typing import List, Dict, Any, Optional
from flask import Flask, request, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from generator import load_csm_1b, Segment
import time
import gc
from collections import deque
from threading import Lock

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

# Silence detection parameters
SILENCE_THRESHOLD = 0.01  # Adjust based on your audio normalization
SILENCE_DURATION_SEC = 1.0  # How long silence must persist

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")
os.makedirs(static_dir, exist_ok=True)

# Setup Flask
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Socket connection management
thread = None
thread_lock = Lock()
active_clients = {}  # Map client_id to client context

# Helper function to convert audio data
def decode_audio_data(audio_data: str) -> torch.Tensor:
    """Decode base64 audio data to a torch tensor"""
    try:
        # Skip empty audio data
        if not audio_data:
            print("Empty audio data received")
            return torch.zeros(generator.sample_rate // 2)  # 0.5 seconds of silence
            
        # Extract the actual base64 content
        if ',' in audio_data:
            audio_data = audio_data.split(',')[1]
            
        # Decode base64 audio data
        try:
            binary_data = base64.b64decode(audio_data)
            print(f"Decoded base64 data: {len(binary_data)} bytes")
        except Exception as e:
            print(f"Base64 decoding error: {str(e)}")
            return torch.zeros(generator.sample_rate // 2)
        
        # Debug: save the raw binary data to examine with external tools
        debug_path = os.path.join(base_dir, "debug_incoming.wav") 
        with open(debug_path, 'wb') as f:
            f.write(binary_data)
        print(f"Saved debug file to {debug_path}")
            
        # Load audio from binary data
        try:
            with BytesIO(binary_data) as temp_file:
                audio_tensor, sample_rate = torchaudio.load(temp_file, format="wav")
                print(f"Loaded audio: shape={audio_tensor.shape}, sample_rate={sample_rate}Hz")
                
                # Check if audio is valid
                if audio_tensor.numel() == 0 or torch.isnan(audio_tensor).any():
                    print("Warning: Empty or invalid audio data detected")
                    return torch.zeros(generator.sample_rate // 2)
        except Exception as e:
            print(f"Audio loading error: {str(e)}")
            # Try saving to a temporary file instead of loading from BytesIO
            try:
                temp_path = os.path.join(base_dir, "temp_incoming.wav")
                with open(temp_path, 'wb') as f:
                    f.write(binary_data)
                print(f"Trying to load from file: {temp_path}")
                audio_tensor, sample_rate = torchaudio.load(temp_path, format="wav")
                print(f"Loaded from file: shape={audio_tensor.shape}, sample_rate={sample_rate}Hz")
                os.remove(temp_path)
            except Exception as e2:
                print(f"Secondary audio loading error: {str(e2)}")
                return torch.zeros(generator.sample_rate // 2)
        
        # Resample if needed
        if sample_rate != generator.sample_rate:
            try:
                print(f"Resampling from {sample_rate}Hz to {generator.sample_rate}Hz")
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor.squeeze(0), 
                    orig_freq=sample_rate, 
                    new_freq=generator.sample_rate
                )
                print(f"Resampled audio shape: {audio_tensor.shape}")
            except Exception as e:
                print(f"Resampling error: {str(e)}")
                return torch.zeros(generator.sample_rate // 2)
        else:
            audio_tensor = audio_tensor.squeeze(0)
            
        print(f"Final audio tensor shape: {audio_tensor.shape}")
        return audio_tensor
    except Exception as e:
        print(f"Error decoding audio: {str(e)}")
        # Return a small silent audio segment as fallback
        return torch.zeros(generator.sample_rate // 2)  # 0.5 seconds of silence


def encode_audio_data(audio_tensor: torch.Tensor) -> str:
    """Encode torch tensor audio to base64 string"""
    buf = BytesIO()
    torchaudio.save(buf, audio_tensor.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
    buf.seek(0)
    audio_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:audio/wav;base64,{audio_base64}"


def transcribe_audio(audio_tensor: torch.Tensor) -> str:
    """Transcribe audio using WhisperX"""
    try:
        # Save the tensor to a temporary file
        temp_path = os.path.join(base_dir, "temp_audio.wav")
        torchaudio.save(temp_path, audio_tensor.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Load and transcribe the audio
        audio = whisperx.load_audio(temp_path)
        result = asr_model.transcribe(audio, batch_size=16)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Get the transcription text
        if result["segments"] and len(result["segments"]) > 0:
            # Combine all segments
            transcription = " ".join([segment["text"] for segment in result["segments"]])
            return transcription.strip()
        else:
            return ""
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        return ""


def generate_response(text: str, conversation_history: List[Segment]) -> str:
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

# Flask routes for serving static content
@app.route('/')
def index():
    return send_from_directory(base_dir, 'index.html')

@app.route('/favicon.ico')
def favicon():
    if os.path.exists(os.path.join(static_dir, 'favicon.ico')):
        return send_from_directory(static_dir, 'favicon.ico')
    return Response(status=204)

@app.route('/voice-chat.js')
def voice_chat_js():
    return send_from_directory(base_dir, 'voice-chat.js')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(static_dir, path)

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    print(f"Client connected: {client_id}")
    
    # Initialize client context
    active_clients[client_id] = {
        'context_segments': [],
        'streaming_buffer': [],
        'is_streaming': False,
        'is_silence': False,
        'last_active_time': time.time(),
        'energy_window': deque(maxlen=10)
    }
    
    emit('status', {'type': 'connected', 'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    if client_id in active_clients:
        del active_clients[client_id]
    print(f"Client disconnected: {client_id}")

@socketio.on('generate')
def handle_generate(data):
    client_id = request.sid
    if client_id not in active_clients:
        emit('error', {'message': 'Client not registered'})
        return
    
    try:
        text = data.get('text', '')
        speaker_id = data.get('speaker', 0)
        
        print(f"Generating audio for: '{text}' with speaker {speaker_id}")
        
        # Generate audio response
        audio_tensor = generator.generate(
            text=text,
            speaker=speaker_id,
            context=active_clients[client_id]['context_segments'],
            max_audio_length_ms=10_000,
        )
        
        # Add to conversation context
        active_clients[client_id]['context_segments'].append(
            Segment(text=text, speaker=speaker_id, audio=audio_tensor)
        )
        
        # Convert audio to base64 and send back to client
        audio_base64 = encode_audio_data(audio_tensor)
        emit('audio_response', {
            'type': 'audio_response',
            'audio': audio_base64
        })
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        emit('error', {
            'type': 'error',
            'message': f"Error generating audio: {str(e)}"
        })

@socketio.on('add_to_context')
def handle_add_to_context(data):
    client_id = request.sid
    if client_id not in active_clients:
        emit('error', {'message': 'Client not registered'})
        return
    
    try:
        text = data.get('text', '')
        speaker_id = data.get('speaker', 0)
        audio_data = data.get('audio', '')
        
        # Convert received audio to tensor
        audio_tensor = decode_audio_data(audio_data)
        
        # Add to conversation context
        active_clients[client_id]['context_segments'].append(
            Segment(text=text, speaker=speaker_id, audio=audio_tensor)
        )
        
        emit('context_updated', {
            'type': 'context_updated',
            'message': 'Audio added to context'
        })
        
    except Exception as e:
        print(f"Error adding to context: {str(e)}")
        emit('error', {
            'type': 'error',
            'message': f"Error processing audio: {str(e)}"
        })

@socketio.on('clear_context')
def handle_clear_context():
    client_id = request.sid
    if client_id in active_clients:
        active_clients[client_id]['context_segments'] = []
        
    emit('context_updated', {
        'type': 'context_updated',
        'message': 'Context cleared'
    })

@socketio.on('stream_audio')
def handle_stream_audio(data):
    client_id = request.sid
    if client_id not in active_clients:
        emit('error', {'message': 'Client not registered'})
        return
    
    client = active_clients[client_id]
    
    try:
        speaker_id = data.get('speaker', 0)
        audio_data = data.get('audio', '')
        
        # Convert received audio to tensor
        audio_chunk = decode_audio_data(audio_data)
        
        # Start streaming mode if not already started
        if not client['is_streaming']:
            client['is_streaming'] = True
            client['streaming_buffer'] = []
            client['energy_window'].clear()
            client['is_silence'] = False
            client['last_active_time'] = time.time()
            print(f"[{client_id}] Streaming started with speaker ID: {speaker_id}")
            emit('streaming_status', {
                'type': 'streaming_status',
                'status': 'started'
            })
        
        # Calculate audio energy for silence detection
        chunk_energy = torch.mean(torch.abs(audio_chunk)).item()
        client['energy_window'].append(chunk_energy)
        avg_energy = sum(client['energy_window']) / len(client['energy_window'])
        
        # Check if audio is silent
        current_silence = avg_energy < SILENCE_THRESHOLD
        
        # Track silence transition
        if not client['is_silence'] and current_silence:
            # Transition to silence
            client['is_silence'] = True
            client['last_active_time'] = time.time()
        elif client['is_silence'] and not current_silence:
            # User started talking again
            client['is_silence'] = False
        
        # Add chunk to buffer regardless of silence state
        client['streaming_buffer'].append(audio_chunk)
            
        # Check if silence has persisted long enough to consider "stopped talking"
        silence_elapsed = time.time() - client['last_active_time']
        
        if client['is_silence'] and silence_elapsed >= SILENCE_DURATION_SEC and len(client['streaming_buffer']) > 0:
            # User has stopped talking - process the collected audio
            print(f"[{client_id}] Processing audio after {silence_elapsed:.2f}s of silence")
            
            full_audio = torch.cat(client['streaming_buffer'], dim=0)
            
            # Process with WhisperX speech-to-text
            print(f"[{client_id}] Starting transcription with WhisperX...")
            transcribed_text = transcribe_audio(full_audio)
            
            # Log the transcription
            print(f"[{client_id}] Transcribed text: '{transcribed_text}'")
            
            # Add to conversation context
            if transcribed_text:
                user_segment = Segment(text=transcribed_text, speaker=speaker_id, audio=full_audio)
                client['context_segments'].append(user_segment)
                
                # Generate a contextual response
                response_text = generate_response(transcribed_text, client['context_segments'])
                
                # Send the transcribed text to client
                emit('transcription', {
                    'type': 'transcription',
                    'text': transcribed_text
                })
                
                # Generate audio for the response
                audio_tensor = generator.generate(
                    text=response_text,
                    speaker=1 if speaker_id == 0 else 0,  # Use opposite speaker
                    context=client['context_segments'],
                    max_audio_length_ms=10_000,
                )
                
                # Add response to context
                ai_segment = Segment(
                    text=response_text, 
                    speaker=1 if speaker_id == 0 else 0, 
                    audio=audio_tensor
                )
                client['context_segments'].append(ai_segment)
                
                # Convert audio to base64 and send back to client
                audio_base64 = encode_audio_data(audio_tensor)
                emit('audio_response', {
                    'type': 'audio_response',
                    'text': response_text,
                    'audio': audio_base64
                })
            else:
                # If transcription failed, send a generic response
                emit('error', {
                    'type': 'error',
                    'message': "Sorry, I couldn't understand what you said. Could you try again?"
                })
            
            # Clear buffer and reset silence detection
            client['streaming_buffer'] = []
            client['energy_window'].clear()
            client['is_silence'] = False
            client['last_active_time'] = time.time()
        
        # If buffer gets too large without silence, process it anyway
        elif len(client['streaming_buffer']) >= 30:  # ~6 seconds of audio at 5 chunks/sec
            full_audio = torch.cat(client['streaming_buffer'], dim=0)
            
            # Process with WhisperX speech-to-text
            transcribed_text = transcribe_audio(full_audio)
            
            if transcribed_text:
                client['context_segments'].append(
                    Segment(text=transcribed_text, speaker=speaker_id, audio=full_audio)
                )
                
                # Send the transcribed text to client
                emit('transcription', {
                    'type': 'transcription',
                    'text': transcribed_text + " (processing continued speech...)"
                })
            
            client['streaming_buffer'] = []
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing streaming audio: {str(e)}")
        emit('error', {
            'type': 'error',
            'message': f"Error processing streaming audio: {str(e)}"
        })

@socketio.on('stop_streaming')
def handle_stop_streaming(data):
    client_id = request.sid
    if client_id not in active_clients:
        return
    
    client = active_clients[client_id]
    client['is_streaming'] = False
    
    if client['streaming_buffer'] and len(client['streaming_buffer']) > 5:
        # Process any remaining audio in the buffer
        full_audio = torch.cat(client['streaming_buffer'], dim=0)
        
        # Process with WhisperX speech-to-text
        transcribed_text = transcribe_audio(full_audio)
        
        if transcribed_text:
            client['context_segments'].append(
                Segment(text=transcribed_text, speaker=data.get("speaker", 0), audio=full_audio)
            )
            
            # Send the transcribed text to client
            emit('transcription', {
                'type': 'transcription',
                'text': transcribed_text
            })
    
    client['streaming_buffer'] = []
    emit('streaming_status', {
        'type': 'streaming_status',
        'status': 'stopped'
    })

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🔊 Sesame AI Voice Chat Server (Flask Implementation)")
    print(f"{'='*60}")
    print(f"📡 Server Information:")
    print(f"   - Local URL: http://localhost:5000")
    print(f"   - Network URL: http://<your-ip-address>:5000")
    print(f"   - WebSocket: ws://<your-ip-address>:5000/socket.io")
    print(f"{'='*60}")
    print(f"💡 To make this server public:")
    print(f"   1. Ensure port 5000 is open in your firewall")
    print(f"   2. Set up port forwarding on your router to port 5000")
    print(f"   3. Or use a service like ngrok with: ngrok http 5000")
    print(f"{'='*60}")
    print(f"🌐 Device: {device.upper()}")
    print(f"🧠 Models loaded: Sesame CSM + WhisperX ({asr_model.device})")
    print(f"🔧 Serving from: {os.path.join(base_dir, 'index.html')}")
    print(f"{'='*60}")
    print(f"Ready to receive connections! Press Ctrl+C to stop the server.\n")
    
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)