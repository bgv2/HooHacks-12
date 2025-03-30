import os
import io
import base64
import time
import torch
import torchaudio
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import queue
import requests
import huggingface_hub
from generator import load_csm_1b, Segment
from collections import deque
import json
import webrtcvad  # For voice activity detection

# Configure environment with longer timeouts
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes timeout for downloads
requests.adapters.DEFAULT_TIMEOUT = 60  # Increase default requests timeout

# Create a models directory for caching
os.makedirs("models", exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Explicitly check for CUDA and print detailed info
print("\n=== CUDA Information ===")
if torch.cuda.is_available():
    print(f"CUDA is available")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available")

# Check for cuDNN
try:
    import ctypes
    ctypes.CDLL("libcudnn_ops_infer.so.8")
    print("cuDNN is available")
except:
    print("cuDNN is not available (libcudnn_ops_infer.so.8 not found)")

# Determine compute device
try:
    if torch.cuda.is_available():
        device = "cuda"
        whisper_compute_type = "float16"
        print("🟢 CUDA is available and initialized successfully")
    elif torch.backends.mps.is_available():
        device = "mps"
        whisper_compute_type = "float32"
        print("🟢 MPS is available (Apple Silicon)")
    else:
        device = "cpu"
        whisper_compute_type = "int8"
        print("🟡 Using CPU (CUDA/MPS not available)")
except Exception as e:
    print(f"🔴 Error initializing CUDA: {e}")
    print("🔴 Falling back to CPU")
    device = "cpu"
    whisper_compute_type = "int8"

print(f"Using device: {device}")

# Initialize models with proper error handling
whisper_model = None
csm_generator = None
llm_model = None
llm_tokenizer = None
vad = None

# Constants
SAMPLE_RATE = 16000  # For VAD
VAD_FRAME_SIZE = 480  # 30ms at 16kHz for VAD
VAD_MODE = 3  # Aggressive mode for better results
AUDIO_CHUNK_SIZE = 2400  # 100ms chunks when streaming AI voice

# Audio sample rates
CLIENT_SAMPLE_RATE = 44100  # Browser WebAudio default
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16kHz

# Session data structures
user_sessions = {}  # session_id -> complete session data

# WebRTC ICE servers (STUN/TURN servers for NAT traversal)
ICE_SERVERS = [
    {"urls": "stun:stun.l.google.com:19302"},
    {"urls": "stun:stun1.l.google.com:19302"}
]

def load_models():
    """Load all necessary models"""
    global whisper_model, csm_generator, llm_model, llm_tokenizer, vad
    
    # Initialize Voice Activity Detector
    try:
        vad = webrtcvad.Vad(VAD_MODE)
        print("Voice Activity Detector initialized")
    except Exception as e:
        print(f"Error initializing VAD: {e}")
        vad = None
    
    # Initialize Faster-Whisper for transcription
    try:
        print("Loading Whisper model...")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("base", device=device, compute_type=whisper_compute_type, download_root="./models/whisper")
        print("Whisper model loaded successfully")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        print("Will use backup speech recognition method if available")
    
    # Initialize CSM model for audio generation
    try:
        print("Loading CSM model...")
        csm_generator = load_csm_1b(device=device)
        print("CSM model loaded successfully")
    except Exception as e:
        print(f"Error loading CSM model: {e}")
        print("Audio generation will not be available")
    
    # Initialize Llama 3.2 model for response generation
    try:
        print("Loading Llama 3.2 model...")
        llm_model_id = "meta-llama/Llama-3.2-1B"
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id, cache_dir="./models/llama")
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            torch_dtype=dtype,
            device_map=device,
            cache_dir="./models/llama",
            low_cpu_mem_usage=True
        )
        print("Llama 3.2 model loaded successfully")
    except Exception as e:
        print(f"Error loading Llama 3.2 model: {e}")
        print("Will use a fallback response generation method")

@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    session_id = request.sid
    print(f"Client connected: {session_id}")
    
    # Initialize session data
    user_sessions[session_id] = {
        # Conversation context
        'segments': [],
        'conversation_history': [],
        'is_turn_active': False,
        
        # Audio buffers and state
        'vad_buffer': deque(maxlen=30),  # ~1s of audio at 30fps
        'audio_buffer': bytearray(),
        'is_user_speaking': False,
        'last_vad_active': time.time(),
        'silence_duration': 0,
        'speech_frames': 0,
        
        # AI state
        'is_ai_speaking': False,
        'should_interrupt_ai': False,
        'ai_stream_queue': queue.Queue(),
        
        # WebRTC status
        'webrtc_connected': False,
        'webrtc_peer_id': None,
        
        # Processing flags
        'is_processing': False,
        'pending_user_audio': None
    }
    
    # Send config to client
    emit('session_ready', {
        'whisper_available': whisper_model is not None,
        'csm_available': csm_generator is not None,
        'llm_available': llm_model is not None,
        'client_sample_rate': CLIENT_SAMPLE_RATE,
        'server_sample_rate': getattr(csm_generator, 'sample_rate', 24000) if csm_generator else 24000,
        'ice_servers': ICE_SERVERS
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    session_id = request.sid
    print(f"Client disconnected: {session_id}")
    
    # Clean up resources
    if session_id in user_sessions:
        # Signal any running threads to stop
        user_sessions[session_id]['should_interrupt_ai'] = True
        
        # Clean up resources
        del user_sessions[session_id]

@socketio.on('webrtc_signal')
def handle_webrtc_signal(data):
    """Handle WebRTC signaling for P2P connection establishment"""
    session_id = request.sid
    if session_id not in user_sessions:
        return
    
    # Simply relay the signal to the client
    # In a multi-user app, we would route this to the correct peer
    emit('webrtc_signal', data)

@socketio.on('webrtc_connected')
def handle_webrtc_connected(data):
    """Client notifies that WebRTC connection is established"""
    session_id = request.sid
    if session_id not in user_sessions:
        return
    
    user_sessions[session_id]['webrtc_connected'] = True
    print(f"WebRTC connected for session {session_id}")
    emit('ready_for_speech', {'message': 'Ready to start conversation'})

@socketio.on('audio_stream')
def handle_audio_stream(data):
    """Process incoming audio stream packets from client"""
    session_id = request.sid
    if session_id not in user_sessions:
        return
    
    session = user_sessions[session_id]
    
    try:
        # Decode audio data
        audio_bytes = base64.b64decode(data.get('audio', ''))
        if not audio_bytes or len(audio_bytes) < 2:  # Need at least one sample
            return
        
        # Add to current audio buffer
        session['audio_buffer'] += audio_bytes
        
        # Check for speech using VAD
        has_speech = detect_speech(audio_bytes, session_id)
        
        # Handle speech state machine
        if has_speech:
            # Reset silence tracking when speech is detected
            session['last_vad_active'] = time.time()
            session['silence_duration'] = 0
            session['speech_frames'] += 1
            
            # If not already marked as speaking and we have enough speech frames
            if not session['is_user_speaking'] and session['speech_frames'] >= 5:
                on_speech_started(session_id)
        else:
            # No speech detected in this frame
            if session['is_user_speaking']:
                # Calculate silence duration
                now = time.time()
                session['silence_duration'] = now - session['last_vad_active']
                
                # If silent for more than 0.5 seconds, end speech segment
                if session['silence_duration'] > 0.8 and session['speech_frames'] > 8:
                    on_speech_ended(session_id)
            else:
                # Not speaking and no speech, just a silent frame
                session['speech_frames'] = max(0, session['speech_frames'] - 1)
    
    except Exception as e:
        print(f"Error processing audio stream: {e}")

def detect_speech(audio_bytes, session_id):
    """Use VAD to check if audio contains speech"""
    if session_id not in user_sessions:
        return False
        
    session = user_sessions[session_id]
    
    # Store in VAD buffer for history
    session['vad_buffer'].append(audio_bytes)
    
    if vad is None:
        # Fallback to simple energy detection
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        energy = np.mean(np.abs(audio_data)) / 32768.0
        return energy > 0.015  # Simple threshold
    
    try:
        # Ensure we have the right amount of data for VAD
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # If we have too much data, use just the right amount
        if len(audio_data) >= VAD_FRAME_SIZE:
            frame = audio_data[:VAD_FRAME_SIZE].tobytes()
            return vad.is_speech(frame, SAMPLE_RATE)
        
        # If too little data, accumulate in the VAD buffer and check periodically
        if len(session['vad_buffer']) >= 3:
            # Combine recent chunks to get enough data
            combined = bytearray()
            for chunk in list(session['vad_buffer'])[-3:]:
                combined.extend(chunk)
            
            # Extract the right amount of data
            if len(combined) >= VAD_FRAME_SIZE:
                frame = combined[:VAD_FRAME_SIZE]
                return vad.is_speech(bytes(frame), SAMPLE_RATE)
        
        return False
    
    except Exception as e:
        print(f"VAD error: {e}")
        return False

def on_speech_started(session_id):
    """Handle start of user speech"""
    if session_id not in user_sessions:
        return
        
    session = user_sessions[session_id]
    
    # Reset audio buffer 
    session['audio_buffer'] = bytearray()
    session['is_user_speaking'] = True
    session['is_turn_active'] = True
    
    # If AI is speaking, we need to interrupt it
    if session['is_ai_speaking']:
        session['should_interrupt_ai'] = True
        emit('ai_interrupted_by_user', room=session_id)
    
    # Notify client that we detected speech
    emit('user_speech_start', room=session_id)

def on_speech_ended(session_id):
    """Handle end of user speech segment"""
    if session_id not in user_sessions:
        return
        
    session = user_sessions[session_id]
    
    # Mark as not speaking anymore
    session['is_user_speaking'] = False
    session['speech_frames'] = 0
    
    # If no audio or already processing, skip
    if len(session['audio_buffer']) < 4000 or session['is_processing']:  # At least 250ms of audio
        session['audio_buffer'] = bytearray()
        return
    
    # Mark as processing to prevent multiple processes
    session['is_processing'] = True
    
    # Create a copy of the audio buffer
    audio_copy = session['audio_buffer']
    session['audio_buffer'] = bytearray()
    
    # Convert audio to the format needed for processing
    try:
        # Convert to float32 between -1 and 1
        audio_np = np.frombuffer(audio_copy, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np)
        
        # Resample to Whisper's expected sample rate if necessary
        if CLIENT_SAMPLE_RATE != WHISPER_SAMPLE_RATE:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, 
                orig_freq=CLIENT_SAMPLE_RATE, 
                new_freq=WHISPER_SAMPLE_RATE
            )
        
        # Save as WAV for transcription
        temp_audio_path = f"temp_audio_{session_id}.wav"
        torchaudio.save(
            temp_audio_path, 
            audio_tensor.unsqueeze(0), 
            WHISPER_SAMPLE_RATE
        )
        
        # Start transcription and response process in a thread
        threading.Thread(
            target=process_user_utterance,
            args=(session_id, temp_audio_path, audio_tensor),
            daemon=True
        ).start()
        
        # Notify client that processing has started
        emit('processing_speech', room=session_id)
    
    except Exception as e:
        print(f"Error preparing audio: {e}")
        session['is_processing'] = False
        emit('error', {'message': f'Error processing audio: {str(e)}'}, room=session_id)

def process_user_utterance(session_id, audio_path, audio_tensor):
    """Process user utterance, transcribe and generate response"""
    if session_id not in user_sessions:
        return
    
    session = user_sessions[session_id]
    
    try:
        # Transcribe audio
        if whisper_model is not None:
            user_text = transcribe_with_whisper(audio_path)
        else:
            # Fallback to another transcription service
            user_text = transcribe_fallback(audio_path)
        
        # Clean up temp file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Check if we got meaningful text
        if not user_text or len(user_text.strip()) < 2:
            emit('no_speech_detected', room=session_id)
            session['is_processing'] = False
            return
        
        print(f"Transcribed: {user_text}")
        
        # Create user segment
        user_segment = Segment(
            text=user_text,
            speaker=0,  # User is speaker 0
            audio=audio_tensor
        )
        session['segments'].append(user_segment)
        
        # Update conversation history
        session['conversation_history'].append({
            'role': 'user',
            'text': user_text
        })
        
        # Send transcription to client
        emit('transcription', {'text': user_text}, room=session_id)
        
        # Generate AI response
        ai_response = generate_ai_response(user_text, session_id)
        
        # Send text response to client
        emit('ai_response_text', {'text': ai_response}, room=session_id)
        
        # Update conversation history
        session['conversation_history'].append({
            'role': 'assistant',
            'text': ai_response
        })
        
        # Generate voice response if CSM is available
        if csm_generator is not None:
            session['is_ai_speaking'] = True
            session['should_interrupt_ai'] = False
            
            # Begin streaming audio response
            threading.Thread(
                target=stream_ai_response,
                args=(ai_response, session_id),
                daemon=True
            ).start()
    
    except Exception as e:
        print(f"Error processing utterance: {e}")
        emit('error', {'message': f'Error: {str(e)}'}, room=session_id)
    
    finally:
        # Clear processing flag
        if session_id in user_sessions:
            session['is_processing'] = False

def transcribe_with_whisper(audio_path):
    """Transcribe audio using Faster-Whisper"""
    segments, info = whisper_model.transcribe(audio_path, beam_size=5)
    
    # Collect all text from segments
    user_text = ""
    for segment in segments:
        user_text += segment.text.strip() + " "
    
    return user_text.strip()

def transcribe_fallback(audio_path):
    """Fallback transcription using Google's speech recognition"""
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return ""
            except sr.RequestError:
                return "[Speech recognition service unavailable]"
    except ImportError:
        return "[Speech recognition not available]"

def generate_ai_response(user_text, session_id):
    """Generate text response using available LLM"""
    if session_id not in user_sessions:
        return "I'm sorry, your session has expired."
    
    session = user_sessions[session_id]
    
    if llm_model is not None and llm_tokenizer is not None:
        # Format conversation history for the LLM
        prompt = "You are a helpful, friendly voice assistant. Keep your responses brief and conversational.\n\n"
        
        # Add recent conversation history (last 6 turns maximum)
        for entry in session['conversation_history'][-6:]:
            if entry['role'] == 'user':
                prompt += f"User: {entry['text']}\n"
            else:
                prompt += f"Assistant: {entry['text']}\n"
        
        # Add current query if not already in history
        if not session['conversation_history'] or session['conversation_history'][-1]['role'] != 'user':
            prompt += f"User: {user_text}\n"
        
        prompt += "Assistant: "
        
        try:
            # Generate response
            inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
            output = llm_model.generate(
                inputs.input_ids, 
                max_new_tokens=100,  # Keep responses shorter for voice
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            response = llm_tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
        
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return fallback_response(user_text)
    else:
        return fallback_response(user_text)

def fallback_response(user_text):
    """Generate simple fallback responses when LLM is unavailable"""
    user_text_lower = user_text.lower()
    
    if "hello" in user_text_lower or "hi" in user_text_lower:
        return "Hello! How can I help you today?"
    
    elif "how are you" in user_text_lower:
        return "I'm doing well, thanks for asking! How about you?"
    
    elif "thank" in user_text_lower:
        return "You're welcome! Happy to help."
    
    elif "bye" in user_text_lower or "goodbye" in user_text_lower:
        return "Goodbye! Have a great day!"
    
    elif any(q in user_text_lower for q in ["what", "who", "where", "when", "why", "how"]):
        return "That's an interesting question. I wish I could provide a better answer in my current fallback mode."
        
    else:
        return "I see. Tell me more about that."

def stream_ai_response(text, session_id):
    """Generate and stream audio response in real-time chunks"""
    if session_id not in user_sessions:
        return
    
    session = user_sessions[session_id]
    
    try:
        # Signal start of AI speech
        emit('ai_speech_start', room=session_id)
        
        # Use the last few conversation segments as context (up to 4)
        context_segments = session['segments'][-4:] if len(session['segments']) > 4 else session['segments']
        
        # Generate audio for bot response
        audio = csm_generator.generate(
            text=text,
            speaker=1,  # Bot is speaker 1
            context=context_segments,
            max_audio_length_ms=10000,  # 10 seconds max
            temperature=0.9,
            topk=50
        )
        
        # Create and store bot segment
        bot_segment = Segment(
            text=text,
            speaker=1,
            audio=audio
        )
        
        if session_id in user_sessions:
            session['segments'].append(bot_segment)
        
        # Stream audio in small chunks for more responsive playback
        chunk_size = AUDIO_CHUNK_SIZE  # Size defined in constants
        
        for i in range(0, len(audio), chunk_size):
            # Check if we should stop (user interrupted)
            if session_id not in user_sessions or session['should_interrupt_ai']:
                print("AI speech interrupted")
                break
            
            # Get next chunk
            chunk = audio[i:i+chunk_size]
            
            # Convert audio chunk to base64 for streaming
            audio_bytes = io.BytesIO()
            torchaudio.save(audio_bytes, chunk.unsqueeze(0).cpu(), csm_generator.sample_rate, format="wav")
            audio_bytes.seek(0)
            audio_b64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
            
            # Send chunk to client
            socketio.emit('ai_speech_chunk', {
                'audio': audio_b64,
                'is_last': i + chunk_size >= len(audio)
            }, room=session_id)
            
            # Small sleep for more natural pacing
            time.sleep(0.06)  # Slight delay for smoother playback
        
        # Signal end of AI speech
        if session_id in user_sessions:
            session['is_ai_speaking'] = False
            session['is_turn_active'] = False  # End conversation turn
            socketio.emit('ai_speech_end', room=session_id)
    
    except Exception as e:
        print(f"Error streaming AI response: {e}")
        if session_id in user_sessions:
            session['is_ai_speaking'] = False
            session['is_turn_active'] = False
            socketio.emit('error', {'message': f'Error generating audio: {str(e)}'}, room=session_id)
            socketio.emit('ai_speech_end', room=session_id)

@socketio.on('interrupt_ai')
def handle_interrupt():
    """Handle explicit AI interruption request from client"""
    session_id = request.sid
    if session_id in user_sessions:
        user_sessions[session_id]['should_interrupt_ai'] = True
        emit('ai_interrupted', room=session_id)

@socketio.on('get_config')
def handle_get_config():
    """Send configuration to client"""
    session_id = request.sid
    if session_id in user_sessions:
        emit('config', {
            'client_sample_rate': CLIENT_SAMPLE_RATE,
            'server_sample_rate': getattr(csm_generator, 'sample_rate', 24000) if csm_generator else 24000,
            'whisper_available': whisper_model is not None,
            'csm_available': csm_generator is not None,
            'ice_servers': ICE_SERVERS
        })

if __name__ == '__main__':
    # Ensure the existing index.html file is in the correct location
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        os.rename('index.html', 'templates/index.html')
    
    # Load models before starting the server
    print("Starting model loading...")
    load_models()
    
    # Start the server
    print("Starting Flask SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)