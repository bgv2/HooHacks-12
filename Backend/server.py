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
from collections import deque
import requests
import huggingface_hub
from generator import load_csm_1b, Segment
import threading
import queue
import asyncio
import json

# Configure environment with longer timeouts
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes timeout for downloads
requests.adapters.DEFAULT_TIMEOUT = 60  # Increase default requests timeout

# Create a models directory for caching
os.makedirs("models", exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Explicitly check for CUDA and print more detailed info
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

# Check for other compute platforms
if torch.backends.mps.is_available():
    print("MPS (Apple Silicon) is available")
else:
    print("MPS is not available")
print("========================\n")

# Check for CUDA availability and handle potential CUDA/cuDNN issues
try:
    if torch.cuda.is_available():
        # Try to initialize CUDA to check if libraries are properly loaded
        _ = torch.zeros(1).cuda()
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

def load_models():
    global whisper_model, csm_generator, llm_model, llm_tokenizer
    
    # Initialize Faster-Whisper for transcription
    try:
        print("Loading Whisper model...")
        # Import here to avoid immediate import errors if package is missing
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
        llm_model_id = "meta-llama/Llama-3.2-1B"  # Choose appropriate size based on resources
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id, cache_dir="./models/llama")
        # Use the right data type based on device
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

# Store conversation context
conversation_context = {}  # session_id -> context
active_audio_streams = {}  # session_id -> stream status

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    conversation_context[request.sid] = {
        'segments': [],
        'speakers': [0, 1],  # 0 = user, 1 = bot
        'audio_buffer': deque(maxlen=10),  # Store recent audio chunks
        'is_speaking': False,
        'last_activity': time.time(),
        'active_session': True,
        'transcription_buffer': []  # For real-time transcription
    }
    emit('ready', {
        'message': 'Connection established',
        'sample_rate': getattr(csm_generator, 'sample_rate', 24000) if csm_generator else 24000
    })

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    session_id = request.sid
    
    # Clean up resources
    if session_id in conversation_context:
        conversation_context[session_id]['active_session'] = False
        del conversation_context[session_id]
    
    if session_id in active_audio_streams:
        active_audio_streams[session_id]['active'] = False
        del active_audio_streams[session_id]

@socketio.on('audio_stream')
def handle_audio_stream(data):
    """Handle incoming audio stream from client"""
    session_id = request.sid
    
    if session_id not in conversation_context:
        return
    
    context = conversation_context[session_id]
    context['last_activity'] = time.time()
    
    # Process different stream events
    if data.get('event') == 'start':
        # Client is starting to send audio
        context['is_speaking'] = True
        context['audio_buffer'].clear()
        context['transcription_buffer'] = []
        print(f"User {session_id} started streaming audio")
        
        # If AI was speaking, interrupt it
        if session_id in active_audio_streams and active_audio_streams[session_id]['active']:
            active_audio_streams[session_id]['active'] = False
            emit('ai_stream_interrupt', {}, room=session_id)
    
    elif data.get('event') == 'data':
        # Audio data received
        if not context['is_speaking']:
            return
            
        # Decode audio chunk
        try:
            audio_data = base64.b64decode(data.get('audio', ''))
            if not audio_data:
                return
                
            audio_numpy = np.frombuffer(audio_data, dtype=np.float32)
            
            # Apply a simple noise gate
            if np.mean(np.abs(audio_numpy)) < 0.01:  # Very quiet
                return
                
            audio_tensor = torch.tensor(audio_numpy)
            
            # Add to audio buffer
            context['audio_buffer'].append(audio_tensor)
            
            # Real-time transcription (periodic)
            if len(context['audio_buffer']) % 3 == 0:  # Process every 3 chunks
                threading.Thread(
                    target=process_realtime_transcription,
                    args=(session_id,),
                    daemon=True
                ).start()
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
    
    elif data.get('event') == 'end':
        # Client has finished sending audio
        context['is_speaking'] = False
        
        if len(context['audio_buffer']) > 0:
            # Process the complete utterance
            threading.Thread(
                target=process_complete_utterance,
                args=(session_id,),
                daemon=True
            ).start()
        
        print(f"User {session_id} stopped streaming audio")

def process_realtime_transcription(session_id):
    """Process incoming audio for real-time transcription"""
    if session_id not in conversation_context or not conversation_context[session_id]['active_session']:
        return
        
    context = conversation_context[session_id]
    
    if not context['audio_buffer'] or not context['is_speaking']:
        return
    
    try:
        # Combine current buffer for transcription
        buffer_copy = list(context['audio_buffer'])
        if not buffer_copy:
            return
            
        full_audio = torch.cat(buffer_copy, dim=0)
        
        # Save audio to temporary WAV file for transcription
        temp_audio_path = f"temp_rt_{session_id}.wav"
        torchaudio.save(
            temp_audio_path, 
            full_audio.unsqueeze(0), 
            44100  # Assuming 44.1kHz from client
        )
        
        # Transcribe with Whisper if available
        if whisper_model is not None:
            segments, _ = whisper_model.transcribe(temp_audio_path, beam_size=5)
            text = " ".join([segment.text for segment in segments])
            
            if text.strip():
                context['transcription_buffer'].append(text)
                # Send partial transcription to client
                emit('partial_transcription', {'text': text}, room=session_id)
    except Exception as e:
        print(f"Error in realtime transcription: {e}")
    finally:
        # Clean up
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def process_complete_utterance(session_id):
    """Process completed user utterance, generate response and stream audio back"""
    if session_id not in conversation_context or not conversation_context[session_id]['active_session']:
        return
    
    context = conversation_context[session_id]
    
    if not context['audio_buffer']:
        return
    
    # Combine audio chunks
    full_audio = torch.cat(list(context['audio_buffer']), dim=0)
    context['audio_buffer'].clear()
    
    # Save audio to temporary WAV file for transcription
    temp_audio_path = f"temp_audio_{session_id}.wav"
    torchaudio.save(
        temp_audio_path, 
        full_audio.unsqueeze(0), 
        44100  # Assuming 44.1kHz from client
    )
    
    try:
        # Try using Whisper first if available
        if whisper_model is not None:
            user_text = transcribe_with_whisper(temp_audio_path)
        else:
            # Fallback to Google's speech recognition
            user_text = transcribe_with_google(temp_audio_path)
        
        if not user_text:
            print("No speech detected.")
            emit('error', {'message': 'No speech detected. Please try again.'}, room=session_id)
            return
            
        print(f"Transcribed: {user_text}")
        
        # Add to conversation segments
        user_segment = Segment(
            text=user_text,
            speaker=0,  # User is speaker 0
            audio=full_audio
        )
        context['segments'].append(user_segment)
        
        # Generate bot response text
        bot_response = generate_llm_response(user_text, context['segments'])
        print(f"Bot response: {bot_response}")
        
        # Send transcribed text to client
        emit('transcription', {'text': user_text}, room=session_id)
        
        # Generate and stream audio response if CSM is available
        if csm_generator is not None:
            # Create stream state object
            active_audio_streams[session_id] = {
                'active': True,
                'text': bot_response
            }
            
            # Send initial response to prepare client
            emit('ai_stream_start', {
                'text': bot_response
            }, room=session_id)
            
            # Start audio generation in a separate thread
            threading.Thread(
                target=generate_and_stream_audio_realtime,
                args=(bot_response, context['segments'], session_id),
                daemon=True
            ).start()
        else:
            # Send text-only response if audio generation isn't available
            emit('text_response', {'text': bot_response}, room=session_id)
            
            # Add text-only bot response to conversation history
            bot_segment = Segment(
                text=bot_response,
                speaker=1,  # Bot is speaker 1
                audio=torch.zeros(1)  # Placeholder empty audio
            )
            context['segments'].append(bot_segment)
        
    except Exception as e:
        print(f"Error processing speech: {e}")
        emit('error', {'message': f'Error processing speech: {str(e)}'}, room=session_id)
    finally:
        # Cleanup temp file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def transcribe_with_whisper(audio_path):
    """Transcribe audio using Faster-Whisper"""
    segments, info = whisper_model.transcribe(audio_path, beam_size=5)
    
    # Collect all text from segments
    user_text = ""
    for segment in segments:
        segment_text = segment.text.strip()
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment_text}")
        user_text += segment_text + " "

    print(f"Transcribed text: {user_text.strip()}")
    
    return user_text.strip()

def transcribe_with_google(audio_path):
    """Fallback transcription using Google's speech recognition"""
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
            # If Google API fails, try a basic energy-based VAD approach
            # This is a very basic fallback and won't give good results
            return "[Speech detected but transcription failed]"

def generate_llm_response(user_text, conversation_segments):
    """Generate text response using available model"""
    if llm_model is not None and llm_tokenizer is not None:
        # Format conversation history for the LLM
        conversation_history = ""
        for segment in conversation_segments[-5:]:  # Use last 5 utterances for context
            speaker_name = "User" if segment.speaker == 0 else "Assistant"
            conversation_history += f"{speaker_name}: {segment.text}\n"
        
        # Add the current user query
        conversation_history += f"User: {user_text}\nAssistant:"
        
        try:
            # Generate response
            inputs = llm_tokenizer(conversation_history, return_tensors="pt").to(device)
            output = llm_model.generate(
                inputs.input_ids, 
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            response = llm_tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"Error generating response with LLM: {e}")
            return fallback_response(user_text)
    else:
        return fallback_response(user_text)

def fallback_response(user_text):
    """Generate a simple fallback response when LLM is not available"""
    # Simple rule-based responses
    user_text_lower = user_text.lower()
    
    if "hello" in user_text_lower or "hi" in user_text_lower:
        return "Hello! I'm a simple fallback assistant. The main language model couldn't be loaded, so I have limited capabilities."
    
    elif "how are you" in user_text_lower:
        return "I'm functioning within my limited capabilities. How can I assist you today?"
    
    elif "thank" in user_text_lower:
        return "You're welcome! Let me know if there's anything else I can help with."
    
    elif "bye" in user_text_lower or "goodbye" in user_text_lower:
        return "Goodbye! Have a great day!"
    
    elif any(q in user_text_lower for q in ["what", "who", "where", "when", "why", "how"]):
        return "I'm running in fallback mode and can't answer complex questions. Please try again when the main language model is available."
        
    else:
        return "I understand you said something about that. Unfortunately, I'm running in fallback mode with limited capabilities. Please try again later when the main model is available."

def generate_and_stream_audio_realtime(text, conversation_segments, session_id):
    """Generate audio response using CSM and stream it in real-time to client"""
    if session_id not in active_audio_streams or not active_audio_streams[session_id]['active']:
        return
    
    try:
        # Use the last few conversation segments as context
        context_segments = conversation_segments[-4:] if len(conversation_segments) > 4 else conversation_segments
        
        # Generate audio for bot response
        audio = csm_generator.generate(
            text=text,
            speaker=1,  # Bot is speaker 1
            context=context_segments,
            max_audio_length_ms=10000,  # 10 seconds max
            temperature=0.9,
            topk=50
        )
        
        # Store the full audio for conversation history
        bot_segment = Segment(
            text=text,
            speaker=1,  # Bot is speaker 1
            audio=audio
        )
        if session_id in conversation_context and conversation_context[session_id]['active_session']:
            conversation_context[session_id]['segments'].append(bot_segment)
        
        # Stream audio in small chunks for more responsive playback
        chunk_size = 4800  # 200ms at 24kHz
        
        for i in range(0, len(audio), chunk_size):
            if session_id not in active_audio_streams or not active_audio_streams[session_id]['active']:
                print("Audio streaming interrupted or session ended")
                break
                
            chunk = audio[i:i+chunk_size]
            
            # Convert audio chunk to base64 for streaming
            audio_bytes = io.BytesIO()
            torchaudio.save(audio_bytes, chunk.unsqueeze(0).cpu(), csm_generator.sample_rate, format="wav")
            audio_bytes.seek(0)
            audio_b64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
            
            # Send chunk to client
            socketio.emit('ai_stream_data', {
                'audio': audio_b64,
                'is_last': i + chunk_size >= len(audio)
            }, room=session_id)
            
            # Simulate real-time speech by adding a small delay
            # Remove this in production for faster response
            time.sleep(0.15)  # Slight delay for more natural timing
        
        # Signal end of stream
        if session_id in active_audio_streams and active_audio_streams[session_id]['active']:
            socketio.emit('ai_stream_end', {}, room=session_id)
            active_audio_streams[session_id]['active'] = False
    
    except Exception as e:
        print(f"Error generating or streaming audio: {e}")
        # Send error message to client
        if session_id in conversation_context and conversation_context[session_id]['active_session']:
            socketio.emit('error', {
                'message': f'Error generating audio: {str(e)}'
            }, room=session_id)
            
        # Signal stream end to unblock client
        socketio.emit('ai_stream_end', {}, room=session_id)
        if session_id in active_audio_streams:
            active_audio_streams[session_id]['active'] = False

if __name__ == '__main__':
    # Ensure the existing index.html file is in the correct location
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        os.rename('index.html', 'templates/index.html')
    
    # Load models before starting the server
    print("Starting model loading...")
    load_models()
    
    # Start the server with eventlet for better WebSocket performance
    print("Starting Flask SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)