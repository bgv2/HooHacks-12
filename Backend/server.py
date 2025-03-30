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
from flask import stream_with_context, Response
import time

# Configure environment with longer timeouts
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes timeout for downloads
requests.adapters.DEFAULT_TIMEOUT = 60  # Increase default requests timeout

# Create a models directory for caching
os.makedirs("models", exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

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
CHUNK_SIZE = 24000  # Number of audio samples per chunk (1 second at 24kHz)
audio_stream_queues = {}  # session_id -> queue for audio chunks

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
        'silence_start': None
    }
    emit('ready', {'message': 'Connection established'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    session_id = request.sid
    
    # Clean up resources
    if session_id in conversation_context:
        del conversation_context[session_id]
    
    if session_id in audio_stream_queues:
        del audio_stream_queues[session_id]

@socketio.on('start_speaking')
def handle_start_speaking():
    if request.sid in conversation_context:
        conversation_context[request.sid]['is_speaking'] = True
        conversation_context[request.sid]['audio_buffer'].clear()
        print(f"User {request.sid} started speaking")

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    if request.sid not in conversation_context:
        return
    
    context = conversation_context[request.sid]
    
    # Decode audio data
    audio_data = base64.b64decode(data['audio'])
    audio_numpy = np.frombuffer(audio_data, dtype=np.float32)
    audio_tensor = torch.tensor(audio_numpy)
    
    # Add to buffer
    context['audio_buffer'].append(audio_tensor)
    
    # Check for silence to detect end of speech
    if context['is_speaking'] and is_silence(audio_tensor):
        if context['silence_start'] is None:
            context['silence_start'] = time.time()
        elif time.time() - context['silence_start'] > 1.0:  # 1 second of silence
            # Process the complete utterance
            process_user_utterance(request.sid)
    else:
        context['silence_start'] = None

@socketio.on('stop_speaking')
def handle_stop_speaking():
    if request.sid in conversation_context:
        conversation_context[request.sid]['is_speaking'] = False
        process_user_utterance(request.sid)
        print(f"User {request.sid} stopped speaking")

def is_silence(audio_tensor, threshold=0.02):
    """Check if an audio chunk is silence based on amplitude threshold"""
    return torch.mean(torch.abs(audio_tensor)) < threshold

def process_user_utterance(session_id):
    """Process completed user utterance, generate response and stream audio back"""
    context = conversation_context[session_id]
    
    if not context['audio_buffer']:
        return
    
    # Combine audio chunks
    full_audio = torch.cat(list(context['audio_buffer']), dim=0)
    context['audio_buffer'].clear()
    context['is_speaking'] = False
    context['silence_start'] = None
    
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
            # Set up streaming queue for this session
            if session_id not in audio_stream_queues:
                audio_stream_queues[session_id] = queue.Queue()
            else:
                # Clear any existing items in the queue
                while not audio_stream_queues[session_id].empty():
                    audio_stream_queues[session_id].get()
            
            # Start audio generation in a separate thread to not block the server
            threading.Thread(
                target=generate_and_stream_audio,
                args=(bot_response, context['segments'], session_id),
                daemon=True
            ).start()
            
            # Initial response with text 
            emit('start_streaming_response', {'text': bot_response}, room=session_id)
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

def generate_audio_response(text, conversation_segments):
    """Generate audio response using CSM"""
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
        
        return audio
    except Exception as e:
        print(f"Error generating audio: {e}")
        # Return silence as fallback
        return torch.zeros(csm_generator.sample_rate * 3)  # 3 seconds of silence

def generate_and_stream_audio(text, conversation_segments, session_id):
    """Generate audio response using CSM and stream it in chunks"""
    try:
        # Use the last few conversation segments as context
        context_segments = conversation_segments[-4:] if len(conversation_segments) > 4 else conversation_segments
        
        # Generate full audio for bot response
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
        if session_id in conversation_context:
            conversation_context[session_id]['segments'].append(bot_segment)
        
        # Split audio into chunks for streaming
        chunk_size = CHUNK_SIZE
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            
            # Convert audio chunk to base64 for streaming
            audio_bytes = io.BytesIO()
            torchaudio.save(audio_bytes, chunk.unsqueeze(0).cpu(), csm_generator.sample_rate, format="wav")
            audio_bytes.seek(0)
            audio_b64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
            
            # Send the chunk to the client
            if session_id in audio_stream_queues:
                audio_stream_queues[session_id].put({
                    'audio': audio_b64,
                    'is_last': i + chunk_size >= len(audio)
                })
            else:
                # Session was disconnected before we finished generating
                break
                
        # Signal the end of streaming if queue still exists
        if session_id in audio_stream_queues:
            # Add an empty chunk as a sentinel to signal end of streaming
            audio_stream_queues[session_id].put(None)
    
    except Exception as e:
        print(f"Error generating or streaming audio: {e}")
        # Send error message to client
        if session_id in conversation_context:
            socketio.emit('error', {
                'message': f'Error generating audio: {str(e)}'
            }, room=session_id)
            
            # Send a final message to unblock the client
            if session_id in audio_stream_queues:
                audio_stream_queues[session_id].put(None)

@socketio.on('request_audio_chunk')
def handle_request_audio_chunk():
    """Send the next audio chunk in the queue to the client"""
    session_id = request.sid
    
    if session_id not in audio_stream_queues:
        emit('error', {'message': 'No audio stream available'})
        return
    
    # Get the next chunk or wait for it to be available
    try:
        if not audio_stream_queues[session_id].empty():
            chunk = audio_stream_queues[session_id].get(block=False)
            
            # If chunk is None, we're done streaming
            if chunk is None:
                emit('end_streaming')
                # Clean up the queue
                if session_id in audio_stream_queues:
                    del audio_stream_queues[session_id]
            else:
                emit('audio_chunk', chunk)
        else:
            # If the queue is empty but we're still generating, tell client to wait
            emit('wait_for_chunk')
    except Exception as e:
        print(f"Error sending audio chunk: {e}")
        emit('error', {'message': f'Error streaming audio: {str(e)}'})

if __name__ == '__main__':
    # Ensure the existing index.html file is in the correct location
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        os.rename('index.html', 'templates/index.html')
    
    # Load models asynchronously before starting the server
    print("Starting model loading...")
    load_models()
    
    # Start the server
    print("Starting Flask SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)