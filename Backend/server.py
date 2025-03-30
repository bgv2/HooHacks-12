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

# Force CPU mode regardless of what's available
# This bypasses the CUDA/cuDNN library requirements
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all CUDA devices
torch.backends.cudnn.enabled = False  # Disable cuDNN

# Configure environment with longer timeouts
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes timeout for downloads
requests.adapters.DEFAULT_TIMEOUT = 60  # Increase default requests timeout

# Create a models directory for caching
os.makedirs("models", exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Force CPU regardless of what hardware is available
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_compute_type = "int8"
print(f"Forcing CPU mode for all models")

# Initialize models with proper error handling
whisper_model = None
csm_generator = None
llm_model = None
llm_tokenizer = None

def load_models():
    global whisper_model, csm_generator, llm_model, llm_tokenizer
    
    # Initialize Faster-Whisper for transcription
    try:
        print("Loading Whisper model on CPU...")
        # Import here to avoid immediate import errors if package is missing
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8", download_root="./models/whisper")
        print("Whisper model loaded successfully")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        print("Will use backup speech recognition method if available")
    
    # Initialize CSM model for audio generation
    try:
        print("Loading CSM model on CPU...")
        csm_generator = load_csm_1b(device="cpu")
        print("CSM model loaded successfully")
    except Exception as e:
        print(f"Error loading CSM model: {e}")
        print("Audio generation will not be available")
    
    # Initialize Llama 3.2 model for response generation
    try:
        print("Loading Llama 3.2 model on CPU...")
        llm_model_id = "meta-llama/Llama-3.2-1B"  # Choose appropriate size based on resources
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id, cache_dir="./models/llama")
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            torch_dtype=torch.float32,  # Use float32 on CPU
            device_map="cpu",
            cache_dir="./models/llama",
            low_cpu_mem_usage=True
        )
        print("Llama 3.2 model loaded successfully")
    except Exception as e:
        print(f"Error loading Llama 3.2 model: {e}")
        print("Will use a fallback response generation method")

# Store conversation context
conversation_context = {}  # session_id -> context

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
    if request.sid in conversation_context:
        del conversation_context[request.sid]

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
    """Process completed user utterance, generate response and send audio back"""
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
        
        # Generate bot response
        bot_response = generate_llm_response(user_text, context['segments'])
        print(f"Bot response: {bot_response}")
        
        # Send transcribed text to client
        emit('transcription', {'text': user_text}, room=session_id)
        
        # Generate and send audio response if CSM is available
        if csm_generator is not None:
            # Convert to audio using CSM
            bot_audio = generate_audio_response(bot_response, context['segments'])
            
            # Convert audio to base64 for sending over websocket
            audio_bytes = io.BytesIO()
            torchaudio.save(audio_bytes, bot_audio.unsqueeze(0).cpu(), csm_generator.sample_rate, format="wav")
            audio_bytes.seek(0)
            audio_b64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
            
            # Add bot response to conversation history
            bot_segment = Segment(
                text=bot_response,
                speaker=1,  # Bot is speaker 1
                audio=bot_audio
            )
            context['segments'].append(bot_segment)
            
            # Send audio response to client
            emit('audio_response', {
                'audio': audio_b64,
                'text': bot_response
            }, room=session_id)
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

if __name__ == '__main__':
    # Ensure the existing index.html file is in the correct location
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        os.rename('index.html', 'templates/index.html')
    
    # Load models asynchronously before starting the server
    print("Starting CPU-only model loading...")
    # In a production environment, you could load models in a separate thread
    load_models()
    
    # Start the server
    print("Starting Flask SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)