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
from faster_whisper import WhisperModel
from generator import load_csm_1b, Segment
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Select the best available device
if torch.cuda.is_available():
    device = "cuda"
    whisper_compute_type = "float16"
elif torch.backends.mps.is_available():
    device = "mps"
    whisper_compute_type = "float32"
else:
    device = "cpu"
    whisper_compute_type = "int8"
    
print(f"Using device: {device}")

# Initialize Faster-Whisper for transcription
print("Loading Whisper model...")
whisper_model = WhisperModel("base", device=device, compute_type=whisper_compute_type)

# Initialize CSM model for audio generation
print("Loading CSM model...")
csm_generator = load_csm_1b(device=device)

# Initialize Llama 3.2 model for response generation
print("Loading Llama 3.2 model...")
llm_model_id = "meta-llama/Llama-3.2-1B"  # Choose appropriate size based on resources
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    torch_dtype=torch.bfloat16,
    device_map=device
)

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
    
    # Save audio to temporary WAV file for Whisper transcription
    temp_audio_path = f"temp_audio_{session_id}.wav"
    torchaudio.save(
        temp_audio_path, 
        full_audio.unsqueeze(0), 
        44100  # Assuming 44.1kHz from client
    )
    
    # Transcribe speech using Faster-Whisper
    try:
        segments, info = whisper_model.transcribe(temp_audio_path, beam_size=5)
        
        # Collect all text from segments
        user_text = ""
        for segment in segments:
            segment_text = segment.text.strip()
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment_text}")
            user_text += segment_text + " "
        
        user_text = user_text.strip()
        
        # Cleanup temp file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        if not user_text:
            print("No speech detected.")
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
        
        # Send transcribed text to client
        emit('transcription', {'text': user_text}, room=session_id)
        
        # Send audio response to client
        emit('audio_response', {
            'audio': audio_b64,
            'text': bot_response
        }, room=session_id)
        
    except Exception as e:
        print(f"Error processing speech: {e}")
        emit('error', {'message': f'Error processing speech: {str(e)}'}, room=session_id)
        # Cleanup temp file in case of error
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def generate_llm_response(user_text, conversation_segments):
    """Generate text response using Llama 3.2"""
    # Format conversation history for the LLM
    conversation_history = ""
    for segment in conversation_segments[-5:]:  # Use last 5 utterances for context
        speaker_name = "User" if segment.speaker == 0 else "Assistant"
        conversation_history += f"{speaker_name}: {segment.text}\n"
    
    # Add the current user query
    conversation_history += f"User: {user_text}\nAssistant:"
    
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

def generate_audio_response(text, conversation_segments):
    """Generate audio response using CSM"""
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

if __name__ == '__main__':
    # Ensure the existing index.html file is in the correct location
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        os.rename('index.html', 'templates/index.html')
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)