import os
import io
import base64
import time
import json
import uuid
import logging
import threading
import queue
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from generator import load_csm_1b, Segment
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120)

# Configure device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

logger.info(f"Using device: {DEVICE}")

# Global variables
active_conversations = {}
user_queues = {}
processing_threads = {}

# Load models
@dataclass
class AppModels:
    generator = None
    tokenizer = None
    llm = None
    asr = None

models = AppModels()

def load_models():
    """Load all required models"""
    global models
    
    logger.info("Loading CSM 1B model...")
    models.generator = load_csm_1b(device=DEVICE)
    
    logger.info("Loading ASR pipeline...")
    models.asr = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-small",
        device=DEVICE
    )
    
    logger.info("Loading Llama 3.2 model...")
    models.llm = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        device_map=DEVICE,
        torch_dtype=torch.bfloat16
    )
    models.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Load models in a background thread
threading.Thread(target=load_models, daemon=True).start()

# Conversation data structure
class Conversation:
    def __init__(self, session_id):
        self.session_id = session_id
        self.segments: List[Segment] = []
        self.current_speaker = 0
        self.last_activity = time.time()
        self.is_processing = False
    
    def add_segment(self, text, speaker, audio):
        segment = Segment(text=text, speaker=speaker, audio=audio)
        self.segments.append(segment)
        self.last_activity = time.time()
        return segment
    
    def get_context(self, max_segments=10):
        """Return the most recent segments for context"""
        return self.segments[-max_segments:] if self.segments else []

# Routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/voice-chat.js')
def voice_chat_js():
    return send_from_directory('.', 'voice-chat.js')

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": models.generator is not None and models.llm is not None
    })

# Socket event handlers
@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    logger.info(f"Client connected: {session_id}")
    
    # Initialize conversation data
    if session_id not in active_conversations:
        active_conversations[session_id] = Conversation(session_id)
        user_queues[session_id] = queue.Queue()
        processing_threads[session_id] = threading.Thread(
            target=process_audio_queue, 
            args=(session_id, user_queues[session_id]),
            daemon=True
        )
        processing_threads[session_id].start()
    
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    logger.info(f"Client disconnected: {session_id}")
    
    # Cleanup
    if session_id in active_conversations:
        # Mark for deletion rather than immediately removing
        # as the processing thread might still be accessing it
        active_conversations[session_id].is_processing = False
        user_queues[session_id].put(None)  # Signal thread to terminate

@socketio.on('start_stream')
def handle_start_stream():
    session_id = request.sid
    logger.info(f"Starting stream for client: {session_id}")
    emit('streaming_status', {'status': 'active'})

@socketio.on('stop_stream')
def handle_stop_stream():
    session_id = request.sid
    logger.info(f"Stopping stream for client: {session_id}")
    emit('streaming_status', {'status': 'inactive'})

@socketio.on('clear_context')
def handle_clear_context():
    session_id = request.sid
    logger.info(f"Clearing context for client: {session_id}")
    
    if session_id in active_conversations:
        active_conversations[session_id].segments = []
        emit('context_updated', {'status': 'cleared'})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    session_id = request.sid
    audio_data = data.get('audio', '')
    speaker_id = int(data.get('speaker', 0))
    
    if not audio_data or not session_id in active_conversations:
        return
    
    # Update the current speaker
    active_conversations[session_id].current_speaker = speaker_id
    
    # Queue audio for processing
    user_queues[session_id].put({
        'audio': audio_data,
        'speaker': speaker_id
    })
    
    emit('processing_status', {'status': 'transcribing'})

def process_audio_queue(session_id, q):
    """Background thread to process audio chunks for a session"""
    logger.info(f"Started processing thread for session: {session_id}")
    
    try:
        while session_id in active_conversations:
            try:
                # Get the next audio chunk with a timeout
                data = q.get(timeout=120)
                if data is None:  # Termination signal
                    break
                
                # Process the audio and generate a response
                process_audio_and_respond(session_id, data)
                
            except queue.Empty:
                # Timeout, check if session is still valid
                continue
            except Exception as e:
                logger.error(f"Error processing audio for {session_id}: {str(e)}")
                socketio.emit('error', {'message': str(e)}, room=session_id)
    finally:
        logger.info(f"Ending processing thread for session: {session_id}")
        # Clean up when thread is done
        with app.app_context():
            if session_id in active_conversations:
                del active_conversations[session_id]
            if session_id in user_queues:
                del user_queues[session_id]

def process_audio_and_respond(session_id, data):
    """Process audio data and generate a response"""
    if models.generator is None or models.asr is None or models.llm is None:
        socketio.emit('error', {'message': 'Models still loading, please wait'}, room=session_id)
        return
    
    conversation = active_conversations[session_id]
    
    try:
        # Set processing flag
        conversation.is_processing = True
        
        # Process base64 audio data
        audio_data = data['audio']
        speaker_id = data['speaker']
        
        # Convert from base64 to WAV
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(temp_path)
            
            # Normalize to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to the CSM sample rate if needed
            if sample_rate != models.generator.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, 
                    orig_freq=sample_rate, 
                    new_freq=models.generator.sample_rate
                )
            
            # Transcribe audio
            socketio.emit('processing_status', {'status': 'transcribing'}, room=session_id)
            
            # Use the ASR pipeline to transcribe
            transcription_result = models.asr(
                {"array": waveform.squeeze().cpu().numpy(), "sampling_rate": models.generator.sample_rate},
                return_timestamps=False
            )
            user_text = transcription_result['text'].strip()
            
            # If no text was recognized, don't process further
            if not user_text:
                socketio.emit('error', {'message': 'No speech detected'}, room=session_id)
                return
            
            # Add the user's message to conversation history
            user_segment = conversation.add_segment(
                text=user_text,
                speaker=speaker_id,
                audio=waveform.squeeze()
            )
            
            # Send transcription to client
            socketio.emit('transcription', {
                'text': user_text, 
                'speaker': speaker_id
            }, room=session_id)
            
            # Generate AI response using Llama
            socketio.emit('processing_status', {'status': 'generating'}, room=session_id)
            
            # Create prompt from conversation history
            conversation_history = ""
            for segment in conversation.segments[-5:]:  # Last 5 segments for context
                role = "User" if segment.speaker == 0 else "Assistant"
                conversation_history += f"{role}: {segment.text}\n"
            
            # Add final prompt
            prompt = f"{conversation_history}Assistant: "
            
            # Generate response with Llama
            input_ids = models.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            
            with torch.no_grad():
                generated_ids = models.llm.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=models.tokenizer.eos_token_id
                )
            
            # Decode the response
            response_text = models.tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Synthesize speech
            socketio.emit('processing_status', {'status': 'synthesizing'}, room=session_id)
            
            # Generate audio with CSM
            ai_speaker_id = 1  # Use speaker 1 for AI responses
            
            # Start sending the audio response
            socketio.emit('audio_response_start', {
                'text': response_text,
                'total_chunks': 1,
                'chunk_index': 0
            }, room=session_id)
            
            # Generate audio
            audio_tensor = models.generator.generate(
                text=response_text,
                speaker=ai_speaker_id,
                context=conversation.get_context(),
                max_audio_length_ms=10_000,
                temperature=0.9
            )
            
            # Add AI response to conversation history
            ai_segment = conversation.add_segment(
                text=response_text,
                speaker=ai_speaker_id,
                audio=audio_tensor
            )
            
            # Convert audio to WAV format
            with io.BytesIO() as wav_io:
                torchaudio.save(
                    wav_io, 
                    audio_tensor.unsqueeze(0).cpu(), 
                    models.generator.sample_rate, 
                    format="wav"
                )
                wav_io.seek(0)
                wav_data = wav_io.read()
            
            # Convert WAV data to base64
            audio_base64 = f"data:audio/wav;base64,{base64.b64encode(wav_data).decode('utf-8')}"
            
            # Send audio chunk to client
            socketio.emit('audio_response_chunk', {
                'chunk': audio_base64,
                'chunk_index': 0,
                'total_chunks': 1,
                'is_last': True
            }, room=session_id)
            
            # Signal completion
            socketio.emit('audio_response_complete', {
                'text': response_text
            }, room=session_id)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        socketio.emit('error', {'message': f'Error: {str(e)}'}, room=session_id)
    finally:
        # Reset processing flag
        conversation.is_processing = False

# Error handler
@socketio.on_error()
def error_handler(e):
    logger.error(f"SocketIO error: {str(e)}")

# Periodic cleanup of inactive sessions
def cleanup_inactive_sessions():
    """Remove sessions that have been inactive for too long"""
    current_time = time.time()
    inactive_timeout = 3600  # 1 hour
    
    for session_id in list(active_conversations.keys()):
        conversation = active_conversations[session_id]
        if (current_time - conversation.last_activity > inactive_timeout and 
            not conversation.is_processing):
            
            logger.info(f"Cleaning up inactive session: {session_id}")
            
            # Signal processing thread to terminate
            if session_id in user_queues:
                user_queues[session_id].put(None)
            
            # Remove from active conversations
            del active_conversations[session_id]

# Start cleanup thread
def start_cleanup_thread():
    while True:
        try:
            cleanup_inactive_sessions()
        except Exception as e:
            logger.error(f"Error in cleanup thread: {str(e)}")
        time.sleep(300)  # Run every 5 minutes

cleanup_thread = threading.Thread(target=start_cleanup_thread, daemon=True)
cleanup_thread.start()

# Start the server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)