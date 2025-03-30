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
import gc
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Import WhisperX for better transcription
import whisperx

from generator import load_csm_1b, Segment
from dataclasses import dataclass

# Add these imports at the top
import psutil
import gc

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
    whisperx_model = None
    whisperx_align_model = None
    whisperx_align_metadata = None
    diarize_model = None
    last_language = None

# Initialize the models object
models = AppModels()

def load_models():
    """Load all required models"""
    global models
    
    socketio.emit('model_status', {'model': 'overall', 'status': 'loading', 'progress': 0})
    
    # CSM 1B loading
    try:
        socketio.emit('model_status', {'model': 'overall', 'status': 'loading', 'progress': 10, 'message': 'Loading CSM voice model'})
        models.generator = load_csm_1b(device=DEVICE)
        logger.info("CSM 1B model loaded successfully")
        socketio.emit('model_status', {'model': 'csm', 'status': 'loaded'})
        socketio.emit('model_status', {'model': 'overall', 'status': 'loading', 'progress': 33})
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error loading CSM 1B model: {str(e)}\n{error_details}")
        socketio.emit('model_status', {'model': 'csm', 'status': 'error', 'message': str(e)})
    
    # WhisperX loading
    try:
        socketio.emit('model_status', {'model': 'overall', 'status': 'loading', 'progress': 40, 'message': 'Loading speech recognition model'})
        # Use WhisperX for better transcription with timestamps
        import whisperx
        
        # Use compute_type based on device
        compute_type = "float16" if DEVICE == "cuda" else "float32"
        
        # Load the WhisperX model (smaller model for faster processing)
        models.whisperx_model = whisperx.load_model("small", DEVICE, compute_type=compute_type)
        
        logger.info("WhisperX model loaded successfully")
        socketio.emit('model_status', {'model': 'asr', 'status': 'loaded'})
        socketio.emit('model_status', {'model': 'overall', 'status': 'loading', 'progress': 66})
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error loading WhisperX model: {str(e)}\n{error_details}")
        socketio.emit('model_status', {'model': 'asr', 'status': 'error', 'message': str(e)})
    
    # Llama loading
    try:
        socketio.emit('model_status', {'model': 'overall', 'status': 'loading', 'progress': 70, 'message': 'Loading language model'})
        models.llm = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            device_map=DEVICE,
            torch_dtype=torch.bfloat16
        )
        models.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        
        # Configure all special tokens
        models.tokenizer.pad_token = models.tokenizer.eos_token
        models.tokenizer.padding_side = "left"  # For causal language modeling

        # Inform the model about the pad token
        if hasattr(models.llm.config, "pad_token_id") and models.llm.config.pad_token_id is None:
            models.llm.config.pad_token_id = models.tokenizer.pad_token_id
        
        logger.info("Llama 3.2 model loaded successfully")
        socketio.emit('model_status', {'model': 'llm', 'status': 'loaded'})
        socketio.emit('model_status', {'model': 'overall', 'status': 'loading', 'progress': 100, 'message': 'All models loaded successfully'})
        socketio.emit('model_status', {'model': 'overall', 'status': 'loaded'})
    except Exception as e:
        logger.error(f"Error loading Llama 3.2 model: {str(e)}")
        socketio.emit('model_status', {'model': 'llm', 'status': 'error', 'message': str(e)})

# Load models in a background thread
threading.Thread(target=load_models, daemon=True).start()

# Conversation data structure
class Conversation:
    def __init__(self, session_id):
        self.session_id = session_id
        self.segments: List[Segment] = []
        self.current_speaker = 0
        self.ai_speaker_id = 1  # Add this property
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

# Fix the system_status function:

@app.route('/api/status')
def system_status():
    return jsonify({
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": DEVICE,
        "models": {
            "generator": models.generator is not None,
            "asr": models.whisperx_model is not None,  # Use the correct model name
            "llm": models.llm is not None
        }
    })

# Add a new endpoint to check system resources
@app.route('/api/system_resources')
def system_resources():
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024 ** 3)
    memory_total_gb = memory.total / (1024 ** 3)
    memory_percent = memory.percent
    
    # Get GPU memory if available
    gpu_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory[f"gpu_{i}"] = {
                "allocated": torch.cuda.memory_allocated(i) / (1024 ** 3),
                "reserved": torch.cuda.memory_reserved(i) / (1024 ** 3),
                "max_allocated": torch.cuda.max_memory_allocated(i) / (1024 ** 3)
            }
    
    return jsonify({
        "cpu_percent": cpu_percent,
        "memory": {
            "used_gb": memory_used_gb,
            "total_gb": memory_total_gb,
            "percent": memory_percent
        },
        "gpu_memory": gpu_memory,
        "active_sessions": len(active_conversations)
    })

# Socket event handlers
@socketio.on('connect')
def handle_connect(auth=None):
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
def handle_disconnect(reason=None):
    session_id = request.sid
    logger.info(f"Client disconnected: {session_id}. Reason: {reason}")
    
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
                # Create an app context for the socket emit
                with app.app_context():
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
    """Process audio data and generate a response using WhisperX"""
    if models.generator is None or models.whisperx_model is None or models.llm is None:
        logger.warning("Models not yet loaded!")
        with app.app_context():
            socketio.emit('error', {'message': 'Models still loading, please wait'}, room=session_id)
        return
    
    logger.info(f"Processing audio for session {session_id}")
    conversation = active_conversations[session_id]
    
    try:
        # Set processing flag
        conversation.is_processing = True
        
        # Process base64 audio data
        audio_data = data['audio']
        speaker_id = data['speaker']
        logger.info(f"Received audio from speaker {speaker_id}")
        
        # Convert from base64 to WAV
        try:
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            logger.info(f"Decoded audio bytes: {len(audio_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error decoding base64 audio: {str(e)}")
            raise
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # Notify client that transcription is starting
            with app.app_context():
                socketio.emit('processing_status', {'status': 'transcribing'}, room=session_id)
            
            # Load audio using WhisperX
            import whisperx
            audio = whisperx.load_audio(temp_path)
            
            # Check audio length and add a warning for short clips
            audio_length = len(audio) / 16000  # assuming 16kHz sample rate
            if audio_length < 1.0:
                logger.warning(f"Audio is very short ({audio_length:.2f}s), may affect transcription quality")
            
            # Transcribe using WhisperX
            batch_size = 16  # adjust based on your GPU memory
            logger.info("Running WhisperX transcription...")
            
            # Handle the warning about audio being shorter than 30s by suppressing it
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="audio is shorter than 30s")
                result = models.whisperx_model.transcribe(audio, batch_size=batch_size)
            
            # Get the detected language
            language_code = result["language"]
            logger.info(f"Detected language: {language_code}")
            
            # Check if alignment model needs to be loaded or updated
            if models.whisperx_align_model is None or language_code != models.last_language:
                # Clean up old models if they exist
                if models.whisperx_align_model is not None:
                    del models.whisperx_align_model
                    del models.whisperx_align_metadata
                    if DEVICE == "cuda":
                        gc.collect()
                        torch.cuda.empty_cache()
                
                # Load new alignment model for the detected language
                logger.info(f"Loading alignment model for language: {language_code}")
                models.whisperx_align_model, models.whisperx_align_metadata = whisperx.load_align_model(
                    language_code=language_code, device=DEVICE
                )
                models.last_language = language_code
            
            # Align the transcript to get word-level timestamps
            if result["segments"] and len(result["segments"]) > 0:
                logger.info("Aligning transcript...")
                result = whisperx.align(
                    result["segments"], 
                    models.whisperx_align_model, 
                    models.whisperx_align_metadata, 
                    audio, 
                    DEVICE, 
                    return_char_alignments=False
                )
                
                # Process the segments for better output
                for segment in result["segments"]:
                    # Round timestamps for better display
                    segment["start"] = round(segment["start"], 2)
                    segment["end"] = round(segment["end"], 2)
                    # Add a confidence score if not present
                    if "confidence" not in segment:
                        segment["confidence"] = 1.0  # Default confidence
            
            # Extract the full text from all segments
            user_text = ' '.join([segment['text'] for segment in result['segments']])
            
            # If no text was recognized, don't process further
            if not user_text or len(user_text.strip()) == 0:
                with app.app_context():
                    socketio.emit('error', {'message': 'No speech detected'}, room=session_id)
                return
                
            logger.info(f"Transcription: {user_text}")
            
            # Load audio for CSM input
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
            
            # Add the user's message to conversation history
            user_segment = conversation.add_segment(
                text=user_text,
                speaker=speaker_id,
                audio=waveform.squeeze()
            )
            
            # Send transcription to client with detailed segments
            with app.app_context():
                socketio.emit('transcription', {
                    'text': user_text, 
                    'speaker': speaker_id,
                    'segments': result['segments']  # Include the detailed segments with timestamps
                }, room=session_id)
            
            # Generate AI response using Llama
            with app.app_context():
                socketio.emit('processing_status', {'status': 'generating'}, room=session_id)
            
            # Create prompt from conversation history
            conversation_history = ""
            for segment in conversation.segments[-5:]:  # Last 5 segments for context
                role = "User" if segment.speaker == 0 else "Assistant"
                conversation_history += f"{role}: {segment.text}\n"
            
            # Add final prompt
            prompt = f"{conversation_history}Assistant: "
            
            # Generate response with Llama
            try:
                # Ensure pad token is set
                if models.tokenizer.pad_token is None:
                    models.tokenizer.pad_token = models.tokenizer.eos_token
                    
                input_tokens = models.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True
                )
                input_ids = input_tokens.input_ids.to(DEVICE)
                attention_mask = input_tokens.attention_mask.to(DEVICE)

                with torch.no_grad():
                    generated_ids = models.llm.generate(
                        input_ids,
                        attention_mask=attention_mask,
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
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                response_text = "I'm sorry, I encountered an error while processing your request."
            
            # Synthesize speech
            with app.app_context():
                socketio.emit('processing_status', {'status': 'synthesizing'}, room=session_id)
                
                # Start sending the audio response
                socketio.emit('audio_response_start', {
                    'text': response_text,
                    'total_chunks': 1,
                    'chunk_index': 0
                }, room=session_id)
            
            # Define AI speaker ID
            ai_speaker_id = conversation.ai_speaker_id
            
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
            with app.app_context():
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
        import traceback
        logger.error(traceback.format_exc())
        with app.app_context():
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
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting server on port {port} (debug={debug_mode})")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug_mode, allow_unsafe_werkzeug=True)