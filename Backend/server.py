import os
import base64
import json
import time
import math
import gc
import logging
import numpy as np
import torch
import torchaudio
import whisperx
from io import BytesIO
from typing import List, Dict, Any, Optional
from flask import Flask, request, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from generator import load_csm_1b, Segment
from collections import deque
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sesame-server")

# CUDA Environment Setup
def setup_cuda_environment():
    """Set up CUDA environment with proper error handling"""
    # Search for CUDA libraries in common locations
    cuda_lib_dirs = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/extras/CUPTI/lib64"
    ]
    
    # Add directories to LD_LIBRARY_PATH if they exist
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    for cuda_dir in cuda_lib_dirs:
        if os.path.exists(cuda_dir) and cuda_dir not in current_ld_path:
            if current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{current_ld_path}:{cuda_dir}"
            else:
                os.environ['LD_LIBRARY_PATH'] = cuda_dir
            current_ld_path = os.environ['LD_LIBRARY_PATH']
    
    logger.info(f"LD_LIBRARY_PATH set to: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")
    
    # Determine best compute device
    device = "cpu"
    compute_type = "int8"
    
    try:
        # Set CUDA preferences
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Limit to first GPU only
        
        # Try enabling TF32 precision if available
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        except Exception as e:
            logger.warning(f"Could not set advanced CUDA options: {e}")
        
        # Test if CUDA is functional
        if torch.cuda.is_available():
            try:
                # Test basic CUDA operations
                x = torch.rand(10, device="cuda")
                y = x + x
                del x, y
                torch.cuda.empty_cache()
                device = "cuda"
                compute_type = "float16"
                logger.info("CUDA is fully functional")
            except Exception as e:
                logger.warning(f"CUDA available but not working correctly: {e}")
                device = "cpu"
        else:
            logger.info("CUDA is not available, using CPU")
    except Exception as e:
        logger.error(f"Error setting up computing environment: {e}")
    
    return device, compute_type

# Set up the compute environment
device, compute_type = setup_cuda_environment()

# Constants and Configuration
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION_SEC = 0.75
MAX_BUFFER_SIZE = 30  # Maximum chunks to buffer before processing
CHUNK_SIZE_MS = 500  # Size of audio chunks when streaming responses

# Define the base directory and static files directory
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")
os.makedirs(static_dir, exist_ok=True)

# Model Loading Functions
def load_speech_models():
    """Load all required speech models with fallbacks"""
    # Load speech generation model (Sesame CSM)
    try:
        logger.info(f"Loading Sesame CSM model on {device}...")
        generator = load_csm_1b(device=device)
        logger.info("Sesame CSM model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Sesame CSM on {device}: {e}")
        if device == "cuda":
            try:
                logger.info("Trying to load Sesame CSM on CPU instead...")
                generator = load_csm_1b(device="cpu")
                logger.info("Sesame CSM model loaded on CPU successfully")
            except Exception as cpu_error:
                logger.critical(f"Failed to load speech synthesis model: {cpu_error}")
                raise RuntimeError("Failed to load speech synthesis model")
        else:
            raise RuntimeError("Failed to load speech synthesis model on any device")
    
    # Load ASR model (WhisperX)
    try:
        logger.info("Loading WhisperX model...")
        # Start with the tiny model on CPU for reliable initialization
        asr_model = whisperx.load_model("tiny", "cpu", compute_type="int8")
        logger.info("WhisperX 'tiny' model loaded on CPU successfully")
        
        # Try upgrading to GPU if available
        if device == "cuda":
            try:
                logger.info("Trying to load WhisperX on CUDA...")
                # Test with a tiny model first
                test_audio = torch.zeros(16000)  # 1 second of silence
                
                cuda_model = whisperx.load_model("tiny", "cuda", compute_type="float16")
                # Test the model with real inference
                _ = cuda_model.transcribe(test_audio.numpy(), batch_size=1)
                asr_model = cuda_model
                logger.info("WhisperX model running on CUDA successfully")
                
                # Try to upgrade to small model
                try:
                    small_model = whisperx.load_model("small", "cuda", compute_type="float16")
                    _ = small_model.transcribe(test_audio.numpy(), batch_size=1)
                    asr_model = small_model
                    logger.info("WhisperX 'small' model loaded on CUDA successfully")
                except Exception as e:
                    logger.warning(f"Staying with 'tiny' model on CUDA: {e}")
            except Exception as e:
                logger.warning(f"CUDA loading failed, staying with CPU model: {e}")
    except Exception as e:
        logger.error(f"Error loading WhisperX model: {e}")
        # Create a minimal dummy model as last resort
        class DummyModel:
            def __init__(self):
                self.device = "cpu"
            def transcribe(self, *args, **kwargs):
                return {"segments": [{"text": "Speech recognition currently unavailable."}]}
        
        asr_model = DummyModel()
        logger.warning("Using dummy transcription model - ASR functionality limited")
    
    return generator, asr_model

# Load speech models
generator, asr_model = load_speech_models()

# Set up Flask and Socket.IO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Socket connection management
thread_lock = Lock()
active_clients = {}  # Map client_id to client context

# Audio Utility Functions
def decode_audio_data(audio_data: str) -> torch.Tensor:
    """Decode base64 audio data to a torch tensor with improved error handling"""
    try:
        # Skip empty audio data
        if not audio_data or len(audio_data) < 100:
            logger.warning("Empty or too short audio data received")
            return torch.zeros(generator.sample_rate // 2)  # 0.5 seconds of silence
            
        # Extract the actual base64 content
        if ',' in audio_data:
            audio_data = audio_data.split(',')[1]
            
        # Decode base64 audio data
        try:
            binary_data = base64.b64decode(audio_data)
            logger.debug(f"Decoded base64 data: {len(binary_data)} bytes")
            
            # Check if we have enough data for a valid WAV
            if len(binary_data) < 44:  # WAV header is 44 bytes
                logger.warning("Data too small to be a valid WAV file")
                return torch.zeros(generator.sample_rate // 2)
        except Exception as e:
            logger.error(f"Base64 decoding error: {e}")
            return torch.zeros(generator.sample_rate // 2)
        
        # Multiple approaches to handle audio data
        audio_tensor = None
        sample_rate = None
        
        # Approach 1: Direct loading with torchaudio
        try:
            with BytesIO(binary_data) as temp_file:
                temp_file.seek(0)
                audio_tensor, sample_rate = torchaudio.load(temp_file, format="wav")
                logger.debug(f"Loaded audio: shape={audio_tensor.shape}, rate={sample_rate}Hz")
                
                # Validate tensor
                if audio_tensor.numel() == 0 or torch.isnan(audio_tensor).any():
                    raise ValueError("Invalid audio tensor")
        except Exception as e:
            logger.warning(f"Direct loading failed: {e}")
            
            # Approach 2: Using wave module and numpy
            try:
                temp_path = os.path.join(base_dir, f"temp_{time.time()}.wav")
                with open(temp_path, 'wb') as f:
                    f.write(binary_data)
                
                import wave
                with wave.open(temp_path, 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    sample_rate = wf.getframerate()
                    n_frames = wf.getnframes()
                    frames = wf.readframes(n_frames)
                    
                    # Convert to numpy array
                    if sample_width == 2:  # 16-bit audio
                        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sample_width == 1:  # 8-bit audio
                        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")
                    
                    # Convert to mono if needed
                    if n_channels > 1:
                        data = data.reshape(-1, n_channels)
                        data = data.mean(axis=1)
                    
                    # Convert to torch tensor
                    audio_tensor = torch.from_numpy(data)
                    logger.info(f"Loaded audio using wave: shape={audio_tensor.shape}")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e2:
                logger.error(f"All audio loading methods failed: {e2}")
                return torch.zeros(generator.sample_rate // 2)
        
        # Format corrections
        if audio_tensor is None:
            return torch.zeros(generator.sample_rate // 2)
            
        # Ensure audio is mono
        if len(audio_tensor.shape) > 1 and audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0)
        
        # Ensure 1D tensor
        audio_tensor = audio_tensor.squeeze()
            
        # Resample if needed
        if sample_rate != generator.sample_rate:
            try:
                logger.debug(f"Resampling from {sample_rate}Hz to {generator.sample_rate}Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=generator.sample_rate
                )
                audio_tensor = resampler(audio_tensor)
            except Exception as e:
                logger.warning(f"Resampling error: {e}")
        
        # Normalize audio to avoid issues
        if torch.abs(audio_tensor).max() > 0:
            audio_tensor = audio_tensor / torch.abs(audio_tensor).max()
        
        return audio_tensor
    except Exception as e:
        logger.error(f"Unhandled error in decode_audio_data: {e}")
        return torch.zeros(generator.sample_rate // 2)

def encode_audio_data(audio_tensor: torch.Tensor) -> str:
    """Encode torch tensor audio to base64 string"""
    try:
        buf = BytesIO()
        torchaudio.save(buf, audio_tensor.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
        buf.seek(0)
        audio_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:audio/wav;base64,{audio_base64}"
    except Exception as e:
        logger.error(f"Error encoding audio: {e}")
        # Return a minimal silent audio file
        silence = torch.zeros(generator.sample_rate // 2).unsqueeze(0)
        buf = BytesIO()
        torchaudio.save(buf, silence, generator.sample_rate, format="wav")
        buf.seek(0)
        return f"data:audio/wav;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def transcribe_audio(audio_tensor: torch.Tensor) -> str:
    """Transcribe audio using WhisperX with robust error handling"""
    global asr_model
    
    try:
        # Save the tensor to a temporary file
        temp_path = os.path.join(base_dir, f"temp_audio_{time.time()}.wav")
        torchaudio.save(temp_path, audio_tensor.unsqueeze(0).cpu(), generator.sample_rate)
        
        logger.info(f"Transcribing audio file: {os.path.getsize(temp_path)} bytes")
        
        # Load the audio for WhisperX
        try:
            audio = whisperx.load_audio(temp_path)
        except Exception as e:
            logger.warning(f"WhisperX load_audio failed: {e}")
            # Fall back to manual loading
            import soundfile as sf
            audio, sr = sf.read(temp_path)
            if sr != 16000:  # WhisperX expects 16kHz audio
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / sr))
        
        # Transcribe with error handling
        try:
            result = asr_model.transcribe(audio, batch_size=4)
        except RuntimeError as e:
            if "CUDA" in str(e) or "libcudnn" in str(e):
                logger.warning(f"CUDA error in transcription, falling back to CPU: {e}")
                try:
                    # Try CPU model
                    cpu_model = whisperx.load_model("tiny", "cpu", compute_type="int8")
                    result = cpu_model.transcribe(audio, batch_size=1)
                    # Update the global model if the original one is broken
                    asr_model = cpu_model
                except Exception as cpu_e:
                    logger.error(f"CPU fallback failed: {cpu_e}")
                    return "I'm having trouble processing audio right now."
            else:
                raise
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Extract text from segments
        if result["segments"] and len(result["segments"]) > 0:
            transcription = " ".join([segment["text"] for segment in result["segments"]])
            logger.info(f"Transcription: '{transcription.strip()}'")
            return transcription.strip()
        
        return ""
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return "I heard something but couldn't understand it."

def generate_response(text: str, conversation_history: List[Segment]) -> str:
    """Generate a contextual response based on the transcribed text"""
    # Simple response logic - can be replaced with a more sophisticated LLM
    responses = {
        "hello": "Hello there! How can I help you today?",
        "hi": "Hi there! What can I do for you?",
        "how are you": "I'm doing well, thanks for asking! How about you?",
        "what is your name": "I'm Sesame, your voice assistant. How can I help you?",
        "who are you": "I'm Sesame, an AI voice assistant. I'm here to chat with you!",
        "bye": "Goodbye! It was nice chatting with you.",
        "thank you": "You're welcome! Is there anything else I can help with?",
        "weather": "I don't have real-time weather data, but I hope it's nice where you are!",
        "help": "I can chat with you using natural voice. Just speak normally and I'll respond.",
        "what can you do": "I can have a conversation with you, answer questions, and provide assistance with various topics.",
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

# Flask Routes
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

# Socket.IO Event Handlers
@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    logger.info(f"Client connected: {client_id}")
    
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
    logger.info(f"Client disconnected: {client_id}")

@socketio.on('generate')
def handle_generate(data):
    client_id = request.sid
    if client_id not in active_clients:
        emit('error', {'message': 'Client not registered'})
        return
    
    try:
        text = data.get('text', '')
        speaker_id = data.get('speaker', 0)
        
        logger.info(f"Generating audio for: '{text}' with speaker {speaker_id}")
        
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
            'audio': audio_base64,
            'text': text
        })
        
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
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
        logger.error(f"Error adding to context: {e}")
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
        
        # Skip if no audio data (might be just a connection test)
        if not audio_data:
            logger.debug("Empty audio data received, ignoring")
            return
        
        # Convert received audio to tensor
        audio_chunk = decode_audio_data(audio_data)
        
        # Start streaming mode if not already started
        if not client['is_streaming']:
            client['is_streaming'] = True
            client['streaming_buffer'] = []
            client['energy_window'].clear()
            client['is_silence'] = False
            client['last_active_time'] = time.time()
            logger.info(f"[{client_id[:8]}] Streaming started with speaker ID: {speaker_id}")
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
            logger.info(f"[{client_id[:8]}] Processing audio after {silence_elapsed:.2f}s of silence")
            process_complete_utterance(client_id, client, speaker_id)
        
        # If buffer gets too large without silence, process it anyway
        elif len(client['streaming_buffer']) >= MAX_BUFFER_SIZE:
            logger.info(f"[{client_id[:8]}] Processing long audio segment without silence")
            process_complete_utterance(client_id, client, speaker_id, is_incomplete=True)
            
            # Keep half of the buffer for context (sliding window approach)
            half_point = len(client['streaming_buffer']) // 2
            client['streaming_buffer'] = client['streaming_buffer'][half_point:]
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error processing streaming audio: {e}")
        emit('error', {
            'type': 'error',
            'message': f"Error processing streaming audio: {str(e)}"
        })

def process_complete_utterance(client_id, client, speaker_id, is_incomplete=False):
    """Process a complete utterance (after silence or buffer limit)"""
    try:
        # Combine audio chunks
        full_audio = torch.cat(client['streaming_buffer'], dim=0)
        
        # Process with speech-to-text
        logger.info(f"[{client_id[:8]}] Starting transcription...")
        transcribed_text = transcribe_audio(full_audio)
        
        # Add suffix for incomplete utterances
        if is_incomplete:
            transcribed_text += " (processing continued speech...)"
        
        # Log the transcription
        logger.info(f"[{client_id[:8]}] Transcribed: '{transcribed_text}'")
        
        # Handle the transcription result
        if transcribed_text:
            # Add user message to context
            user_segment = Segment(text=transcribed_text, speaker=speaker_id, audio=full_audio)
            client['context_segments'].append(user_segment)
            
            # Send the transcribed text to client
            emit('transcription', {
                'type': 'transcription',
                'text': transcribed_text
            }, room=client_id)
            
            # Only generate a response if this is a complete utterance
            if not is_incomplete:
                # Generate a contextual response
                response_text = generate_response(transcribed_text, client['context_segments'])
                logger.info(f"[{client_id[:8]}] Generating response: '{response_text}'")
                
                # Let the client know we're processing
                emit('processing_status', {
                    'type': 'processing_status',
                    'status': 'generating_audio',
                    'message': 'Generating audio response...'
                }, room=client_id)
                
                # Generate audio for the response
                try:
                    # Use a different speaker than the user
                    ai_speaker_id = 1 if speaker_id == 0 else 0
                    
                    # Generate the full response
                    audio_tensor = generator.generate(
                        text=response_text,
                        speaker=ai_speaker_id,
                        context=client['context_segments'],
                        max_audio_length_ms=10_000,
                    )
                    
                    # Add response to context
                    ai_segment = Segment(
                        text=response_text, 
                        speaker=ai_speaker_id, 
                        audio=audio_tensor
                    )
                    client['context_segments'].append(ai_segment)
                    
                    # Convert audio to base64 and send back to client
                    audio_base64 = encode_audio_data(audio_tensor)
                    emit('audio_response', {
                        'type': 'audio_response',
                        'text': response_text,
                        'audio': audio_base64
                    }, room=client_id)
                    
                    logger.info(f"[{client_id[:8]}] Audio response sent")
                    
                except Exception as e:
                    logger.error(f"Error generating audio response: {e}")
                    emit('error', {
                        'type': 'error',
                        'message': "Sorry, there was an error generating the audio response."
                    }, room=client_id)
        else:
            # If transcription failed, send a notification
            emit('error', {
                'type': 'error',
                'message': "Sorry, I couldn't understand what you said. Could you try again?"
            }, room=client_id)
        
        # Only clear buffer for complete utterances
        if not is_incomplete:
            # Reset state
            client['streaming_buffer'] = []
            client['energy_window'].clear()
            client['is_silence'] = False
            client['last_active_time'] = time.time()
            
    except Exception as e:
        logger.error(f"Error processing utterance: {e}")
        emit('error', {
            'type': 'error',
            'message': f"Error processing audio: {str(e)}"
        }, room=client_id)

@socketio.on('stop_streaming')
def handle_stop_streaming(data):
    client_id = request.sid
    if client_id not in active_clients:
        return
    
    client = active_clients[client_id]
    client['is_streaming'] = False
    
    if client['streaming_buffer'] and len(client['streaming_buffer']) > 5:
        # Process any remaining audio in the buffer
        logger.info(f"[{client_id[:8]}] Processing final audio buffer on stop")
        process_complete_utterance(client_id, client, data.get("speaker", 0))
    
    client['streaming_buffer'] = []
    emit('streaming_status', {
        'type': 'streaming_status',
        'status': 'stopped'
    })

def stream_audio_to_client(client_id, audio_tensor, text, speaker_id, chunk_size_ms=CHUNK_SIZE_MS):
    """Stream audio to client in chunks to simulate real-time generation"""
    try:
        if client_id not in active_clients:
            logger.warning(f"Client {client_id} not found for streaming")
            return
            
        # Calculate chunk size in samples
        chunk_size = int(generator.sample_rate * chunk_size_ms / 1000)
        total_chunks = math.ceil(audio_tensor.size(0) / chunk_size)
        
        logger.info(f"Streaming audio in {total_chunks} chunks of {chunk_size_ms}ms each")
        
        # Send initial response with text but no audio yet
        socketio.emit('audio_response_start', {
            'type': 'audio_response_start',
            'text': text,
            'total_chunks': total_chunks
        }, room=client_id)
        
        # Stream each chunk
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, audio_tensor.size(0))
            
            # Extract chunk
            chunk = audio_tensor[start_idx:end_idx]
            
            # Encode chunk
            chunk_base64 = encode_audio_data(chunk)
            
            # Send chunk
            socketio.emit('audio_response_chunk', {
                'type': 'audio_response_chunk',
                'chunk_index': i,
                'total_chunks': total_chunks,
                'audio': chunk_base64,
                'is_last': i == total_chunks - 1
            }, room=client_id)
            
            # Brief pause between chunks to simulate streaming
            time.sleep(0.1)
            
        # Send completion message
        socketio.emit('audio_response_complete', {
            'type': 'audio_response_complete',
            'text': text
        }, room=client_id)
        
        logger.info(f"Audio streaming complete: {total_chunks} chunks sent")
        
    except Exception as e:
        logger.error(f"Error streaming audio to client: {e}")
        import traceback
        traceback.print_exc()

# Main server start
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🔊 Sesame AI Voice Chat Server")
    print(f"{'='*60}")
    print(f"📡 Server Information:")
    print(f"   - Local URL: http://localhost:5000")
    print(f"   - Network URL: http://<your-ip-address>:5000")
    print(f"{'='*60}")
    print(f"🌐 Device: {device.upper()}")
    print(f"🧠 Models: Sesame CSM + WhisperX ASR")
    print(f"🔧 Serving from: {os.path.join(base_dir, 'index.html')}")
    print(f"{'='*60}")
    print(f"Ready to receive connections! Press Ctrl+C to stop the server.\n")
    
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)