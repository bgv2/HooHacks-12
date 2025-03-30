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
from io import BytesIO
from typing import List, Dict, Any, Optional
from flask import Flask, request, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from generator import load_csm_1b, Segment
from collections import deque
from threading import Lock
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sesame-server")

# Determine best compute device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    try:
        # Test CUDA functionality
        torch.rand(10, device="cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        device = "cuda"
        logger.info("CUDA is fully functional")
    except Exception as e:
        logger.warning(f"CUDA available but not working correctly: {e}")
        device = "cpu"
else:
    device = "cpu"
    logger.info("Using CPU")

# Constants and Configuration
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION_SEC = 0.75
MAX_BUFFER_SIZE = 30  # Maximum chunks to buffer before processing
CHUNK_SIZE_MS = 500  # Size of audio chunks when streaming responses

# Define the base directory and static files directory
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")
os.makedirs(static_dir, exist_ok=True)

# Define a simple energy-based speech detector
class SpeechDetector:
    def __init__(self):
        self.min_speech_energy = 0.01
        self.speech_window = 0.2  # seconds
    
    def detect_speech(self, audio_tensor, sample_rate):
        # Calculate frame size based on window size
        frame_size = int(sample_rate * self.speech_window)
        
        # If audio is shorter than frame size, use the entire audio
        if audio_tensor.shape[0] < frame_size:
            frames = [audio_tensor]
        else:
            # Split audio into frames
            frames = [audio_tensor[i:i+frame_size] for i in range(0, len(audio_tensor), frame_size)]
        
        # Calculate energy per frame
        energies = [torch.mean(frame**2).item() for frame in frames]
        
        # Determine if there's speech based on energy threshold
        has_speech = any(e > self.min_speech_energy for e in energies)
        
        return has_speech

speech_detector = SpeechDetector()
logger.info("Initialized simple speech detector")

# Model Loading Functions
def load_speech_models():
    """Load speech generation and recognition models"""
    # Load CSM (existing code)
    generator = load_csm_1b(device=device)
    
    # Load Whisper model for speech recognition
    try:
        logger.info(f"Loading speech recognition model on {device}...")
        speech_recognizer = pipeline("automatic-speech-recognition", 
                                    model="openai/whisper-small", 
                                    device=device)
        logger.info("Speech recognition model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading speech recognition model: {e}")
        speech_recognizer = None
    
    return generator, speech_recognizer

# Unpack both models
generator, speech_recognizer = load_speech_models()

# Initialize Llama 3.2 model for conversation responses
def load_llm_model():
    """Load Llama 3.2 model for generating text responses"""
    try:
        logger.info("Loading Llama 3.2 model for conversational responses...")
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Determine compute device for LLM
        llm_device = "cpu"  # Default to CPU for LLM
        
        # Use CUDA if available and there's enough VRAM
        if device == "cuda" and torch.cuda.is_available():
            try:
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                # If we have at least 2GB free, use CUDA for LLM
                if free_mem > 2 * 1024 * 1024 * 1024:
                    llm_device = "cuda"
            except:
                pass
        
        logger.info(f"Using {llm_device} for Llama 3.2 model")
        
        # Load the model with lower precision for efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if llm_device == "cuda" else torch.float32,
            device_map=llm_device
        )
        
        # Create a pipeline for easier inference
        llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        logger.info("Llama 3.2 model loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Error loading Llama 3.2 model: {e}")
        return None

# Load the LLM model
llm = load_llm_model()

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

def process_speech(audio_tensor: torch.Tensor, client_id: str) -> str:
    """Process speech with speech recognition"""
    if not speech_recognizer:
        # Fallback to basic detection if model failed to load
        return detect_speech_energy(audio_tensor)
    
    try:
        # Save audio to temp file for Whisper
        temp_path = os.path.join(base_dir, f"temp_{time.time()}.wav")
        torchaudio.save(temp_path, audio_tensor.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Perform speech recognition - using input_features instead of inputs
        result = speech_recognizer(temp_path, input_features=None)  # input_features=None forces use of the correct parameter name
        transcription = result["text"]
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Return empty string if no speech detected
        if not transcription or transcription.isspace():
            return "I didn't detect any speech. Could you please try again?"
        
        return transcription
        
    except Exception as e:
        logger.error(f"Speech recognition error: {e}")
        return "Sorry, I couldn't understand what you said. Could you try again?"

def detect_speech_energy(audio_tensor: torch.Tensor) -> str:
    """Basic speech detection based on audio energy levels"""
    # Calculate audio energy
    energy = torch.mean(torch.abs(audio_tensor)).item()
    
    logger.debug(f"Audio energy detected: {energy:.6f}")
    
    # Generate response based on energy level
    if energy > 0.1:  # Louder speech
        return "I heard you speaking clearly. How can I help you today?"
    elif energy > 0.05:  # Moderate speech
        return "I heard you say something. Could you please repeat that?"
    elif energy > 0.02:  # Soft speech
        return "I detected some speech, but it was quite soft. Could you speak up a bit?"
    else:  # Very soft or no speech
        return "I didn't detect any speech. Could you please try again?"

def generate_response(text: str, conversation_history: List[Segment]) -> str:
    """Generate a contextual response based on the transcribed text using Llama 3.2"""
    # If LLM is not available, use simple responses
    if llm is None:
        return generate_simple_response(text)
    
    try:
        # Create a conversational prompt based on history
        # Format: recent conversation turns (up to 4) + current user input
        history_str = ""
        
        # Add up to 4 recent conversation turns (excluding the current one)
        recent_segments = [
            seg for seg in conversation_history[-8:] 
            if seg.text and not seg.text.isspace()
        ]
        
        for i, segment in enumerate(recent_segments):
            speaker_name = "User" if segment.speaker == 0 else "Assistant"
            history_str += f"{speaker_name}: {segment.text}\n"
            
        # Construct the prompt for Llama 3.2
        prompt = f"""<|system|>
You are Sesame, a helpful, friendly and concise voice assistant. 
Keep your responses conversational, natural, and to the point.
Respond to the user's latest message in the context of the conversation.
<|end|>

{history_str}
User: {text}
Assistant:"""

        logger.debug(f"LLM Prompt: {prompt}")
        
        # Generate response with the LLM
        result = llm(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Extract the generated text
        response = result[0]["generated_text"]
        
        # Extract just the Assistant's response (after the prompt)
        response = response.split("Assistant:")[-1].strip()
        
        # Clean up and ensure it's not too long for TTS
        response = response.split("\n")[0].strip()
        if len(response) > 200:
            response = response[:197] + "..."
            
        logger.info(f"LLM response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        # Fall back to simple responses
        return generate_simple_response(text)

def generate_simple_response(text: str) -> str:
    """Generate a simple rule-based response as fallback"""
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
        return f"I heard you say something about that. Can you tell me more?"

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
        
        # Process audio to generate a response (using speech recognition)
        generated_text = process_speech(full_audio, client_id)
        
        # Add suffix for incomplete utterances
        if is_incomplete:
            generated_text += " (processing continued speech...)"
        
        # Log the generated text
        logger.info(f"[{client_id[:8]}] Generated text: '{generated_text}'")
        
        # Handle the result
        if generated_text:
            # Add user message to context
            user_segment = Segment(text=generated_text, speaker=speaker_id, audio=full_audio)
            client['context_segments'].append(user_segment)
            
            # Send the text to client
            emit('transcription', {
                'type': 'transcription',
                'text': generated_text
            }, room=client_id)
            
            # Only generate a response if this is a complete utterance
            if not is_incomplete:
                # Generate a contextual response
                response_text = generate_response(generated_text, client['context_segments'])
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
                    
                    # CHANGE HERE: Use the streaming function instead of sending all at once
                    # Check if the audio is short enough to send at once or if it should be streamed
                    if audio_tensor.size(0) < generator.sample_rate * 2:  # Less than 2 seconds
                        # For short responses, just send in one go for better responsiveness
                        audio_base64 = encode_audio_data(audio_tensor)
                        emit('audio_response', {
                            'type': 'audio_response',
                            'text': response_text,
                            'audio': audio_base64
                        }, room=client_id)
                        logger.info(f"[{client_id[:8]}] Short audio response sent in one piece")
                    else:
                        # For longer responses, use streaming
                        logger.info(f"[{client_id[:8]}] Using streaming for audio response")
                        # Start a new thread for streaming to avoid blocking the main thread
                        import threading
                        stream_thread = threading.Thread(
                            target=stream_audio_to_client,
                            args=(client_id, audio_tensor, response_text, ai_speaker_id)
                        )
                        stream_thread.start()
                        
                except Exception as e:
                    logger.error(f"Error generating audio response: {e}")
                    emit('error', {
                        'type': 'error',
                        'message': "Sorry, there was an error generating the audio response."
                    }, room=client_id)
        else:
            # If processing failed, send a notification
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
    print(f"🧠 Models: Sesame CSM (TTS only)")
    print(f"🔧 Serving from: {os.path.join(base_dir, 'index.html')}")
    print(f"{'='*60}")
    print(f"Ready to receive connections! Press Ctrl+C to stop the server.\n")
    
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)