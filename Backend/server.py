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

# Add this at the top of your file, replacing your current CUDA setup

# CUDA setup with robust error handling
try:
    # Handle CUDA issues
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Limit to first GPU only
    
    # Try enabling TF32 precision
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except:
        pass  # Ignore if not supported
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        try:
            # Test CUDA functionality
            x = torch.rand(10, device="cuda")
            y = x + x
            del x, y
            device = "cuda"
            compute_type = "float16"
            print("CUDA is fully functional")
        except Exception as cuda_error:
            print(f"CUDA is available but not working correctly: {str(cuda_error)}")
            device = "cpu"
            compute_type = "int8"
    else:
        device = "cpu"
        compute_type = "int8"
except Exception as e:
    print(f"Error setting up CUDA: {str(e)}")
    device = "cpu"
    compute_type = "int8"

print(f"Using device: {device} with compute type: {compute_type}")

# Initialize the Sesame CSM model with robust error handling
try:
    print(f"Loading Sesame CSM model on {device}...")
    generator = load_csm_1b(device=device)
    print("Sesame CSM model loaded successfully")
except Exception as model_error:
    print(f"Error loading Sesame CSM on {device}: {str(model_error)}")
    if device == "cuda":
        # Try on CPU as fallback
        try:
            print("Trying to load Sesame CSM on CPU instead...")
            device = "cpu"  # Update global device setting
            generator = load_csm_1b(device="cpu")
            print("Sesame CSM model loaded on CPU successfully")
        except Exception as cpu_error:
            print(f"Fatal error - could not load Sesame CSM model: {str(cpu_error)}")
            raise RuntimeError("Failed to load speech synthesis model")
    else:
        # Already tried CPU and it failed
        raise RuntimeError("Failed to load speech synthesis model on any device")

# Replace the WhisperX model loading section

# Initialize WhisperX for ASR with robust error handling
print("Loading WhisperX model...")
asr_model = None  # Initialize to None first to avoid scope issues

try:
    # Always start with the tiny model on CPU for stability
    asr_model = whisperx.load_model("tiny", "cpu", compute_type="int8")
    print("WhisperX 'tiny' model loaded on CPU successfully")
    
    # If CPU works, try CUDA if available
    if device == "cuda":
        try:
            print("Trying to load WhisperX on CUDA...")
            cuda_model = whisperx.load_model("tiny", "cuda", compute_type="float16")
            # Test the model to ensure it works
            test_audio = torch.zeros(16000)  # 1 second of silence at 16kHz
            _ = cuda_model.transcribe(test_audio.numpy(), batch_size=1)
            # If we get here, CUDA works
            asr_model = cuda_model
            print("WhisperX model moved to CUDA successfully")
            
            # Try to upgrade to small model on CUDA
            try:
                small_model = whisperx.load_model("small", "cuda", compute_type="float16")
                # Test it
                _ = small_model.transcribe(test_audio.numpy(), batch_size=1)
                asr_model = small_model
                print("WhisperX 'small' model loaded on CUDA successfully")
            except Exception as upgrade_error:
                print(f"Staying with 'tiny' model on CUDA: {str(upgrade_error)}")
        except Exception as cuda_error:
            print(f"CUDA loading failed, staying with CPU model: {str(cuda_error)}")
except Exception as e:
    print(f"Error loading WhisperX model: {str(e)}")
    # Create a minimal dummy model as last resort
    class DummyModel:
        def __init__(self):
            self.device = "cpu"
        def transcribe(self, *args, **kwargs):
            return {"segments": [{"text": "Speech recognition currently unavailable."}]}
    
    asr_model = DummyModel()
    print("WARNING: Using dummy transcription model - ASR functionality limited")

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
    """Decode base64 audio data to a torch tensor with improved error handling"""
    try:
        # Skip empty audio data
        if not audio_data or len(audio_data) < 100:
            print("Empty or too short audio data received")
            return torch.zeros(generator.sample_rate // 2)  # 0.5 seconds of silence
            
        # Extract the actual base64 content
        if ',' in audio_data:
            # Handle data URL format (data:audio/wav;base64,...)
            audio_data = audio_data.split(',')[1]
            
        # Decode base64 audio data
        try:
            binary_data = base64.b64decode(audio_data)
            print(f"Decoded base64 data: {len(binary_data)} bytes")
            
            # Check if we have enough data for a valid WAV
            if len(binary_data) < 44:  # WAV header is 44 bytes
                print("Data too small to be a valid WAV file")
                return torch.zeros(generator.sample_rate // 2)
        except Exception as e:
            print(f"Base64 decoding error: {str(e)}")
            return torch.zeros(generator.sample_rate // 2)
        
        # Save for debugging
        debug_path = os.path.join(base_dir, "debug_incoming.wav") 
        with open(debug_path, 'wb') as f:
            f.write(binary_data)
        print(f"Saved debug file: {debug_path}")
        
        # Approach 1: Load directly with torchaudio
        try:
            with BytesIO(binary_data) as temp_file:
                temp_file.seek(0)  # Ensure we're at the start of the buffer
                audio_tensor, sample_rate = torchaudio.load(temp_file, format="wav")
                print(f"Direct loading success: shape={audio_tensor.shape}, rate={sample_rate}Hz")
                
                # Check if audio is valid
                if audio_tensor.numel() == 0 or torch.isnan(audio_tensor).any():
                    raise ValueError("Empty or invalid audio tensor detected")
        except Exception as e:
            print(f"Direct loading failed: {str(e)}")
            
            # Approach 2: Try to fix/normalize the WAV data
            try:
                # Sometimes WAV headers can be malformed, attempt to fix
                temp_path = os.path.join(base_dir, "temp_fixing.wav")
                with open(temp_path, 'wb') as f:
                    f.write(binary_data)
                
                # Use a simpler numpy approach as backup
                import numpy as np
                import wave
                
                try:
                    with wave.open(temp_path, 'rb') as wf:
                        n_channels = wf.getnchannels()
                        sample_width = wf.getsampwidth()
                        sample_rate = wf.getframerate()
                        n_frames = wf.getnframes()
                        
                        # Read the frames
                        frames = wf.readframes(n_frames)
                        print(f"Wave reading: channels={n_channels}, rate={sample_rate}Hz, frames={n_frames}")
                        
                        # Convert to numpy and then to torch
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
                        print(f"Successfully converted with numpy: shape={audio_tensor.shape}")
                except Exception as wave_error:
                    print(f"Wave processing failed: {str(wave_error)}")
                    # Try with torchaudio as last resort
                    audio_tensor, sample_rate = torchaudio.load(temp_path, format="wav")
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e2:
                print(f"All WAV loading methods failed: {str(e2)}")
                print("Returning silence as fallback")
                return torch.zeros(generator.sample_rate // 2)
        
        # Ensure audio is the right shape (mono)
        if len(audio_tensor.shape) > 1 and audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0)
        
        # Ensure we have a 1D tensor
        audio_tensor = audio_tensor.squeeze()
            
        # Resample if needed
        if sample_rate != generator.sample_rate:
            try:
                print(f"Resampling from {sample_rate}Hz to {generator.sample_rate}Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=generator.sample_rate
                )
                audio_tensor = resampler(audio_tensor)
            except Exception as e:
                print(f"Resampling error: {str(e)}")
                # If resampling fails, just return the original audio
                # The model can often handle different sample rates
        
        # Normalize audio to avoid issues
        if torch.abs(audio_tensor).max() > 0:
            audio_tensor = audio_tensor / torch.abs(audio_tensor).max()
        
        print(f"Final audio tensor: shape={audio_tensor.shape}, min={audio_tensor.min().item():.4f}, max={audio_tensor.max().item():.4f}")
        return audio_tensor
    except Exception as e:
        print(f"Unhandled error in decode_audio_data: {str(e)}")
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
    """Transcribe audio using WhisperX with robust error handling"""
    global asr_model  # Declare global at the beginning of the function
    
    try:
        # Save the tensor to a temporary file
        temp_path = os.path.join(base_dir, "temp_audio.wav")
        torchaudio.save(temp_path, audio_tensor.unsqueeze(0).cpu(), generator.sample_rate)
        
        print(f"Transcribing audio file: {temp_path} (size: {os.path.getsize(temp_path)} bytes)")
        
        # Load the audio file using whisperx's function
        try:
            audio = whisperx.load_audio(temp_path)
        except Exception as audio_load_error:
            print(f"WhisperX load_audio failed: {str(audio_load_error)}")
            # Fall back to manual loading
            import soundfile as sf
            audio, sr = sf.read(temp_path)
            if sr != 16000:  # WhisperX expects 16kHz audio
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / sr))
        
        # Transcribe with error handling for CUDA issues
        try:
            # Try with original device
            result = asr_model.transcribe(audio, batch_size=8)
        except RuntimeError as cuda_error:
            if "CUDA" in str(cuda_error) or "libcudnn" in str(cuda_error):
                print(f"CUDA error in transcription, falling back to CPU: {str(cuda_error)}")
                
                # Try to load a CPU model as fallback
                try:
                    # Move model to CPU and try again
                    asr_model = whisperx.load_model("tiny", "cpu", compute_type="int8")
                    result = asr_model.transcribe(audio, batch_size=1)
                except Exception as e:
                    print(f"CPU fallback also failed: {str(e)}")
                    return "I'm having trouble processing audio right now."
            else:
                # Re-raise if it's not a CUDA error
                raise
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Get the transcription text
        if result["segments"] and len(result["segments"]) > 0:
            # Combine all segments
            transcription = " ".join([segment["text"] for segment in result["segments"]])
            print(f"Transcription successful: '{transcription.strip()}'")
            return transcription.strip()
        else:
            print("Transcription returned no segments")
            return ""
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        return "I heard something but couldn't understand it."


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
            
            # Handle the transcription result
            if transcribed_text:
                # Add user message to context
                user_segment = Segment(text=transcribed_text, speaker=speaker_id, audio=full_audio)
                client['context_segments'].append(user_segment)
                
                # Send the transcribed text to client
                emit('transcription', {
                    'type': 'transcription',
                    'text': transcribed_text
                })
                
                # Generate a contextual response
                response_text = generate_response(transcribed_text, client['context_segments'])
                print(f"[{client_id}] Generating audio response: '{response_text}'")
                
                # Let the client know we're processing
                emit('processing_status', {
                    'type': 'processing_status',
                    'status': 'generating_audio',
                    'message': 'Generating audio response...'
                })
                
                # Generate audio for the response
                try:
                    # Use a different speaker than the user
                    ai_speaker_id = 1 if speaker_id == 0 else 0
                    
                    # Start audio generation with streaming (chunk by chunk)
                    audio_chunks = []
                    
                    # This version tries to stream the audio generation in smaller chunks
                    # Note: CSM model doesn't natively support incremental generation,
                    # so we're simulating it here for a more responsive UI experience
                    
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
                    })
                    
                    print(f"[{client_id}] Audio response sent: {len(audio_base64)} bytes")
                    
                except Exception as gen_error:
                    print(f"Error generating audio response: {str(gen_error)}")
                    emit('error', {
                        'type': 'error',
                        'message': "Sorry, there was an error generating the audio response."
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
            print(f"[{client_id}] Processing long audio segment without silence")
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
            
            # Keep half of the buffer for context (sliding window approach)
            half_point = len(client['streaming_buffer']) // 2
            client['streaming_buffer'] = client['streaming_buffer'][half_point:]
            
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

def stream_audio_to_client(client_id, audio_tensor, text, speaker_id, chunk_size_ms=500):
    """Stream audio to client in chunks to simulate real-time generation"""
    try:
        if client_id not in active_clients:
            print(f"Client {client_id} not found for streaming")
            return
            
        # Calculate chunk size in samples
        chunk_size = int(generator.sample_rate * chunk_size_ms / 1000)
        total_chunks = math.ceil(audio_tensor.size(0) / chunk_size)
        
        print(f"Streaming audio in {total_chunks} chunks of {chunk_size_ms}ms each")
        
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
        
        print(f"Audio streaming complete: {total_chunks} chunks sent")
        
    except Exception as e:
        print(f"Error streaming audio to client: {str(e)}")
        import traceback
        traceback.print_exc()

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