import os
import logging
import threading
from dataclasses import dataclass
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure device
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

logger.info(f"Using device: {DEVICE}")

# Initialize Flask app
app = Flask(__name__, static_folder='../', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120)

# Global variables for conversation state
active_conversations = {}
user_queues = {}
processing_threads = {}

# Model storage
@dataclass
class AppModels:
    generator = None
    tokenizer = None
    llm = None
    whisperx_model = None
    whisperx_align_model = None
    whisperx_align_metadata = None
    last_language = None

models = AppModels()

def load_models():
    """Load all required models"""
    from generator import load_csm_1b
    import whisperx
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Import routes and socket handlers
from api.routes import register_routes
from api.socket_handlers import register_handlers

# Register routes and socket handlers
register_routes(app)
register_handlers(socketio, app, models, active_conversations, user_queues, processing_threads, DEVICE)

# Run server if executed directly
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting server on port {port} (debug={debug_mode})")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug_mode, allow_unsafe_werkzeug=True)