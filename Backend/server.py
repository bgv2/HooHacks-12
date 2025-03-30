import os
import logging
import torch
import eventlet
import base64
import tempfile
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import whisper
import torchaudio
from src.models.conversation import Segment
from src.services.tts_service import load_csm_1b
from src.llm.generator import generate_llm_response
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.audio.streaming import AudioStreamer
from src.services.transcription_service import TranscriptionService
from src.services.tts_service import TextToSpeechService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
socketio = SocketIO(app)

# Initialize services
transcription_service = TranscriptionService()
tts_service = TextToSpeechService()
audio_streamer = AudioStreamer()

@socketio.on('audio_input')
def handle_audio_input(data):
    audio_chunk = data['audio']
    speaker_id = data['speaker']
    
    # Process audio and convert to text
    text = transcription_service.transcribe(audio_chunk)
    logging.info(f"Transcribed text: {text}")

    # Generate response using Llama 3.2
    response_text = tts_service.generate_response(text, speaker_id)
    logging.info(f"Generated response: {response_text}")

    # Convert response text to audio
    audio_response = tts_service.text_to_speech(response_text, speaker_id)
    
    # Stream audio response back to client
    socketio.emit('audio_response', {'audio': audio_response})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)