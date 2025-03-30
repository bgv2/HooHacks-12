from flask import Blueprint, request
from flask_socketio import SocketIO, emit
from src.audio.processor import process_audio
from src.services.transcription_service import TranscriptionService
from src.services.tts_service import TextToSpeechService

streaming_bp = Blueprint('streaming', __name__)
socketio = SocketIO()

transcription_service = TranscriptionService()
tts_service = TextToSpeechService()

@socketio.on('audio_stream')
def handle_audio_stream(data):
    audio_chunk = data['audio']
    speaker_id = data['speaker']
    
    # Process the audio chunk
    processed_audio = process_audio(audio_chunk)
    
    # Transcribe the audio to text
    transcription = transcription_service.transcribe(processed_audio)
    
    # Generate a response using the LLM
    response_text = generate_response(transcription, speaker_id)
    
    # Convert the response text back to audio
    response_audio = tts_service.convert_text_to_speech(response_text, speaker_id)
    
    # Emit the response audio back to the client
    emit('audio_response', {'audio': response_audio})

def generate_response(transcription, speaker_id):
    # Placeholder for the actual response generation logic
    return f"Response to: {transcription}"