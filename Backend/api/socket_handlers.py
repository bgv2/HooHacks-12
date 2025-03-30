from flask import request
from flask_socketio import SocketIO, emit
from src.audio.processor import process_audio
from src.services.transcription_service import TranscriptionService
from src.services.tts_service import TextToSpeechService
from src.llm.generator import load_csm_1b

socketio = SocketIO()

transcription_service = TranscriptionService()
tts_service = TextToSpeechService()
generator = load_csm_1b()

@socketio.on('audio_stream')
def handle_audio_stream(data):
    audio_data = data['audio']
    speaker_id = data['speaker']
    
    # Process the incoming audio
    processed_audio = process_audio(audio_data)
    
    # Transcribe the audio to text
    transcription = transcription_service.transcribe(processed_audio)
    
    # Generate a response using the LLM
    response_text = generator.generate(text=transcription, speaker=speaker_id)
    
    # Convert the response text back to audio
    response_audio = tts_service.convert_text_to_speech(response_text)
    
    # Emit the response audio back to the client
    emit('audio_response', {'audio': response_audio, 'speaker': speaker_id})