from flask import Blueprint, request, jsonify
from src.services.transcription_service import TranscriptionService
from src.services.tts_service import TextToSpeechService

api = Blueprint('api', __name__)

transcription_service = TranscriptionService()
tts_service = TextToSpeechService()

@api.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_data = request.files.get('audio')
    if not audio_data:
        return jsonify({'error': 'No audio file provided'}), 400
    
    text = transcription_service.transcribe(audio_data)
    return jsonify({'transcription': text})

@api.route('/generate-response', methods=['POST'])
def generate_response():
    data = request.json
    user_input = data.get('input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    response_text = tts_service.generate_response(user_input)
    audio_data = tts_service.text_to_speech(response_text)
    
    return jsonify({'response': response_text, 'audio': audio_data})