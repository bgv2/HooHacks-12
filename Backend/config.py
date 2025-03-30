from pathlib import Path

class Config:
    def __init__(self):
        self.MODEL_PATH = Path("path/to/your/model")
        self.AUDIO_MODEL_PATH = Path("path/to/your/audio/model")
        self.WATERMARK_KEY = "your_watermark_key"
        self.SOCKETIO_CORS = "*"
        self.API_KEY = "your_api_key"
        self.DEBUG = True
        self.LOGGING_LEVEL = "INFO"
        self.TTS_SERVICE_URL = "http://localhost:5001/tts"
        self.TRANSCRIPTION_SERVICE_URL = "http://localhost:5002/transcribe"