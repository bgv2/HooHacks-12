# filepath: /csm-conversation-bot/csm-conversation-bot/src/utils/config.py

import os

class Config:
    # General configuration
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key_here')

    # API configuration
    API_URL = os.getenv('API_URL', 'http://localhost:5000')

    # Model configuration
    LLM_MODEL_PATH = os.getenv('LLM_MODEL_PATH', 'path/to/llm/model')
    AUDIO_MODEL_PATH = os.getenv('AUDIO_MODEL_PATH', 'path/to/audio/model')

    # Socket.IO configuration
    SOCKETIO_MESSAGE_QUEUE = os.getenv('SOCKETIO_MESSAGE_QUEUE', 'redis://localhost:6379/0')

    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # Other configurations can be added as needed