"""
CSM Voice Chat Server
A voice chat application that uses CSM 1B for voice synthesis,
WhisperX for speech recognition, and Llama 3.2 for language generation.
"""

# Start the Flask application
from api.app import app, socketio

if __name__ == '__main__':
    import os
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting server on port {port} (debug={debug_mode})")
    print("Visit http://localhost:5000 to access the application")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=debug_mode, allow_unsafe_werkzeug=True)