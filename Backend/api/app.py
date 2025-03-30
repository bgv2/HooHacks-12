from flask import Flask
from flask_socketio import SocketIO
from src.utils.config import Config
from src.utils.logger import setup_logger
from api.routes import setup_routes
from api.socket_handlers import setup_socket_handlers

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    setup_logger(app)
    setup_routes(app)
    setup_socket_handlers(app)

    return app

app = create_app()
socketio = SocketIO(app)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)