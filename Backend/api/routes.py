import os
import torch
import psutil
from flask import send_from_directory, jsonify, request

def register_routes(app):
    """Register HTTP routes for the application"""
    
    @app.route('/')
    def index():
        """Serve the main application page"""
        return send_from_directory(app.static_folder, 'index.html')

    @app.route('/voice-chat.js')
    def serve_js():
        """Serve the JavaScript file"""
        return send_from_directory(app.static_folder, 'voice-chat.js')

    @app.route('/api/status')
    def system_status():
        """Return the system status"""
        # Import here to avoid circular imports
        from api.app import models, DEVICE
        
        return jsonify({
            "status": "ok",
            "cuda_available": torch.cuda.is_available(),
            "device": DEVICE,
            "models": {
                "generator": models.generator is not None,
                "asr": models.whisperx_model is not None,
                "llm": models.llm is not None
            },
            "versions": {
                "transformers": "4.49.0",  # Replace with actual version
                "torch": torch.__version__
            }
        })

    @app.route('/api/system_resources')
    def system_resources():
        """Return system resource usage"""
        # Import here to avoid circular imports
        from api.app import active_conversations, DEVICE
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024 ** 3)
        memory_total_gb = memory.total / (1024 ** 3)
        memory_percent = memory.percent
        
        # Get GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f"gpu_{i}"] = {
                    "allocated": torch.cuda.memory_allocated(i) / (1024 ** 3),
                    "reserved": torch.cuda.memory_reserved(i) / (1024 ** 3),
                    "max_allocated": torch.cuda.max_memory_allocated(i) / (1024 ** 3)
                }
        
        return jsonify({
            "cpu_percent": cpu_percent,
            "memory": {
                "used_gb": memory_used_gb,
                "total_gb": memory_total_gb,
                "percent": memory_percent
            },
            "gpu_memory": gpu_memory,
            "active_sessions": len(active_conversations)
        })