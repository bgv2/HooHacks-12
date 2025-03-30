import os
import io
import base64
import time
import threading
import queue
import tempfile
import gc
import logging
import traceback
from typing import Dict, List, Optional

import torch
import torchaudio
import numpy as np
from flask import request
from flask_socketio import emit

# Import conversation model
from generator import Segment

logger = logging.getLogger(__name__)

# Conversation data structure
class Conversation:
    def __init__(self, session_id):
        self.session_id = session_id
        self.segments: List[Segment] = []
        self.current_speaker = 0
        self.ai_speaker_id = 1  # Default AI speaker ID
        self.last_activity = time.time()
        self.is_processing = False
    
    def add_segment(self, text, speaker, audio):
        segment = Segment(text=text, speaker=speaker, audio=audio)
        self.segments.append(segment)
        self.last_activity = time.time()
        return segment
    
    def get_context(self, max_segments=10):
        """Return the most recent segments for context"""
        return self.segments[-max_segments:] if self.segments else []

def register_handlers(socketio, app, models, active_conversations, user_queues, processing_threads, DEVICE):
    """Register Socket.IO event handlers"""
    
    @socketio.on('connect')
    def handle_connect(auth=None):
        """Handle client connection"""
        session_id = request.sid
        logger.info(f"Client connected: {session_id}")
        
        # Initialize conversation data
        if session_id not in active_conversations:
            active_conversations[session_id] = Conversation(session_id)
            user_queues[session_id] = queue.Queue()
            processing_threads[session_id] = threading.Thread(
                target=process_audio_queue, 
                args=(session_id, user_queues[session_id], app, socketio, models, active_conversations, DEVICE),
                daemon=True
            )
            processing_threads[session_id].start()
        
        emit('connection_status', {'status': 'connected'})
    
    @socketio.on('disconnect')
    def handle_disconnect(reason=None):
        """Handle client disconnection"""
        session_id = request.sid
        logger.info(f"Client disconnected: {session_id}. Reason: {reason}")
        
        # Cleanup
        if session_id in active_conversations:
            # Mark for deletion rather than immediately removing
            # as the processing thread might still be accessing it
            active_conversations[session_id].is_processing = False
            user_queues[session_id].put(None)  # Signal thread to terminate
    
    @socketio.on('audio_data')
    def handle_audio_data(data):
        """Handle incoming audio data"""
        session_id = request.sid
        logger.info(f"Received audio data from {session_id}")
        
        # Check if the models are loaded
        if models.generator is None or models.whisperx_model is None or models.llm is None:
            emit('error', {'message': 'Models still loading, please wait'})
            return
        
        # Check if we're already processing for this session
        if session_id in active_conversations and active_conversations[session_id].is_processing:
            emit('error', {'message': 'Still processing previous audio, please wait'})
            return
        
        # Add to processing queue
        if session_id in user_queues:
            user_queues[session_id].put(data)
        else:
            emit('error', {'message': 'Session not initialized, please refresh the page'})

def process_audio_queue(session_id, q, app, socketio, models, active_conversations, DEVICE):
    """Background thread to process audio chunks for a session"""
    logger.info(f"Started processing thread for session: {session_id}")
    
    try:
        while session_id in active_conversations:
            try:
                # Get the next audio chunk with a timeout
                data = q.get(timeout=120)
                if data is None:  # Termination signal
                    break
                
                # Process the audio and generate a response
                process_audio_and_respond(session_id, data, app, socketio, models, active_conversations, DEVICE)
                
            except queue.Empty:
                # Timeout, check if session is still valid
                continue
            except Exception as e:
                logger.error(f"Error processing audio for {session_id}: {str(e)}")
                # Create an app context for the socket emit
                with app.app_context():
                    socketio.emit('error', {'message': str(e)}, room=session_id)
    finally:
        logger.info(f"Ending processing thread for session: {session_id}")
        # Clean up when thread is done
        with app.app_context():
            if session_id in active_conversations:
                del active_conversations[session_id]
            if session_id in user_queues:
                del user_queues[session_id]

def process_audio_and_respond(session_id, data, app, socketio, models, active_conversations, DEVICE):
    """Process audio data and generate a response using WhisperX"""
    if models.generator is None or models.whisperx_model is None or models.llm is None:
        logger.warning("Models not yet loaded!")
        with app.app_context():
            socketio.emit('error', {'message': 'Models still loading, please wait'}, room=session_id)
        return
    
    logger.info(f"Processing audio for session {session_id}")
    conversation = active_conversations[session_id]
    
    try:
        # Set processing flag
        conversation.is_processing = True
        
        # Process base64 audio data
        audio_data = data['audio']
        speaker_id = data['speaker']
        logger.info(f"Received audio from speaker {speaker_id}")
        
        # Convert from base64 to WAV
        try:
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            logger.info(f"Decoded audio bytes: {len(audio_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error decoding base64 audio: {str(e)}")
            raise
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # Notify client that transcription is starting
            with app.app_context():
                socketio.emit('processing_status', {'status': 'transcribing'}, room=session_id)
            
            # Load audio using WhisperX
            import whisperx
            audio = whisperx.load_audio(temp_path)
            
            # Check audio length and add a warning for short clips
            audio_length = len(audio) / 16000  # assuming 16kHz sample rate
            if audio_length < 1.0:
                logger.warning(f"Audio is very short ({audio_length:.2f}s), may affect transcription quality")
            
            # Transcribe using WhisperX
            batch_size = 16  # adjust based on your GPU memory
            logger.info("Running WhisperX transcription...")
            
            # Handle the warning about audio being shorter than 30s by suppressing it
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="audio is shorter than 30s")
                result = models.whisperx_model.transcribe(audio, batch_size=batch_size)
            
            # Get the detected language
            language_code = result["language"]
            logger.info(f"Detected language: {language_code}")
            
            # Check if alignment model needs to be loaded or updated
            if models.whisperx_align_model is None or language_code != models.last_language:
                # Clean up old models if they exist
                if models.whisperx_align_model is not None:
                    del models.whisperx_align_model
                    del models.whisperx_align_metadata
                    if DEVICE == "cuda":
                        gc.collect()
                        torch.cuda.empty_cache()
                
                # Load new alignment model for the detected language
                logger.info(f"Loading alignment model for language: {language_code}")
                models.whisperx_align_model, models.whisperx_align_metadata = whisperx.load_align_model(
                    language_code=language_code, device=DEVICE
                )
                models.last_language = language_code
            
            # Align the transcript to get word-level timestamps
            if result["segments"] and len(result["segments"]) > 0:
                logger.info("Aligning transcript...")
                result = whisperx.align(
                    result["segments"], 
                    models.whisperx_align_model, 
                    models.whisperx_align_metadata, 
                    audio, 
                    DEVICE, 
                    return_char_alignments=False
                )
                
                # Process the segments for better output
                for segment in result["segments"]:
                    # Round timestamps for better display
                    segment["start"] = round(segment["start"], 2)
                    segment["end"] = round(segment["end"], 2)
                    # Add a confidence score if not present
                    if "confidence" not in segment:
                        segment["confidence"] = 1.0  # Default confidence
            
            # Extract the full text from all segments
            user_text = ' '.join([segment['text'] for segment in result['segments']])
            
            # If no text was recognized, don't process further
            if not user_text or len(user_text.strip()) == 0:
                with app.app_context():
                    socketio.emit('error', {'message': 'No speech detected'}, room=session_id)
                return
                
            logger.info(f"Transcription: {user_text}")
            
            # Load audio for CSM input
            waveform, sample_rate = torchaudio.load(temp_path)
            
            # Normalize to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to the CSM sample rate if needed
            if sample_rate != models.generator.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, 
                    orig_freq=sample_rate, 
                    new_freq=models.generator.sample_rate
                )
            
            # Add the user's message to conversation history
            user_segment = conversation.add_segment(
                text=user_text,
                speaker=speaker_id,
                audio=waveform.squeeze()
            )
            
            # Send transcription to client with detailed segments
            with app.app_context():
                socketio.emit('transcription', {
                    'text': user_text, 
                    'speaker': speaker_id,
                    'segments': result['segments']  # Include the detailed segments with timestamps
                }, room=session_id)
            
            # Generate AI response using Llama
            with app.app_context():
                socketio.emit('processing_status', {'status': 'generating'}, room=session_id)
            
            # Create prompt from conversation history
            conversation_history = ""
            for segment in conversation.segments[-5:]:  # Last 5 segments for context
                role = "User" if segment.speaker == 0 else "Assistant"
                conversation_history += f"{role}: {segment.text}\n"
            
            # Add final prompt
            prompt = f"{conversation_history}Assistant: "
            
            # Generate response with Llama
            try:
                # Ensure pad token is set
                if models.tokenizer.pad_token is None:
                    models.tokenizer.pad_token = models.tokenizer.eos_token
                    
                input_tokens = models.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True
                )
                input_ids = input_tokens.input_ids.to(DEVICE)
                attention_mask = input_tokens.attention_mask.to(DEVICE)

                with torch.no_grad():
                    generated_ids = models.llm.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=100,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=models.tokenizer.eos_token_id
                    )
                
                # Decode the response
                response_text = models.tokenizer.decode(
                    generated_ids[0][input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                logger.error(traceback.format_exc())
                response_text = "I'm sorry, I encountered an error while processing your request."
            
            # Synthesize speech
            with app.app_context():
                socketio.emit('processing_status', {'status': 'synthesizing'}, room=session_id)
                
                # Start sending the audio response
                socketio.emit('audio_response_start', {
                    'text': response_text,
                    'total_chunks': 1,
                    'chunk_index': 0
                }, room=session_id)
            
            # Define AI speaker ID
            ai_speaker_id = conversation.ai_speaker_id
            
            # Generate audio
            audio_tensor = models.generator.generate(
                text=response_text,
                speaker=ai_speaker_id,
                context=conversation.get_context(),
                max_audio_length_ms=10_000,
                temperature=0.9
            )
            
            # Add AI response to conversation history
            ai_segment = conversation.add_segment(
                text=response_text,
                speaker=ai_speaker_id,
                audio=audio_tensor
            )
            
            # Convert audio to WAV format
            with io.BytesIO() as wav_io:
                torchaudio.save(
                    wav_io, 
                    audio_tensor.unsqueeze(0).cpu(), 
                    models.generator.sample_rate, 
                    format="wav"
                )
                wav_io.seek(0)
                wav_data = wav_io.read()
            
            # Convert WAV data to base64
            audio_base64 = f"data:audio/wav;base64,{base64.b64encode(wav_data).decode('utf-8')}"
            
            # Send audio chunk to client
            with app.app_context():
                socketio.emit('audio_response_chunk', {
                    'chunk': audio_base64,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'is_last': True
                }, room=session_id)
                
                # Signal completion
                socketio.emit('audio_response_complete', {
                    'text': response_text
                }, room=session_id)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())
        with app.app_context():
            socketio.emit('error', {'message': f'Error: {str(e)}'}, room=session_id)
    finally:
        # Reset processing flag
        conversation.is_processing = False