/**
 * CSM AI Voice Chat Client
 * 
 * A web client that connects to a CSM AI voice chat server and enables 
 * real-time voice conversation with an AI assistant.
 */

// Configuration constants
const SERVER_URL = window.location.hostname === 'localhost' ? 
    'http://localhost:5000' : window.location.origin;
const ENERGY_WINDOW_SIZE = 15;
const CLIENT_SILENCE_DURATION_MS = 750;

// DOM elements
const elements = {
    conversation: document.getElementById('conversation'),
    streamButton: document.getElementById('streamButton'),
    clearButton: document.getElementById('clearButton'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    visualizerCanvas: document.getElementById('audioVisualizer'),
    visualizerLabel: document.getElementById('visualizerLabel'),
    volumeLevel: document.getElementById('volumeLevel'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    speakerSelection: document.getElementById('speakerSelect'),
    autoPlayResponses: document.getElementById('autoPlayResponses'),
    showVisualizer: document.getElementById('showVisualizer')
};

// Application state
const state = {
    socket: null,
    audioContext: null,
    analyser: null,
    microphone: null,
    streamProcessor: null,
    isStreaming: false,
    isSpeaking: false,
    silenceThreshold: 0.01,
    energyWindow: [],
    silenceTimer: null,
    volumeUpdateInterval: null,
    visualizerAnimationFrame: null,
    currentSpeaker: 0,
    aiSpeakerId: 1  // Define the AI's speaker ID to match server.py
};

// Visualizer variables
let canvasContext = null;
let visualizerBufferLength = 0;
let visualizerDataArray = null;

// Audio streaming state
const streamingAudio = {
    messageElement: null,
    audioElement: null,
    chunks: [],
    totalChunks: 0,
    receivedChunks: 0,
    text: '',
    complete: false
};

// Initialize the application
function initializeApp() {
    // Initialize the UI elements
    initializeUIElements();
    
    // Initialize socket.io connection
    setupSocketConnection();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize visualizer
    setupVisualizer();
    
    // Show welcome message
    addSystemMessage('Welcome to CSM Voice Chat! Click "Start Conversation" to begin.');
}

// Initialize UI elements
function initializeUIElements() {
    // Update threshold display
    if (elements.thresholdValue) {
        elements.thresholdValue.textContent = state.silenceThreshold.toFixed(3);
    }
}

// Setup Socket.IO connection
function setupSocketConnection() {
    state.socket = io(SERVER_URL);
    
    // Connection events
    state.socket.on('connect', () => {
        updateConnectionStatus(true);
        addSystemMessage('Connected to server.');
    });
    
    state.socket.on('disconnect', () => {
        updateConnectionStatus(false);
        addSystemMessage('Disconnected from server.');
        stopStreaming(false);
    });
    
    state.socket.on('error', (data) => {
        console.error('Server error:', data.message);
        
        // Make the error more user-friendly
        let userMessage = data.message;
        
        // Check for common errors and provide more helpful messages
        if (data.message.includes('Models still loading')) {
            userMessage = 'The AI models are still loading. Please wait a moment and try again.';
        } else if (data.message.includes('No speech detected')) {
            userMessage = 'No speech was detected. Please speak clearly and try again.';
        }
        
        addSystemMessage(`Error: ${userMessage}`);
        
        // Reset button state if it was processing
        if (elements.streamButton.classList.contains('processing')) {
            elements.streamButton.classList.remove('processing');
            elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Start Conversation';
        }
    });
    
    // Register message handlers
    state.socket.on('transcription', handleTranscription);
    state.socket.on('context_updated', handleContextUpdate);
    state.socket.on('streaming_status', handleStreamingStatus);
    state.socket.on('processing_status', handleProcessingStatus);
    
    // Add model status handlers
    state.socket.on('model_status', handleModelStatusUpdate);

    // Handlers for incremental audio streaming
    state.socket.on('audio_response_start', handleAudioResponseStart);
    state.socket.on('audio_response_chunk', handleAudioResponseChunk);
    state.socket.on('audio_response_complete', handleAudioResponseComplete);
}

// Setup event listeners
function setupEventListeners() {
    // Stream button
    elements.streamButton.addEventListener('click', toggleStreaming);
    
    // Clear button
    elements.clearButton.addEventListener('click', clearConversation);
    
    // Threshold slider
    if (elements.thresholdSlider) {
        elements.thresholdSlider.addEventListener('input', updateThreshold);
    }
    
    // Speaker selection
    elements.speakerSelection.addEventListener('change', () => {
        state.currentSpeaker = parseInt(elements.speakerSelection.value);
    });
    
    // Visualizer toggle
    if (elements.showVisualizer) {
        elements.showVisualizer.addEventListener('change', toggleVisualizerVisibility);
    }
}

// Setup audio visualizer
function setupVisualizer() {
    if (!elements.visualizerCanvas) return;
    
    canvasContext = elements.visualizerCanvas.getContext('2d');
    
    // Set canvas dimensions
    elements.visualizerCanvas.width = elements.visualizerCanvas.offsetWidth;
    elements.visualizerCanvas.height = elements.visualizerCanvas.offsetHeight;
    
    // Initialize visualization data array
    visualizerDataArray = new Uint8Array(128);
    
    // Start the visualizer animation
    drawVisualizer();
}

// Update connection status UI
function updateConnectionStatus(isConnected) {
    if (isConnected) {
        elements.statusDot.classList.add('active');
        elements.statusText.textContent = 'Connected';
    } else {
        elements.statusDot.classList.remove('active');
        elements.statusText.textContent = 'Disconnected';
    }
}

// Toggle streaming state
function toggleStreaming() {
    if (state.isStreaming) {
        stopStreaming();
    } else {
        startStreaming();
    }
}

// Start streaming audio to the server
function startStreaming() {
    if (!state.socket || !state.socket.connected) {
        addSystemMessage('Not connected to server. Please refresh the page.');
        return;
    }
    
    // Check if models are loaded via the API
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            if (!data.models.generator || !data.models.asr || !data.models.llm) {
                addSystemMessage('Still loading AI models. Please wait...');
                return;
            }
            
            // Continue with recording if models are loaded
            initializeRecording();
        })
        .catch(error => {
            console.error('Error checking model status:', error);
            // Try anyway, the server will respond with an error if models aren't ready
            initializeRecording();
        });
}

// Extracted the recording initialization to a separate function
function initializeRecording() {
    // Request microphone access
    navigator.mediaDevices.getUserMedia({ audio: true, video: false })
        .then(stream => {
            state.isStreaming = true;
            elements.streamButton.classList.add('recording');
            elements.streamButton.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
            
            // Initialize Web Audio API
            state.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            state.microphone = state.audioContext.createMediaStreamSource(stream);
            state.analyser = state.audioContext.createAnalyser();
            state.analyser.fftSize = 2048;
            
            // Setup analyzer for visualizer
            visualizerBufferLength = state.analyser.frequencyBinCount;
            visualizerDataArray = new Uint8Array(visualizerBufferLength);
            
            state.microphone.connect(state.analyser);
            
            // Create processor node for audio data
            const processorNode = state.audioContext.createScriptProcessor(4096, 1, 1);
            processorNode.onaudioprocess = handleAudioProcess;
            state.analyser.connect(processorNode);
            processorNode.connect(state.audioContext.destination);
            state.streamProcessor = processorNode;
            
            state.silenceTimer = null;
            state.energyWindow = [];
            state.isSpeaking = false;
            
            // Notify server
            state.socket.emit('start_stream');
            
            // Start volume meter updates
            state.volumeUpdateInterval = setInterval(updateVolumeMeter, 100);
            
            // Make sure visualizer is visible if enabled
            if (elements.showVisualizer && elements.showVisualizer.checked) {
                elements.visualizerLabel.style.opacity = '0';
            }
            
            addSystemMessage('Recording started. Speak now...');
        })
        .catch(error => {
            console.error('Error accessing microphone:', error);
            addSystemMessage('Could not access microphone. Please check permissions.');
        });
}

// Stop streaming audio
function stopStreaming(notifyServer = true) {
    if (state.isStreaming) {
        state.isStreaming = false;
        elements.streamButton.classList.remove('recording');
        elements.streamButton.classList.remove('processing');
        elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Start Conversation';
        
        // Clean up audio resources
        if (state.streamProcessor) {
            state.streamProcessor.disconnect();
            state.streamProcessor = null;
        }
        
        if (state.analyser) {
            state.analyser.disconnect();
            state.analyser = null;
        }
        
        if (state.microphone) {
            state.microphone.disconnect();
            state.microphone = null;
        }
        
        if (state.audioContext) {
            state.audioContext.close().catch(err => console.warn('Error closing audio context:', err));
            state.audioContext = null;
        }
        
        // Clear any pending silence timer
        if (state.silenceTimer) {
            clearTimeout(state.silenceTimer);
            state.silenceTimer = null;
        }
        
        // Clear volume meter updates
        if (state.volumeUpdateInterval) {
            clearInterval(state.volumeUpdateInterval);
            state.volumeUpdateInterval = null;
            
            // Reset volume meter
            if (elements.volumeLevel) {
                elements.volumeLevel.style.width = '0%';
            }
        }
        
        // Show visualizer label
        if (elements.visualizerLabel) {
            elements.visualizerLabel.style.opacity = '0.7';
        }
        
        // Notify server if needed
        if (notifyServer && state.socket && state.socket.connected) {
            state.socket.emit('stop_stream');
        }
        
        addSystemMessage('Recording stopped.');
    }
}

// Handle audio processing
function handleAudioProcess(event) {
    if (!state.isStreaming) return;
    
    const inputData = event.inputBuffer.getChannelData(0);
    const energy = calculateAudioEnergy(inputData);
    updateEnergyWindow(energy);
    
    const averageEnergy = calculateAverageEnergy();
    const isSilent = averageEnergy < state.silenceThreshold;
    
    handleSpeechState(isSilent);
}

// Calculate audio energy (volume)
function calculateAudioEnergy(buffer) {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
        sum += buffer[i] * buffer[i];
    }
    return Math.sqrt(sum / buffer.length);
}

// Update energy window for averaging
function updateEnergyWindow(energy) {
    state.energyWindow.push(energy);
    if (state.energyWindow.length > ENERGY_WINDOW_SIZE) {
        state.energyWindow.shift();
    }
}

// Calculate average energy from window
function calculateAverageEnergy() {
    if (state.energyWindow.length === 0) return 0;
    
    const sum = state.energyWindow.reduce((acc, val) => acc + val, 0);
    return sum / state.energyWindow.length;
}

// Update the threshold from the slider
function updateThreshold() {
    state.silenceThreshold = parseFloat(elements.thresholdSlider.value);
    elements.thresholdValue.textContent = state.silenceThreshold.toFixed(3);
}

// Update the volume meter display
function updateVolumeMeter() {
    if (!state.isStreaming || !state.energyWindow.length || !elements.volumeLevel) return;
    
    const avgEnergy = calculateAverageEnergy();
    
    // Scale energy to percentage (0-100)
    // Energy values are typically very small (e.g., 0.001 to 0.1)
    const scaleFactor = 1000;
    const percentage = Math.min(100, Math.max(0, avgEnergy * scaleFactor));
    
    // Update volume meter width
    elements.volumeLevel.style.width = `${percentage}%`;
    
    // Change color based on level
    if (percentage > 70) {
        elements.volumeLevel.style.backgroundColor = '#ff5252';
    } else if (percentage > 30) {
        elements.volumeLevel.style.backgroundColor = '#4CAF50';
    } else {
        elements.volumeLevel.style.backgroundColor = '#4c84ff';
    }
}

// Handle speech/silence state transitions
function handleSpeechState(isSilent) {
    if (state.isSpeaking) {
        if (isSilent) {
            // User was speaking but now is silent
            if (!state.silenceTimer) {
                state.silenceTimer = setTimeout(() => {
                    // Silence lasted long enough, consider speech done
                    if (state.isSpeaking) {
                        state.isSpeaking = false;
                        
                        try {
                            // Get the current audio data and send it
                            const audioBuffer = new Float32Array(state.audioContext.sampleRate * 5); // 5 seconds max
                            state.analyser.getFloatTimeDomainData(audioBuffer);
                            
                            // Check if audio has content
                            const hasAudioContent = audioBuffer.some(sample => Math.abs(sample) > 0.01);
                            
                            if (!hasAudioContent) {
                                console.warn('Audio buffer appears to be empty or very quiet');
                                addSystemMessage('No speech detected. Please try again and speak clearly.');
                                return;
                            }
                            
                            // Create WAV blob
                            const wavBlob = createWavBlob(audioBuffer, state.audioContext.sampleRate);
                            
                            // Convert to base64
                            const reader = new FileReader();
                            reader.onloadend = function() {
                                sendAudioChunk(reader.result, state.currentSpeaker);
                            };
                            reader.readAsDataURL(wavBlob);
                            
                            // Update button state
                            elements.streamButton.classList.add('processing');
                            elements.streamButton.innerHTML = '<i class="fas fa-cog fa-spin"></i> Processing...';
                            
                            addSystemMessage('Processing your message...');
                        } catch (e) {
                            console.error('Error recording audio:', e);
                            addSystemMessage('Error recording audio. Please try again.');
                        }
                    }
                }, CLIENT_SILENCE_DURATION_MS);
            }
        } else {
            // User is still speaking, reset silence timer
            if (state.silenceTimer) {
                clearTimeout(state.silenceTimer);
                state.silenceTimer = null;
            }
        }
    } else {
        if (!isSilent) {
            // User started speaking
            state.isSpeaking = true;
            if (state.silenceTimer) {
                clearTimeout(state.silenceTimer);
                state.silenceTimer = null;
            }
        }
    }
}

// Send audio chunk to server
function sendAudioChunk(audioData, speaker) {
    if (state.socket && state.socket.connected) {
        state.socket.emit('audio_chunk', {
            audio: audioData,
            speaker: speaker
        });
    }
}

// Create WAV blob from audio data
function createWavBlob(audioData, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    
    // Create buffer for WAV file
    const buffer = new ArrayBuffer(44 + audioData.length * bytesPerSample);
    const view = new DataView(buffer);
    
    // Write WAV header
    // "RIFF" chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + audioData.length * bytesPerSample, true);
    writeString(view, 8, 'WAVE');
    
    // "fmt " sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // subchunk1size
    view.setUint16(20, 1, true); // audio format (PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bytesPerSample, true); // byte rate
    view.setUint16(32, numChannels * bytesPerSample, true); // block align
    view.setUint16(34, bitsPerSample, true);
    
    // "data" sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, audioData.length * bytesPerSample, true);
    
    // Write audio data
    const audioDataStart = 44;
    for (let i = 0; i < audioData.length; i++) {
        const sample = Math.max(-1, Math.min(1, audioData[i]));
        const value = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(audioDataStart + i * bytesPerSample, value, true);
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
}

// Helper function to write strings to DataView
function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// Clear conversation history
function clearConversation() {
    elements.conversation.innerHTML = '';
    if (state.socket && state.socket.connected) {
        state.socket.emit('clear_context');
    }
    addSystemMessage('Conversation cleared.');
}

// Draw audio visualizer
function drawVisualizer() {
    if (!canvasContext || !elements.visualizerCanvas) {
        state.visualizerAnimationFrame = requestAnimationFrame(drawVisualizer);
        return;
    }
    
    state.visualizerAnimationFrame = requestAnimationFrame(drawVisualizer);
    
    // Skip drawing if visualizer is hidden or not enabled
    if (elements.showVisualizer && !elements.showVisualizer.checked) {
        if (elements.visualizerCanvas.style.opacity !== '0') {
            elements.visualizerCanvas.style.opacity = '0';
        }
        return;
    } else if (elements.visualizerCanvas.style.opacity !== '1') {
        elements.visualizerCanvas.style.opacity = '1';
    }
    
    // Get frequency data if available
    if (state.isStreaming && state.analyser) {
        try {
            state.analyser.getByteFrequencyData(visualizerDataArray);
        } catch (e) {
            console.warn('Error getting frequency data:', e);
        }
    } else {
        // Fade out when not streaming
        for (let i = 0; i < visualizerDataArray.length; i++) {
            visualizerDataArray[i] = Math.max(0, visualizerDataArray[i] - 5);
        }
    }
    
    // Clear canvas
    canvasContext.fillStyle = 'rgb(0, 0, 0)';
    canvasContext.fillRect(0, 0, elements.visualizerCanvas.width, elements.visualizerCanvas.height);
    
    // Draw gradient bars
    const width = elements.visualizerCanvas.width;
    const height = elements.visualizerCanvas.height;
    const barCount = Math.min(visualizerBufferLength, 64);
    const barWidth = width / barCount - 1;
    
    for (let i = 0; i < barCount; i++) {
        const index = Math.floor(i * visualizerBufferLength / barCount);
        const value = visualizerDataArray[index];
        
        // Use logarithmic scale for better audio visualization
        const logFactor = 20;
        const scaledValue = Math.log(1 + (value / 255) * logFactor) / Math.log(1 + logFactor);
        const barHeight = scaledValue * height;
        
        // Position bars
        const x = i * (barWidth + 1);
        const y = height - barHeight;
        
        // Create color gradient based on frequency and amplitude
        const hue = i / barCount * 360; // Full color spectrum
        const saturation = 80 + (value / 255 * 20); // Higher values more saturated
        const lightness = 40 + (value / 255 * 20); // Dynamic brightness
        
        // Draw main bar
        canvasContext.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        canvasContext.fillRect(x, y, barWidth, barHeight);
        
        // Add highlight effect
        if (barHeight > 5) {
            const gradient = canvasContext.createLinearGradient(
                x, y, 
                x, y + barHeight * 0.5
            );
            gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness + 20}%, 0.4)`);
            gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness}%, 0)`);
            canvasContext.fillStyle = gradient;
            canvasContext.fillRect(x, y, barWidth, barHeight * 0.5);
            
            // Add highlight on top of the bar
            canvasContext.fillStyle = `hsla(${hue}, ${saturation - 20}%, ${lightness + 30}%, 0.7)`;
            canvasContext.fillRect(x, y, barWidth, 2);
        }
    }
}

// Toggle visualizer visibility
function toggleVisualizerVisibility() {
    const isVisible = elements.showVisualizer.checked;
    elements.visualizerCanvas.style.opacity = isVisible ? '1' : '0';
}

// Add a message to the conversation
function addMessage(text, type) {
    if (!elements.conversation) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const textElement = document.createElement('p');
    textElement.textContent = text;
    messageDiv.appendChild(textElement);
    
    // Add timestamp to every message
    const timestamp = new Date().toLocaleTimeString();
    const timeLabel = document.createElement('div');
    timeLabel.className = 'message-timestamp';
    timeLabel.textContent = timestamp;
    messageDiv.appendChild(timeLabel);
    
    elements.conversation.appendChild(messageDiv);
    
    // Auto-scroll to the bottom
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
    
    return messageDiv;
}

// Add a system message to the conversation
function addSystemMessage(text) {
    if (!elements.conversation) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';
    messageDiv.textContent = text;
    
    elements.conversation.appendChild(messageDiv);
    
    // Auto-scroll to the bottom
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
    
    return messageDiv;
}

// Handle transcription response from server
function handleTranscription(data) {
    const speaker = data.speaker === 0 ? 'user' : 'ai';
    
    // Create the message div
    const messageDiv = addMessage(data.text, speaker);
    
    // If we have detailed segments from WhisperX, add timestamps
    if (data.segments && data.segments.length > 0) {
        // Add a timestamps container
        const timestampsContainer = document.createElement('div');
        timestampsContainer.className = 'timestamps-container';
        timestampsContainer.style.display = 'none'; // Hidden by default
        
        // Add a toggle button
        const toggleButton = document.createElement('button');
        toggleButton.className = 'timestamp-toggle';
        toggleButton.textContent = 'Show Timestamps';
        toggleButton.onclick = function() {
            const isHidden = timestampsContainer.style.display === 'none';
            timestampsContainer.style.display = isHidden ? 'block' : 'none';
            toggleButton.textContent = isHidden ? 'Hide Timestamps' : 'Show Timestamps';
        };
        
        // Add timestamps for each segment
        data.segments.forEach(segment => {
            const timestampDiv = document.createElement('div');
            timestampDiv.className = 'timestamp';
            
            // Format start and end times
            const startTime = formatTime(segment.start);
            const endTime = formatTime(segment.end);
            
            timestampDiv.innerHTML = `
                <span class="time">[${startTime} - ${endTime}]</span>
                <span class="text">${segment.text}</span>
            `;
            
            timestampsContainer.appendChild(timestampDiv);
        });
        
        // Add the timestamp elements to the message
        messageDiv.appendChild(toggleButton);
        messageDiv.appendChild(timestampsContainer);
    } else {
        // No timestamp data available - add a simple timestamp for the entire message
        const timestamp = new Date().toLocaleTimeString();
        const timeLabel = document.createElement('div');
        timeLabel.className = 'simple-timestamp';
        timeLabel.textContent = timestamp;
        messageDiv.appendChild(timeLabel);
    }
    
    return messageDiv;
}

// Helper function to format time in seconds to MM:SS.ms format
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
}

// Handle context update from server
function handleContextUpdate(data) {
    if (data.status === 'cleared') {
        elements.conversation.innerHTML = '';
        addSystemMessage('Conversation context cleared.');
    }
}

// Handle streaming status updates from server
function handleStreamingStatus(data) {
    if (data.status === 'active') {
        console.log('Server acknowledged streaming is active');
    } else if (data.status === 'inactive') {
        console.log('Server acknowledged streaming is inactive');
    }
}

// Handle processing status updates
function handleProcessingStatus(data) {
    switch (data.status) {
        case 'transcribing':
            addSystemMessage('Transcribing your message...');
            break;
        case 'generating':
            addSystemMessage('Generating response...');
            break;
        case 'synthesizing':
            addSystemMessage('Synthesizing voice...');
            break;
    }
}

// Handle the start of an audio streaming response
function handleAudioResponseStart(data) {
    console.log(`Expecting ${data.total_chunks} audio chunks`);
    
    // Reset streaming state
    streamingAudio.chunks = [];
    streamingAudio.totalChunks = data.total_chunks;
    streamingAudio.receivedChunks = 0;
    streamingAudio.text = data.text;
    streamingAudio.complete = false;
}

// Handle an incoming audio chunk
function handleAudioResponseChunk(data) {
    // Create or update audio element for playback
    const audioElement = document.createElement('audio');
    if (elements.autoPlayResponses.checked) {
        audioElement.autoplay = true;
    }
    audioElement.controls = true;
    audioElement.className = 'audio-player';
    audioElement.src = data.chunk;
    
    // Store the chunk
    streamingAudio.chunks[data.chunk_index] = data.chunk;
    streamingAudio.receivedChunks++;
    
    // Store audio element reference for later use
    streamingAudio.audioElement = audioElement;
    
    // Add to the conversation
    const messages = elements.conversation.querySelectorAll('.message.ai');
    if (messages.length > 0) {
        const lastAiMessage = messages[messages.length - 1];
        streamingAudio.messageElement = lastAiMessage;
        
        // Replace existing audio player if there is one
        const existingPlayer = lastAiMessage.querySelector('.audio-player');
        if (existingPlayer) {
            lastAiMessage.replaceChild(audioElement, existingPlayer);
        } else {
            lastAiMessage.appendChild(audioElement);
        }
    } else {
        // Create a new message for the AI response
        const aiMessage = document.createElement('div');
        aiMessage.className = 'message ai';
        streamingAudio.messageElement = aiMessage;
        
        if (streamingAudio.text) {
            const textElement = document.createElement('p');
            textElement.textContent = streamingAudio.text;
            aiMessage.appendChild(textElement);
        }
        
        aiMessage.appendChild(audioElement);
        elements.conversation.appendChild(aiMessage);
    }
    
    // Auto-scroll
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
    
    // If this is the last chunk or we've received all expected chunks
    if (data.is_last || streamingAudio.receivedChunks >= streamingAudio.totalChunks) {
        streamingAudio.complete = true;
        
        // Reset stream button if we're still streaming
        if (state.isStreaming) {
            elements.streamButton.classList.remove('processing');
            elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Listening...';
        }
    }
}

// Handle completion of audio streaming
function handleAudioResponseComplete(data) {
    console.log('Audio response complete:', data);
    streamingAudio.complete = true;
    
    // Make sure we finalize the audio even if some chunks were missed
    finalizeStreamingAudio();
    
    // Update UI to normal state
    if (state.isStreaming) {
        elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Listening...';
        elements.streamButton.classList.add('recording');
        elements.streamButton.classList.remove('processing');
    }
}

// Finalize streaming audio by combining chunks and updating the UI
function finalizeStreamingAudio() {
    if (!streamingAudio.messageElement || streamingAudio.chunks.length === 0) {
        return;
    }
    
    try {
        // For more sophisticated audio streaming, you would need to properly concatenate
        // the WAV files, but for now we'll use the last chunk as the complete audio
        // since it should contain the entire response due to how the server is implementing it
        const lastChunkIndex = streamingAudio.chunks.length - 1;
        const audioData = streamingAudio.chunks[lastChunkIndex] || streamingAudio.chunks[0];
        
        // Update the audio element with the complete audio
        if (streamingAudio.audioElement) {
            streamingAudio.audioElement.src = audioData;
            
            // Auto-play if enabled and not already playing
            if (elements.autoPlayResponses && elements.autoPlayResponses.checked && 
                streamingAudio.audioElement.paused) {
                streamingAudio.audioElement.play()
                    .catch(err => {
                        console.warn('Auto-play failed:', err);
                        addSystemMessage('Auto-play failed. Please click play to hear the response.');
                    });
            }
        }
        
        // Remove loading indicator and processing class
        if (streamingAudio.messageElement) {
            const loadingElement = streamingAudio.messageElement.querySelector('.loading-indicator');
            if (loadingElement) {
                streamingAudio.messageElement.removeChild(loadingElement);
            }
            streamingAudio.messageElement.classList.remove('processing');
        }
        
        console.log('Audio response finalized and ready for playback');
    } catch (e) {
        console.error('Error finalizing streaming audio:', e);
    }
    
    // Reset streaming audio state
    streamingAudio.chunks = [];
    streamingAudio.totalChunks = 0;
    streamingAudio.receivedChunks = 0;
    streamingAudio.messageElement = null;
    streamingAudio.audioElement = null;
}

// Handle model status updates
function handleModelStatusUpdate(data) {
    const { model, status, message } = data;
    
    if (status === 'loaded') {
        console.log(`Model ${model} loaded successfully`);
        addSystemMessage(`${model.toUpperCase()} model loaded successfully`);
        
        // Update UI to show model is ready
        const modelStatusElement = document.getElementById(`${model}Status`);
        if (modelStatusElement) {
            modelStatusElement.classList.remove('loading');
            modelStatusElement.classList.add('loaded');
            modelStatusElement.title = 'Model loaded successfully';
        }
        
        // Check if the required models are loaded to enable conversation
        checkAllModelsLoaded();
    } else if (status === 'error') {
        console.error(`Error loading ${model} model: ${message}`);
        addSystemMessage(`Error loading ${model.toUpperCase()} model: ${message}`);
        
        // Update UI to show model loading failed
        const modelStatusElement = document.getElementById(`${model}Status`);
        if (modelStatusElement) {
            modelStatusElement.classList.remove('loading');
            modelStatusElement.classList.add('error');
            modelStatusElement.title = `Error: ${message}`;
        }
    }
}

// Check if all required models are loaded and enable UI accordingly
function checkAllModelsLoaded() {
    // When all models are loaded, enable the stream button if it was disabled
    const allLoaded = 
        document.getElementById('csmStatus')?.classList.contains('loaded') &&
        document.getElementById('asrStatus')?.classList.contains('loaded') &&
        document.getElementById('llmStatus')?.classList.contains('loaded');
    
    if (allLoaded) {
        elements.streamButton.disabled = false;
        addSystemMessage('All models loaded. Ready for conversation!');
    }
}

// Add CSS styles for new UI elements
document.addEventListener('DOMContentLoaded', function() {
    // Add styles for processing state and timestamps
    const style = document.createElement('style');
    style.textContent = `
        .message.processing {
            opacity: 0.8;
        }
        
        .loading-indicator {
            display: flex;
            align-items: center;
            margin-top: 8px;
            font-size: 0.9em;
            color: #666;
        }
        
        .loading-spinner {
            width: 16px;
            height: 16px;
            border: 2px solid #ddd;
            border-top: 2px solid var(--primary-color);
            border-radius: 50%;
            margin-right: 8px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Timestamp styles */
        .timestamp-toggle {
            font-size: 0.75em;
            padding: 4px 8px;
            margin-top: 8px;
            background-color: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .timestamp-toggle:hover {
            background-color: #e0e0e0;
        }
        
        .timestamps-container {
            margin-top: 8px;
            padding: 8px;
            background-color: #f9f9f9;
            border-radius: 4px;
            font-size: 0.85em;
        }
        
        .timestamp {
            margin-bottom: 4px;
            padding: 2px 0;
        }
        
        .timestamp .time {
            color: #666;
            font-family: monospace;
            margin-right: 8px;
        }
        
        .timestamp .text {
            color: #333;
        }
    `;
    document.head.appendChild(style);
});

// Initialize the application when DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeApp);

