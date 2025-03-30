/**
 * Sesame AI Voice Chat Client
 * 
 * A web client that connects to a Sesame AI voice chat server and enables 
 * real-time voice conversation with an AI assistant.
 */

// Configuration constants
const SERVER_URL = window.location.hostname === 'localhost' ? 
    'http://localhost:5000' : window.location.origin;
const ENERGY_WINDOW_SIZE = 15;
const CLIENT_SILENCE_DURATION_MS = 750;

// DOM elements
const elements = {
    conversation: null,
    streamButton: null,
    clearButton: null,
    thresholdSlider: null,
    thresholdValue: null,
    visualizerCanvas: null,
    visualizerLabel: null,
    volumeLevel: null,
    statusDot: null,
    statusText: null,
    speakerSelection: null,
    autoPlayResponses: null,
    showVisualizer: null
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
    currentSpeaker: 0
};

// Visualizer variables
let canvasContext = null;
let visualizerBufferLength = 0;
let visualizerDataArray = null;

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
    addSystemMessage('Welcome to Sesame AI Voice Chat! Click "Start Conversation" to begin.');
}

// Initialize UI elements
function initializeUIElements() {
    // Main UI containers
    const chatContainer = document.querySelector('.chat-container');
    const controlPanel = document.querySelector('.control-panel');

    // Create conversation section
    chatContainer.innerHTML = `
        <div class="chat-header">
            <h2>Conversation</h2>
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span class="status-text">Disconnected</span>
            </div>
        </div>
        <div class="conversation"></div>
    `;

    // Create control panel
    controlPanel.innerHTML = `
        <div class="visualizer-section">
            <div class="visualizer-container">
                <canvas id="audioVisualizer"></canvas>
                <div class="visualizer-label">Speak to see audio visualization</div>
            </div>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <div class="control-label">Voice Controls</div>
                
                <div class="volume-indicator">
                    <div class="volume-level" style="width:0%"></div>
                </div>
                
                <div class="slider-container">
                    <div class="slider-label">
                        <span>Silence Threshold</span>
                        <span id="thresholdValue">0.01</span>
                    </div>
                    <input type="range" id="thresholdSlider" min="0.001" max="0.05" step="0.001" value="0.01">
                </div>
                
                <select id="speakerSelection">
                    <option value="0">Speaker 1 (You)</option>
                    <option value="1">Speaker 2 (Alternative)</option>
                </select>
                
                <div class="button-row">
                    <button id="streamButton"><i class="fas fa-microphone"></i> Start Conversation</button>
                    <button id="clearButton"><i class="fas fa-trash"></i> Clear</button>
                </div>
            </div>
            
            <div class="control-group settings-panel">
                <div class="control-label">Settings</div>
                
                <div class="settings-toggles">
                    <div class="toggle-switch">
                        <input type="checkbox" id="autoPlayResponses" checked>
                        <label for="autoPlayResponses">Auto-play AI responses</label>
                    </div>
                    
                    <div class="toggle-switch">
                        <input type="checkbox" id="showVisualizer" checked>
                        <label for="showVisualizer">Show audio visualizer</label>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Store references to UI elements
    elements.conversation = document.querySelector('.conversation');
    elements.streamButton = document.getElementById('streamButton');
    elements.clearButton = document.getElementById('clearButton');
    elements.thresholdSlider = document.getElementById('thresholdSlider');
    elements.thresholdValue = document.getElementById('thresholdValue');
    elements.visualizerCanvas = document.getElementById('audioVisualizer');
    elements.visualizerLabel = document.querySelector('.visualizer-label');
    elements.volumeLevel = document.querySelector('.volume-level');
    elements.statusDot = document.querySelector('.status-dot');
    elements.statusText = document.querySelector('.status-text');
    elements.speakerSelection = document.getElementById('speakerSelection');
    elements.autoPlayResponses = document.getElementById('autoPlayResponses');
    elements.showVisualizer = document.getElementById('showVisualizer');
}

// Setup Socket.IO connection
function setupSocketConnection() {
    state.socket = io(SERVER_URL);
    
    // Connection events
    state.socket.on('connect', () => {
        console.log('Connected to server');
        updateConnectionStatus(true);
    });
    
    state.socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
        
        // Stop streaming if active
        if (state.isStreaming) {
            stopStreaming(false);
        }
    });
    
    state.socket.on('error', (data) => {
        console.error('Socket error:', data.message);
        addSystemMessage(`Error: ${data.message}`);
    });
    
    // Register message handlers
    state.socket.on('audio_response', handleAudioResponse);
    state.socket.on('transcription', handleTranscription);
    state.socket.on('context_updated', handleContextUpdate);
    state.socket.on('streaming_status', handleStreamingStatus);
}

// Setup event listeners
function setupEventListeners() {
    // Stream button
    elements.streamButton.addEventListener('click', toggleStreaming);
    
    // Clear button
    elements.clearButton.addEventListener('click', clearConversation);
    
    // Threshold slider
    elements.thresholdSlider.addEventListener('input', updateThreshold);
    
    // Speaker selection
    elements.speakerSelection.addEventListener('change', () => {
        state.currentSpeaker = parseInt(elements.speakerSelection.value, 10);
    });
    
    // Visualizer toggle
    elements.showVisualizer.addEventListener('change', toggleVisualizerVisibility);
}

// Setup audio visualizer
function setupVisualizer() {
    if (!elements.visualizerCanvas) return;
    
    canvasContext = elements.visualizerCanvas.getContext('2d');
    
    // Set canvas dimensions
    elements.visualizerCanvas.width = elements.visualizerCanvas.offsetWidth;
    elements.visualizerCanvas.height = elements.visualizerCanvas.offsetHeight;
    
    // Initialize the visualizer
    drawVisualizer();
}

// Update connection status UI
function updateConnectionStatus(isConnected) {
    elements.statusDot.classList.toggle('active', isConnected);
    elements.statusText.textContent = isConnected ? 'Connected' : 'Disconnected';
}

// Toggle streaming state
function toggleStreaming() {
    if (state.isStreaming) {
        stopStreaming(true);
    } else {
        startStreaming();
    }
}

// Start streaming audio to the server
function startStreaming() {
    if (state.isStreaming) return;
    
    // Request microphone access
    navigator.mediaDevices.getUserMedia({ audio: true, video: false })
        .then(stream => {
            // Show processing state while setting up
            elements.streamButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Initializing...';
            
            // Create audio context
            state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create microphone source
            state.microphone = state.audioContext.createMediaStreamSource(stream);
            
            // Create analyser for visualizer
            state.analyser = state.audioContext.createAnalyser();
            state.analyser.fftSize = 256;
            visualizerBufferLength = state.analyser.frequencyBinCount;
            visualizerDataArray = new Uint8Array(visualizerBufferLength);
            
            // Connect microphone to analyser
            state.microphone.connect(state.analyser);
            
            // Create script processor for audio processing
            const bufferSize = 4096;
            state.streamProcessor = state.audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            // Set up audio processing callback
            state.streamProcessor.onaudioprocess = handleAudioProcess;
            
            // Connect the processors
            state.analyser.connect(state.streamProcessor);
            state.streamProcessor.connect(state.audioContext.destination);
            
            // Update UI
            state.isStreaming = true;
            elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Listening...';
            elements.streamButton.classList.add('recording');
            
            // Initialize energy window
            state.energyWindow = [];
            
            // Start volume meter updates
            state.volumeUpdateInterval = setInterval(updateVolumeMeter, 100);
            
            // Start visualizer if enabled
            if (elements.showVisualizer.checked && !state.visualizerAnimationFrame) {
                drawVisualizer();
            }
            
            // Show starting message
            addSystemMessage('Listening... Speak clearly into your microphone.');
            
            // Notify the server that we're starting
            state.socket.emit('stream_audio', {
                audio: '',
                speaker: state.currentSpeaker
            });
        })
        .catch(err => {
            console.error('Error accessing microphone:', err);
            addSystemMessage(`Error: ${err.message}. Please make sure your microphone is connected and you've granted permission.`);
            elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Start Conversation';
        });
}

// Stop streaming audio
function stopStreaming(notifyServer = true) {
    if (!state.isStreaming) return;
    
    // Update UI first
    elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Start Conversation';
    elements.streamButton.classList.remove('recording');
    elements.streamButton.classList.remove('processing');
    
    // Stop volume meter updates
    if (state.volumeUpdateInterval) {
        clearInterval(state.volumeUpdateInterval);
        state.volumeUpdateInterval = null;
    }
    
    // Stop all audio processing
    if (state.streamProcessor) {
        state.streamProcessor.disconnect();
        state.streamProcessor = null;
    }
    
    if (state.analyser) {
        state.analyser.disconnect();
    }
    
    if (state.microphone) {
        state.microphone.disconnect();
    }
    
    // Close audio context
    if (state.audioContext && state.audioContext.state !== 'closed') {
        state.audioContext.close().catch(err => console.warn('Error closing audio context:', err));
    }
    
    // Cleanup animation frames
    if (state.visualizerAnimationFrame) {
        cancelAnimationFrame(state.visualizerAnimationFrame);
        state.visualizerAnimationFrame = null;
    }
    
    // Reset state
    state.isStreaming = false;
    state.isSpeaking = false;
    
    // Notify the server
    if (notifyServer && state.socket && state.socket.connected) {
        state.socket.emit('stop_streaming', {
            speaker: state.currentSpeaker
        });
    }
    
    // Show message
    addSystemMessage('Conversation paused. Click "Start Conversation" to resume.');
}

// Handle audio processing
function handleAudioProcess(event) {
    const inputData = event.inputBuffer.getChannelData(0);
    
    // Calculate audio energy (volume level)
    const energy = calculateAudioEnergy(inputData);
    
    // Update energy window for averaging
    updateEnergyWindow(energy);
    
    // Calculate average energy
    const avgEnergy = calculateAverageEnergy();
    
    // Determine if audio is silent
    const isSilent = avgEnergy < state.silenceThreshold;
    
    // Handle speech state based on silence
    handleSpeechState(isSilent);
    
    // Only send audio chunk if we detect speech
    if (!isSilent) {
        // Create a resampled version at 24kHz for the server
        // Most WebRTC audio is 48kHz, but we want 24kHz for the model
        const resampledData = downsampleBuffer(inputData, state.audioContext.sampleRate, 24000);
        
        // Send the audio chunk to the server
        sendAudioChunk(resampledData, state.currentSpeaker);
    }
}

// Cleanup audio resources when done
function cleanupAudioResources() {
    // Stop all audio processing
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
    
    // Close audio context
    if (state.audioContext && state.audioContext.state !== 'closed') {
        state.audioContext.close().catch(err => console.warn('Error closing audio context:', err));
    }
    
    // Cancel all timers and animation frames
    if (state.volumeUpdateInterval) {
        clearInterval(state.volumeUpdateInterval);
        state.volumeUpdateInterval = null;
    }
    
    if (state.visualizerAnimationFrame) {
        cancelAnimationFrame(state.visualizerAnimationFrame);
        state.visualizerAnimationFrame = null;
    }
    
    if (state.silenceTimer) {
        clearTimeout(state.silenceTimer);
        state.silenceTimer = null;
    }
}

// Clear conversation history
function clearConversation() {
    if (elements.conversation) {
        elements.conversation.innerHTML = '';
        addSystemMessage('Conversation cleared.');
        
        // Notify server to clear context
        if (state.socket && state.socket.connected) {
            state.socket.emit('clear_context');
        }
    }
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
    
    const sum = state.energyWindow.reduce((a, b) => a + b, 0);
    return sum / state.energyWindow.length;
}

// Update the threshold from the slider
function updateThreshold() {
    state.silenceThreshold = parseFloat(elements.thresholdSlider.value);
    elements.thresholdValue.textContent = state.silenceThreshold.toFixed(3);
}

// Update the volume meter display
function updateVolumeMeter() {
    if (!state.isStreaming || !state.energyWindow.length) return;
    
    const avgEnergy = calculateAverageEnergy();
    
    // Scale energy to percentage (0-100)
    // Typically, energy values will be very small (e.g., 0.001 to 0.1)
    // So we multiply by a factor to make it more visible
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
    if (state.isSpeaking && isSilent) {
        // Transition from speaking to silence
        if (!state.silenceTimer) {
            state.silenceTimer = setTimeout(() => {
                // Only consider it a real silence after a certain duration
                // This prevents detecting brief pauses as the end of speech
                state.isSpeaking = false;
                state.silenceTimer = null;
            }, CLIENT_SILENCE_DURATION_MS);
        }
    } else if (state.silenceTimer && !isSilent) {
        // User started speaking again, cancel the silence timer
        clearTimeout(state.silenceTimer);
        state.silenceTimer = null;
    }
    
    // Update speaking state for non-silent audio
    if (!isSilent) {
        state.isSpeaking = true;
    }
}

// Send audio chunk to server
function sendAudioChunk(audioData, speaker) {
    if (!state.socket || !state.socket.connected) {
        console.warn('Socket not connected');
        return;
    }
    
    const wavData = createWavBlob(audioData, 24000);
    const reader = new FileReader();
    
    reader.onloadend = function() {
        const base64data = reader.result;
        
        // Send the audio chunk to the server
        state.socket.emit('stream_audio', {
            audio: base64data,
            speaker: speaker
        });
    };
    
    reader.readAsDataURL(wavData);
}

// Draw audio visualizer
function drawVisualizer() {
    if (!canvasContext) {
        return;
    }
    
    state.visualizerAnimationFrame = requestAnimationFrame(drawVisualizer);
    
    // Skip drawing if visualizer is hidden
    if (!elements.showVisualizer.checked) {
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
        // This makes low values more visible while still maintaining full range
        const logFactor = 20;
        const scaledValue = Math.log(1 + (value / 255) * logFactor) / Math.log(1 + logFactor);
        const barHeight = scaledValue * height;
        
        // Position bars
        const x = i * (barWidth + 1);
        const y = height - barHeight;
        
        // Create color gradient based on frequency and amplitude
        const hue = i / barCount * 360; // Full color spectrum
        const saturation = 80 + (value / 255 * 20); // Higher values more saturated
        const lightness = 40 + (value / 255 * 20); // Dynamic brightness based on amplitude
        
        // Draw main bar
        canvasContext.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        canvasContext.fillRect(x, y, barWidth, barHeight);
        
        // Add reflection effect
        if (barHeight > 5) {
            const gradient = canvasContext.createLinearGradient(
                x, y, 
                x, y + barHeight * 0.5
            );
            gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness + 20}%, 0.4)`);
            gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness}%, 0)`);
            canvasContext.fillStyle = gradient;
            canvasContext.fillRect(x, y, barWidth, barHeight * 0.5);
            
            // Add highlight on top of the bar for better 3D effect
            canvasContext.fillStyle = `hsla(${hue}, ${saturation - 20}%, ${lightness + 30}%, 0.7)`;
            canvasContext.fillRect(x, y, barWidth, 2);
        }
    }
    
    // Show/hide the label
    elements.visualizerLabel.style.opacity = (state.isStreaming) ? '0' : '0.7';
}

// Toggle visualizer visibility
function toggleVisualizerVisibility() {
    const isVisible = elements.showVisualizer.checked;
    elements.visualizerCanvas.style.opacity = isVisible ? '1' : '0';
    
    if (isVisible && state.isStreaming && !state.visualizerAnimationFrame) {
        drawVisualizer();
    }
}

// Handle audio response from server
function handleAudioResponse(data) {
    console.log('Received audio response');
    
    // Create message container
    const messageElement = document.createElement('div');
    messageElement.className = 'message ai';
    
    // Add text content if available
    if (data.text) {
        const textElement = document.createElement('p');
        textElement.textContent = data.text;
        messageElement.appendChild(textElement);
    }
    
    // Create and configure audio element
    const audioElement = document.createElement('audio');
    audioElement.controls = true;
    audioElement.className = 'audio-player';
    
    // Set audio source
    const audioSource = document.createElement('source');
    audioSource.src = data.audio;
    audioSource.type = 'audio/wav';
    
    // Add fallback text
    audioElement.textContent = 'Your browser does not support the audio element.';
    
    // Assemble audio element
    audioElement.appendChild(audioSource);
    messageElement.appendChild(audioElement);
    
    // Add timestamp
    const timeElement = document.createElement('span');
    timeElement.className = 'message-time';
    timeElement.textContent = new Date().toLocaleTimeString();
    messageElement.appendChild(timeElement);
    
    // Add to conversation
    elements.conversation.appendChild(messageElement);
    
    // Auto-scroll to bottom
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
    
    // Auto-play if enabled
    if (elements.autoPlayResponses.checked) {
        audioElement.play()
            .catch(err => {
                console.warn('Auto-play failed:', err);
                addSystemMessage('Auto-play failed. Please click play to hear the response.');
            });
    }
    
    // Re-enable stream button after processing is complete
    if (state.isStreaming) {
        elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Listening...';
        elements.streamButton.classList.add('recording');
        elements.streamButton.classList.remove('processing');
    }
}

// Handle transcription response from server
function handleTranscription(data) {
    console.log('Received transcription:', data.text);
    
    // Create message element
    const messageElement = document.createElement('div');
    messageElement.className = 'message user';
    
    // Add text content
    const textElement = document.createElement('p');
    textElement.textContent = data.text;
    messageElement.appendChild(textElement);
    
    // Add timestamp
    const timeElement = document.createElement('span');
    timeElement.className = 'message-time';
    timeElement.textContent = new Date().toLocaleTimeString();
    messageElement.appendChild(timeElement);
    
    // Add to conversation
    elements.conversation.appendChild(messageElement);
    
    // Auto-scroll to bottom
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

// Handle context update from server
function handleContextUpdate(data) {
    console.log('Context updated:', data.message);
}

// Handle streaming status updates from server
function handleStreamingStatus(data) {
    console.log('Streaming status:', data.status);
    
    if (data.status === 'stopped') {
        // Reset UI if needed
        if (state.isStreaming) {
            stopStreaming(false); // Don't send to server since this came from server
        }
    }
}

// Add a system message to the conversation
function addSystemMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message system';
    messageElement.textContent = message;
    elements.conversation.appendChild(messageElement);
    
    // Auto-scroll to bottom
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

// Create WAV blob from audio data
function createWavBlob(audioData, sampleRate) {
    // Function to convert Float32Array to Int16Array for WAV format
    function floatTo16BitPCM(output, offset, input) {
        for (let i = 0; i < input.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, input[i]));
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
    }
    
    // Create WAV header
    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
    
    // Create WAV file with header
    function encodeWAV(samples) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        
        // RIFF chunk descriptor
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, 'WAVE');
        
        // fmt sub-chunk
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM format
        view.setUint16(22, 1, true); // Mono channel
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true); // Byte rate
        view.setUint16(32, 2, true); // Block align
        view.setUint16(34, 16, true); // Bits per sample
        
        // data sub-chunk
        writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);
        floatTo16BitPCM(view, 44, samples);
        
        return buffer;
    }
    
    // Convert audio data to TypedArray if it's a regular Array
    const samples = Array.isArray(audioData) ? new Float32Array(audioData) : audioData;
    
    // Create WAV blob
    const wavBuffer = encodeWAV(samples);
    return new Blob([wavBuffer], { type: 'audio/wav' });
}

// Downsample audio buffer to target sample rate
function downsampleBuffer(buffer, originalSampleRate, targetSampleRate) {
    if (originalSampleRate === targetSampleRate) {
        return buffer;
    }
    
    const ratio = originalSampleRate / targetSampleRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
        const pos = Math.round(i * ratio);
        result[i] = buffer[pos];
    }
    
    return result;
}

// Initialize the application when DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeApp);

