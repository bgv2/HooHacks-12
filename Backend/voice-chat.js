/**
 * Sesame AI Voice Chat Application
 * 
 * This script handles the audio streaming, visualization, 
 * and Socket.IO communication for the voice chat application.
 */

// Application state
const state = {
    socket: null,
    audioContext: null,
    streamProcessor: null,
    analyser: null,
    microphone: null,
    isStreaming: false,
    isSpeaking: false,
    silenceTimer: null,
    energyWindow: [],
    currentSpeaker: 0,
    silenceThreshold: 0.01,
    visualizerAnimationFrame: null,
    volumeUpdateInterval: null,
    connectionAttempts: 0
};

// Constants
const ENERGY_WINDOW_SIZE = 10;
const CLIENT_SILENCE_DURATION_MS = 1000; // 1 second of silence before processing
const MAX_CONNECTION_ATTEMPTS = 5;
const RECONNECTION_DELAY_MS = 2000;

// DOM elements
const elements = {
    conversation: document.getElementById('conversation'),
    speakerSelect: document.getElementById('speakerSelect'),
    streamButton: document.getElementById('streamButton'),
    clearButton: document.getElementById('clearButton'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    visualizerCanvas: document.getElementById('audioVisualizer'),
    visualizerLabel: document.getElementById('visualizerLabel'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    volumeLevel: document.getElementById('volumeLevel'),
    autoPlayResponses: document.getElementById('autoPlayResponses'),
    showVisualizer: document.getElementById('showVisualizer')
};

// Visualization variables
let canvasContext;
let visualizerBufferLength;
let visualizerDataArray;

// Initialize the application
function initializeApp() {
    // Set up event listeners
    elements.streamButton.addEventListener('click', toggleStreaming);
    elements.clearButton.addEventListener('click', clearConversation);
    elements.thresholdSlider.addEventListener('input', updateThreshold);
    elements.speakerSelect.addEventListener('change', () => {
        state.currentSpeaker = parseInt(elements.speakerSelect.value);
    });
    elements.showVisualizer.addEventListener('change', toggleVisualizerVisibility);

    // Initialize audio context
    setupAudioContext();
    
    // Set up visualization
    setupVisualizer();
    
    // Connect to Socket.IO server
    connectToServer();

    // Add welcome message
    addSystemMessage('Welcome to Sesame AI Voice Chat! Click "Start Conversation" to begin speaking.');
}

// Connect to Socket.IO server
function connectToServer() {
    try {
        // Use the server URL with or without a specific port
        const serverUrl = window.location.origin;
        
        updateStatus('Connecting...', 'connecting');
        console.log(`Connecting to Socket.IO server at ${serverUrl}`);
        
        state.socket = io(serverUrl, {
            reconnectionDelay: RECONNECTION_DELAY_MS,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: MAX_CONNECTION_ATTEMPTS
        });
        
        setupSocketListeners();
    } catch (error) {
        console.error('Error connecting to server:', error);
        updateStatus('Connection failed. Retrying...', 'error');
        
        // Try to reconnect
        if (state.connectionAttempts < MAX_CONNECTION_ATTEMPTS) {
            state.connectionAttempts++;
            setTimeout(connectToServer, RECONNECTION_DELAY_MS);
        } else {
            updateStatus('Could not connect to server', 'error');
            addSystemMessage('Failed to connect to the server. Please check your connection and refresh the page.');
        }
    }
}

// Set up Socket.IO event listeners
function setupSocketListeners() {
    if (!state.socket) return;
    
    state.socket.on('connect', () => {
        console.log('Connected to Socket.IO server');
        updateStatus('Connected', 'connected');
        state.connectionAttempts = 0;
        elements.streamButton.disabled = false;
        addSystemMessage('Connected to server');
    });
    
    state.socket.on('disconnect', () => {
        console.log('Disconnected from Socket.IO server');
        updateStatus('Disconnected', 'disconnected');
        
        // Stop streaming if active
        if (state.isStreaming) {
            stopStreaming(false); // false = don't send to server
        }
        
        elements.streamButton.disabled = true;
        addSystemMessage('Disconnected from server. Trying to reconnect...');
    });
    
    state.socket.on('status', (data) => {
        console.log('Status:', data);
        addSystemMessage(data.message);
    });
    
    state.socket.on('error', (data) => {
        console.error('Server error:', data);
        addSystemMessage(`Error: ${data.message}`);
    });
    
    state.socket.on('audio_response', handleAudioResponse);
    state.socket.on('transcription', handleTranscription);
    state.socket.on('context_updated', handleContextUpdate);
    state.socket.on('streaming_status', handleStreamingStatus);
    
    state.socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        updateStatus('Connection Error', 'error');
    });
}

// Update the connection status in the UI
function updateStatus(message, status) {
    elements.statusText.textContent = message;
    elements.statusDot.className = 'status-dot';
    
    if (status === 'connected') {
        elements.statusDot.classList.add('active');
    } else if (status === 'connecting') {
        elements.statusDot.style.backgroundColor = '#FFA500';
    } else if (status === 'error') {
        elements.statusDot.style.backgroundColor = '#F44336';
    }
}

// Set up audio context
function setupAudioContext() {
    try {
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        console.log('Audio context initialized');
    } catch (err) {
        console.error('Error setting up audio context:', err);
        addSystemMessage(`Audio context error: ${err.message}`);
        elements.streamButton.disabled = true;
    }
}

// Set up audio visualizer
function setupVisualizer() {
    canvasContext = elements.visualizerCanvas.getContext('2d');
    
    // Set canvas size to match container
    function resizeCanvas() {
        const container = elements.visualizerCanvas.parentElement;
        elements.visualizerCanvas.width = container.clientWidth;
        elements.visualizerCanvas.height = container.clientHeight;
    }
    
    // Call initially and on window resize
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Create placeholder data array
    visualizerBufferLength = 128;
    visualizerDataArray = new Uint8Array(visualizerBufferLength);
}

// Toggle stream on/off
function toggleStreaming() {
    if (state.isStreaming) {
        stopStreaming(true); // true = send to server
    } else {
        startStreaming();
    }
}

// Start streaming audio to the server
async function startStreaming() {
    if (!state.socket || !state.socket.connected) {
        addSystemMessage('Cannot start conversation: Not connected to server');
        return;
    }
    
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Update state
        state.isStreaming = true;
        state.isSpeaking = false;
        state.energyWindow = [];
        state.currentSpeaker = parseInt(elements.speakerSelect.value);
        
        // Update UI
        elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Listening...';
        elements.streamButton.classList.add('recording');
        elements.visualizerLabel.style.opacity = '0';
        
        // Set up audio processing
        setupAudioProcessing(stream);
        
        // Start volume meter updates
        state.volumeUpdateInterval = setInterval(updateVolumeMeter, 100);
        
        addSystemMessage('Listening - speak naturally and pause when finished');
        
    } catch (err) {
        console.error('Error starting audio stream:', err);
        addSystemMessage(`Microphone error: ${err.message}`);
        cleanupAudioResources();
    }
}

// Set up audio processing pipeline
function setupAudioProcessing(stream) {
    // Store microphone stream for later cleanup
    state.microphone = stream;
    
    // Create source from microphone
    const source = state.audioContext.createMediaStreamSource(stream);
    
    // Setup analyzer for visualization
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 256;
    state.analyser.smoothingTimeConstant = 0.8;
    state.analyser.minDecibels = -90;
    state.analyser.maxDecibels = -10;
    
    visualizerBufferLength = state.analyser.frequencyBinCount;
    visualizerDataArray = new Uint8Array(visualizerBufferLength);
    
    // Connect source to analyzer
    source.connect(state.analyser);
    
    // Start visualization
    if (state.visualizerAnimationFrame) {
        cancelAnimationFrame(state.visualizerAnimationFrame);
    }
    drawVisualizer();
    
    // Setup audio processor
    state.streamProcessor = state.audioContext.createScriptProcessor(4096, 1, 1);
    
    // Connect audio nodes
    source.connect(state.streamProcessor);
    state.streamProcessor.connect(state.audioContext.destination);
    
    // Process audio
    state.streamProcessor.onaudioprocess = handleAudioProcess;
}

// Handle each frame of audio data
function handleAudioProcess(e) {
    const audioData = e.inputBuffer.getChannelData(0);
    
    // Calculate energy (volume) for silence detection
    const energy = calculateAudioEnergy(audioData);
    updateEnergyWindow(energy);
    
    // Check if currently silent
    const avgEnergy = calculateAverageEnergy();
    const isSilent = avgEnergy < state.silenceThreshold;
    
    // Handle silence/speech transitions
    handleSpeechState(isSilent);
    
    // Process and send audio
    const downsampled = downsampleBuffer(audioData, state.audioContext.sampleRate, 24000);
    sendAudioChunk(downsampled, state.currentSpeaker);
}

// Stop streaming audio
function stopStreaming(sendToServer = true) {
    // Cleanup audio resources
    cleanupAudioResources();
    
    // Reset state
    state.isStreaming = false;
    state.isSpeaking = false;
    state.energyWindow = [];
    
    // Update UI
    elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Start Conversation';
    elements.streamButton.classList.remove('recording', 'processing');
    elements.streamButton.style.backgroundColor = '';
    elements.volumeLevel.style.width = '100%';
    
    // Clear volume meter updates
    if (state.volumeUpdateInterval) {
        clearInterval(state.volumeUpdateInterval);
        state.volumeUpdateInterval = null;
    }
    
    addSystemMessage('Conversation paused');
    
    // Notify server
    if (sendToServer && state.socket && state.socket.connected) {
        state.socket.emit('stop_streaming', {
            speaker: state.currentSpeaker
        });
    }
}

// Clean up audio processing resources
function cleanupAudioResources() {
    // Stop microphone stream
    if (state.microphone) {
        state.microphone.getTracks().forEach(track => track.stop());
        state.microphone = null;
    }
    
    // Disconnect audio processor
    if (state.streamProcessor) {
        state.streamProcessor.disconnect();
        state.streamProcessor.onaudioprocess = null;
        state.streamProcessor = null;
    }
    
    // Disconnect analyzer
    if (state.analyser) {
        state.analyser.disconnect();
        state.analyser = null;
    }
    
    // Cancel visualizer animation
    if (state.visualizerAnimationFrame) {
        cancelAnimationFrame(state.visualizerAnimationFrame);
        state.visualizerAnimationFrame = null;
    }
    
    // Cancel silence timer
    if (state.silenceTimer) {
        clearTimeout(state.silenceTimer);
        state.silenceTimer = null;
    }
    
    // Reset visualizer display
    if (canvasContext) {
        canvasContext.clearRect(0, 0, elements.visualizerCanvas.width, elements.visualizerCanvas.height);
        elements.visualizerLabel.style.opacity = '0.7';
    }
}

// Clear conversation history
function clearConversation() {
    // Clear UI
    elements.conversation.innerHTML = '';
    addSystemMessage('Conversation cleared');
    
    // Notify server
    if (state.socket && state.socket.connected) {
        state.socket.emit('clear_context');
    }
}

// Calculate audio energy (volume)
function calculateAudioEnergy(buffer) {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
        sum += Math.abs(buffer[i]);
    }
    return sum / buffer.length;
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
    return state.energyWindow.reduce((sum, val) => sum + val, 0) / state.energyWindow.length;
}

// Update the threshold from the slider
function updateThreshold() {
    state.silenceThreshold = parseFloat(elements.thresholdSlider.value);
    elements.thresholdValue.textContent = state.silenceThreshold.toFixed(3);
}

// Update the volume meter display
function updateVolumeMeter() {
    if (!state.isStreaming || !state.analyser) return;
    
    // Get current volume level
    const dataArray = new Uint8Array(state.analyser.frequencyBinCount);
    state.analyser.getByteFrequencyData(dataArray);
    
    // Calculate average volume
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
    }
    const average = sum / dataArray.length;
    
    // Normalize to 0-100%
    const percentage = Math.min(100, Math.max(0, average / 128 * 100));
    
    // Invert because we're showing the "empty" portion
    elements.volumeLevel.style.width = (100 - percentage) + '%';
    
    // Change color based on level
    if (percentage > 70) {
        elements.volumeLevel.style.backgroundColor = 'rgba(244, 67, 54, 0.5)'; // Red
    } else if (percentage > 30) {
        elements.volumeLevel.style.backgroundColor = 'rgba(255, 235, 59, 0.5)'; // Yellow
    } else {
        elements.volumeLevel.style.backgroundColor = 'rgba(0, 0, 0, 0.5)'; // Dark
    }
}

// Handle speech/silence state transitions
function handleSpeechState(isSilent) {
    if (state.isSpeaking && isSilent) {
        // Transition from speaking to silence
        if (!state.silenceTimer) {
            state.silenceTimer = setTimeout(() => {
                // Silence persisted long enough - process the audio
                elements.streamButton.innerHTML = '<i class="fas fa-cog fa-spin"></i> Processing...';
                elements.streamButton.classList.remove('recording');
                elements.streamButton.classList.add('processing');
                addSystemMessage('Detected pause in speech, processing response...');
            }, CLIENT_SILENCE_DURATION_MS);
        }
    } else if (!state.isSpeaking && !isSilent) {
        // Transition from silence to speaking
        state.isSpeaking = true;
        elements.streamButton.innerHTML = '<i class="fas fa-microphone"></i> Listening...';
        elements.streamButton.classList.add('recording');
        elements.streamButton.classList.remove('processing');
        
        // Clear silence timer
        if (state.silenceTimer) {
            clearTimeout(state.silenceTimer);
            state.silenceTimer = null;
        }
    } else if (state.isSpeaking && !isSilent) {
        // Still speaking, reset silence timer
        if (state.silenceTimer) {
            clearTimeout(state.silenceTimer);
            state.silenceTimer = null;
        }
    }
    
    // Update speaking state for non-silent audio
    if (!isSilent) {
        state.isSpeaking = true;
    }
}

// Send audio chunk to server
function sendAudioChunk(audioData, speaker) {
    if (!state.socket || !state.socket.connected) {
        console.warn('Cannot send audio: socket not connected');
        return;
    }
    
    const wavData = createWavBlob(audioData, 24000);
    const reader = new FileReader();
    
    reader.onloadend = function() {
        const base64data = reader.result;
        
        // Send to server using Socket.IO
        state.socket.emit('stream_audio', {
            speaker: speaker,
            audio: base64data
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
            console.error("Error getting frequency data:", e);
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

