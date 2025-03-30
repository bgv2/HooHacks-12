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

// New state variables to track incremental audio streaming
const streamingAudio = {
    messageElement: null,
    audioElement: null,
    chunks: [],
    totalChunks: 0,
    receivedChunks: 0,
    text: '',
    mediaSource: null,
    sourceBuffer: null,
    audioContext: null,
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
    addSystemMessage('Welcome to Sesame AI Voice Chat! Click "Start Conversation" to begin.');
}

// Initialize UI elements
function initializeUIElements() {
    // Store references to UI elements
    elements.conversation = document.getElementById('conversation');
    elements.streamButton = document.getElementById('streamButton');
    elements.clearButton = document.getElementById('clearButton');
    elements.thresholdSlider = document.getElementById('thresholdSlider');
    elements.thresholdValue = document.getElementById('thresholdValue');
    elements.visualizerCanvas = document.getElementById('audioVisualizer');
    elements.visualizerLabel = document.getElementById('visualizerLabel');
    elements.volumeLevel = document.getElementById('volumeLevel');
    elements.statusDot = document.getElementById('statusDot');
    elements.statusText = document.getElementById('statusText');
    elements.speakerSelection = document.getElementById('speakerSelect'); // Changed to match HTML
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
    
    // New event handlers for incremental audio streaming
    state.socket.on('audio_response_start', handleAudioResponseStart);
    state.socket.on('audio_response_chunk', handleAudioResponseChunk);
    state.socket.on('audio_response_complete', handleAudioResponseComplete);
    state.socket.on('processing_status', handleProcessingStatus);
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
    
    // Debug logging only if significant changes in audio patterns
    if (Math.random() < 0.05) { // Log only 5% of frames to avoid console spam
        console.log(`Audio: len=${inputData.length}, energy=${energy.toFixed(4)}, avg=${avgEnergy.toFixed(4)}, silent=${isSilent}`);
    }
    
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
    
    console.log(`Preparing audio chunk: length=${audioData.length}, speaker=${speaker}`);
    
    // Check for NaN or invalid values
    let hasInvalidValues = false;
    for (let i = 0; i < audioData.length; i++) {
        if (isNaN(audioData[i]) || !isFinite(audioData[i])) {
            hasInvalidValues = true;
            console.warn(`Invalid audio value at index ${i}: ${audioData[i]}`);
            break;
        }
    }
    
    if (hasInvalidValues) {
        console.warn('Audio data contains invalid values. Creating silent audio.');
        audioData = new Float32Array(audioData.length).fill(0);
    }
    
    try {
        // Create WAV blob
        const wavData = createWavBlob(audioData, 24000);
        console.log(`WAV blob created: ${wavData.size} bytes`);
        
        const reader = new FileReader();
        
        reader.onloadend = function() {
            try {
                // Get base64 data
                const base64data = reader.result;
                console.log(`Base64 data created: ${base64data.length} bytes`);
                
                // Send to server
                state.socket.emit('stream_audio', {
                    audio: base64data,
                    speaker: speaker
                });
                console.log('Audio chunk sent to server');
            } catch (err) {
                console.error('Error preparing audio data:', err);
            }
        };
        
        reader.onerror = function() {
            console.error('Error reading audio data as base64');
        };
        
        reader.readAsDataURL(wavData);
    } catch (err) {
        console.error('Error creating WAV data:', err);
    }
}

// Create WAV blob from audio data with improved error handling
function createWavBlob(audioData, sampleRate) {
    // Validate input
    if (!audioData || audioData.length === 0) {
        console.warn('Empty audio data provided to createWavBlob');
        audioData = new Float32Array(1024).fill(0); // Create 1024 samples of silence
    }
    
    // Function to convert Float32Array to Int16Array for WAV format
    function floatTo16BitPCM(output, offset, input) {
        for (let i = 0; i < input.length; i++, offset += 2) {
            // Ensure values are in -1 to 1 range
            const s = Math.max(-1, Math.min(1, input[i]));
            // Convert to 16-bit PCM
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
    }
    
    // Create WAV header
    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
    
    try {
        // Create WAV file with header - careful with buffer sizes
        const buffer = new ArrayBuffer(44 + audioData.length * 2);
        const view = new DataView(buffer);
        
        // RIFF identifier
        writeString(view, 0, 'RIFF');
        
        // File length (will be filled later)
        view.setUint32(4, 36 + audioData.length * 2, true);
        
        // WAVE identifier
        writeString(view, 8, 'WAVE');
        
        // fmt chunk identifier
        writeString(view, 12, 'fmt ');
        
        // fmt chunk length
        view.setUint32(16, 16, true);
        
        // Sample format (1 is PCM)
        view.setUint16(20, 1, true);
        
        // Mono channel
        view.setUint16(22, 1, true);
        
        // Sample rate
        view.setUint32(24, sampleRate, true);
        
        // Byte rate (sample rate * block align)
        view.setUint32(28, sampleRate * 2, true);
        
        // Block align (channels * bytes per sample)
        view.setUint16(32, 2, true);
        
        // Bits per sample
        view.setUint16(34, 16, true);
        
        // data chunk identifier
        writeString(view, 36, 'data');
        
        // data chunk length
        view.setUint32(40, audioData.length * 2, true);
        
        // Write the PCM samples
        floatTo16BitPCM(view, 44, audioData);
        
        // Create and return blob
        return new Blob([view], { type: 'audio/wav' });
    } catch (err) {
        console.error('Error in createWavBlob:', err);
        
        // Create a minimal valid WAV file with silence as fallback
        const fallbackSamples = new Float32Array(1024).fill(0);
        const fallbackBuffer = new ArrayBuffer(44 + fallbackSamples.length * 2);
        const fallbackView = new DataView(fallbackBuffer);
        
        writeString(fallbackView, 0, 'RIFF');
        fallbackView.setUint32(4, 36 + fallbackSamples.length * 2, true);
        writeString(fallbackView, 8, 'WAVE');
        writeString(fallbackView, 12, 'fmt ');
        fallbackView.setUint32(16, 16, true);
        fallbackView.setUint16(20, 1, true);
        fallbackView.setUint16(22, 1, true);
        fallbackView.setUint32(24, sampleRate, true);
        fallbackView.setUint32(28, sampleRate * 2, true);
        fallbackView.setUint16(32, 2, true);
        fallbackView.setUint16(34, 16, true);
        writeString(fallbackView, 36, 'data');
        fallbackView.setUint32(40, fallbackSamples.length * 2, true);
        floatTo16BitPCM(fallbackView, 44, fallbackSamples);
        
        return new Blob([fallbackView], { type: 'audio/wav' });
    }
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

// Handle processing status updates
function handleProcessingStatus(data) {
    console.log('Processing status update:', data);
    
    // Show processing status in UI
    if (data.status === 'generating_audio') {
        elements.streamButton.innerHTML = '<i class="fas fa-cog fa-spin"></i> Processing...';
        elements.streamButton.classList.add('processing');
        elements.streamButton.classList.remove('recording');
        
        // Show message to user
        addSystemMessage(data.message || 'Processing your request...');
    }
}

// Handle the start of an audio streaming response
function handleAudioResponseStart(data) {
    console.log('Audio response starting:', data);
    
    // Reset streaming audio state
    streamingAudio.chunks = [];
    streamingAudio.totalChunks = data.total_chunks;
    streamingAudio.receivedChunks = 0;
    streamingAudio.text = data.text;
    streamingAudio.complete = false;
    
    // Create message container now, so we can update it as chunks arrive
    const messageElement = document.createElement('div');
    messageElement.className = 'message ai processing';
    
    // Add text content if available
    if (data.text) {
        const textElement = document.createElement('p');
        textElement.textContent = data.text;
        messageElement.appendChild(textElement);
    }
    
    // Create audio element (will be populated as chunks arrive)
    const audioElement = document.createElement('audio');
    audioElement.controls = true;
    audioElement.className = 'audio-player';
    audioElement.textContent = 'Audio is being generated...';
    messageElement.appendChild(audioElement);
    
    // Add timestamp
    const timeElement = document.createElement('span');
    timeElement.className = 'message-time';
    timeElement.textContent = new Date().toLocaleTimeString();
    messageElement.appendChild(timeElement);
    
    // Add loading indicator
    const loadingElement = document.createElement('div');
    loadingElement.className = 'loading-indicator';
    loadingElement.innerHTML = '<div class="loading-spinner"></div><span>Generating audio response...</span>';
    messageElement.appendChild(loadingElement);
    
    // Add to conversation
    elements.conversation.appendChild(messageElement);
    
    // Auto-scroll to bottom
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
    
    // Store elements for later updates
    streamingAudio.messageElement = messageElement;
    streamingAudio.audioElement = audioElement;
}

// Handle an incoming audio chunk
function handleAudioResponseChunk(data) {
    console.log(`Received audio chunk ${data.chunk_index + 1}/${data.total_chunks}`);
    
    // Store the chunk
    streamingAudio.chunks[data.chunk_index] = data.audio;
    streamingAudio.receivedChunks++;
    
    // Update progress in the UI
    if (streamingAudio.messageElement) {
        const loadingElement = streamingAudio.messageElement.querySelector('.loading-indicator span');
        if (loadingElement) {
            loadingElement.textContent = `Generating audio response... ${Math.round((streamingAudio.receivedChunks / data.total_chunks) * 100)}%`;
        }
    }
    
    // If this is the first chunk, start playing it immediately for faster response
    if (data.chunk_index === 0 && streamingAudio.audioElement && elements.autoPlayResponses && elements.autoPlayResponses.checked) {
        try {
            streamingAudio.audioElement.src = data.audio;
            streamingAudio.audioElement.play().catch(err => console.warn('Auto-play failed:', err));
        } catch (e) {
            console.error('Error playing first chunk:', e);
        }
    }
    
    // If this is the last chunk or we've received all chunks, finalize the audio
    if (data.is_last || streamingAudio.receivedChunks >= data.total_chunks) {
        finalizeStreamingAudio();
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

// Add CSS styles for new UI elements
document.addEventListener('DOMContentLoaded', function() {
    // Add styles for processing state
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
    `;
    document.head.appendChild(style);
});

// Initialize the application when DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeApp);

