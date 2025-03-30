// This file contains the client-side JavaScript code that handles audio streaming and communication with the server.

const SERVER_URL = window.location.hostname === 'localhost' ? 
    'http://localhost:5000' : window.location.origin;

const elements = {
    conversation: document.getElementById('conversation'),
    streamButton: document.getElementById('streamButton'),
    clearButton: document.getElementById('clearButton'),
    speakerSelection: document.getElementById('speakerSelect'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
};

const state = {
    socket: null,
    isStreaming: false,
    currentSpeaker: 0,
};

// Initialize the application
function initializeApp() {
    setupSocketConnection();
    setupEventListeners();
}

// Setup Socket.IO connection
function setupSocketConnection() {
    state.socket = io(SERVER_URL);

    state.socket.on('connect', () => {
        updateConnectionStatus(true);
    });

    state.socket.on('disconnect', () => {
        updateConnectionStatus(false);
    });

    state.socket.on('audio_response', handleAudioResponse);
    state.socket.on('transcription', handleTranscription);
}

// Setup event listeners
function setupEventListeners() {
    elements.streamButton.addEventListener('click', toggleStreaming);
    elements.clearButton.addEventListener('click', clearConversation);
    elements.speakerSelection.addEventListener('change', (event) => {
        state.currentSpeaker = event.target.value;
    });
}

// Update connection status UI
function updateConnectionStatus(isConnected) {
    elements.statusDot.classList.toggle('active', isConnected);
    elements.statusText.textContent = isConnected ? 'Connected' : 'Disconnected';
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
    if (state.isStreaming) return;

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    sendAudioChunk(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                state.isStreaming = false;
                elements.streamButton.innerHTML = 'Start Conversation';
            };

            state.isStreaming = true;
            elements.streamButton.innerHTML = 'Stop Conversation';
        })
        .catch(err => {
            console.error('Error accessing microphone:', err);
        });
}

// Stop streaming audio
function stopStreaming() {
    if (!state.isStreaming) return;

    // Logic to stop the media recorder would go here
}

// Send audio chunk to server
function sendAudioChunk(audioData) {
    const reader = new FileReader();
    reader.onloadend = () => {
        const arrayBuffer = reader.result;
        state.socket.emit('audio_chunk', { audio: arrayBuffer, speaker: state.currentSpeaker });
    };
    reader.readAsArrayBuffer(audioData);
}

// Handle audio response from server
function handleAudioResponse(data) {
    const audioElement = new Audio(URL.createObjectURL(new Blob([data.audio])));
    audioElement.play();
}

// Handle transcription response from server
function handleTranscription(data) {
    const messageElement = document.createElement('div');
    messageElement.textContent = `AI: ${data.transcription}`;
    elements.conversation.appendChild(messageElement);
}

// Clear conversation history
function clearConversation() {
    elements.conversation.innerHTML = '';
}

// Initialize the application when DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeApp);