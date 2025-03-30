document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const startButton = document.getElementById('start-button');
    const interruptButton = document.getElementById('interrupt-button');
    const conversationDiv = document.getElementById('conversation');
    const connectionDot = document.getElementById('connection-dot');
    const connectionStatus = document.getElementById('connection-status');
    const whisperStatus = document.getElementById('whisper-status');
    const csmStatus = document.getElementById('csm-status');
    const llmStatus = document.getElementById('llm-status');
    const webrtcStatus = document.getElementById('webrtc-status');
    const micAnimation = document.getElementById('mic-animation');
    const loadingDiv = document.getElementById('loading');
    const loadingText = document.getElementById('loading-text');
    
    // State variables
    let socket;
    let isConnected = false;
    let isListening = false;
    let isAiSpeaking = false;
    let audioContext;
    let mediaStream;
    let audioRecorder;
    let audioProcessor;
    const audioChunks = [];
    
    // WebRTC variables
    let peerConnection;
    let dataChannel;
    let hasActiveConnection = false;
    
    // Audio playback
    let audioQueue = [];
    let isPlaying = false;
    
    // Configuration variables
    let serverSampleRate = 24000;
    let clientSampleRate = 44100;
    let iceServers = [];
    
    // Initialize the application
    initApp();
    
    // Main initialization function
    function initApp() {
        updateConnectionStatus('connecting');
        setupSocketConnection();
        setupEventListeners();
    }
    
    // Set up Socket.IO connection with server
    function setupSocketConnection() {
        socket = io();
        
        socket.on('connect', () => {
            console.log('Connected to server');
            updateConnectionStatus('connected');
            isConnected = true;
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            updateConnectionStatus('disconnected');
            isConnected = false;
            cleanupAudio();
            cleanupWebRTC();
        });
        
        socket.on('session_ready', (data) => {
            console.log('Session ready:', data);
            updateModelStatus(data);
            clientSampleRate = data.client_sample_rate;
            serverSampleRate = data.server_sample_rate;
            iceServers = data.ice_servers;
            
            // Initialize WebRTC if models are available
            if (data.whisper_available && data.llm_available) {
                initializeWebRTC();
            }
        });
        
        socket.on('ready_for_speech', (data) => {
            console.log('Ready for speech:', data);
            startButton.disabled = false;
            addInfoMessage('Ready for conversation. Click "Start Listening" to begin.');
        });
        
        socket.on('webrtc_signal', (data) => {
            handleWebRTCSignal(data);
        });
        
        socket.on('transcription', (data) => {
            console.log('Transcription:', data);
            addUserMessage(data.text);
            loadingDiv.style.display = 'none';
        });
        
        socket.on('ai_response_text', (data) => {
            console.log('AI response text:', data);
            addAIMessage(data.text);
            loadingDiv.style.display = 'none';
        });
        
        socket.on('ai_speech_start', () => {
            console.log('AI started speaking');
            isAiSpeaking = true;
            interruptButton.disabled = false;
        });
        
        socket.on('ai_speech_chunk', (data) => {
            console.log('Received AI speech chunk');
            playAudioChunk(data.audio, data.is_last);
        });
        
        socket.on('ai_speech_end', () => {
            console.log('AI stopped speaking');
            isAiSpeaking = false;
            interruptButton.disabled = true;
        });
        
        socket.on('user_speech_start', () => {
            console.log('User speech detected');
            showSpeakingIndicator(true);
        });
        
        socket.on('processing_speech', () => {
            console.log('Processing speech');
            showSpeakingIndicator(false);
            showLoadingIndicator('Processing your speech...');
        });
        
        socket.on('no_speech_detected', () => {
            console.log('No speech detected');
            hideLoadingIndicator();
            addInfoMessage('No speech detected. Please try again.');
        });
        
        socket.on('ai_interrupted', () => {
            console.log('AI interrupted');
            clearAudioQueue();
            isAiSpeaking = false;
            interruptButton.disabled = true;
        });
        
        socket.on('ai_interrupted_by_user', () => {
            console.log('AI interrupted by user');
            clearAudioQueue();
            isAiSpeaking = false;
            interruptButton.disabled = true;
            addInfoMessage('AI interrupted by your speech');
        });
        
        socket.on('error', (data) => {
            console.error('Server error:', data);
            hideLoadingIndicator();
            addInfoMessage(`Error: ${data.message}`);
        });
    }
    
    // Set up UI event listeners
    function setupEventListeners() {
        startButton.addEventListener('click', toggleListening);
        interruptButton.addEventListener('click', interruptAI);
    }
    
    // Update UI connection status
    function updateConnectionStatus(status) {
        connectionDot.className = 'status-dot ' + status;
        
        switch (status) {
            case 'connected':
                connectionStatus.textContent = 'Connected';
                break;
            case 'connecting':
                connectionStatus.textContent = 'Connecting...';
                break;
            case 'disconnected':
                connectionStatus.textContent = 'Disconnected';
                startButton.disabled = true;
                interruptButton.disabled = true;
                break;
        }
    }
    
    // Update model status indicators
    function updateModelStatus(data) {
        whisperStatus.textContent = data.whisper_available ? 'Available' : 'Not Available';
        whisperStatus.style.color = data.whisper_available ? 'green' : 'red';
        
        csmStatus.textContent = data.csm_available ? 'Available' : 'Not Available';
        csmStatus.style.color = data.csm_available ? 'green' : 'red';
        
        llmStatus.textContent = data.llm_available ? 'Available' : 'Not Available';
        llmStatus.style.color = data.llm_available ? 'green' : 'red';
    }
    
    // Initialize WebRTC connection
    function initializeWebRTC() {
        if (!isConnected) return;
        
        const configuration = {
            iceServers: iceServers
        };
        
        peerConnection = new RTCPeerConnection(configuration);
        
        // Create data channel for WebRTC communication
        dataChannel = peerConnection.createDataChannel('audioData', {
            ordered: true
        });
        
        dataChannel.onopen = () => {
            console.log('WebRTC data channel open');
            hasActiveConnection = true;
            webrtcStatus.textContent = 'Connected';
            webrtcStatus.style.color = 'green';
            socket.emit('webrtc_connected', { status: 'connected' });
        };
        
        dataChannel.onclose = () => {
            console.log('WebRTC data channel closed');
            hasActiveConnection = false;
            webrtcStatus.textContent = 'Disconnected';
            webrtcStatus.style.color = 'red';
        };
        
        // Handle ICE candidates
        peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                socket.emit('webrtc_signal', {
                    type: 'ice_candidate',
                    candidate: event.candidate
                });
            }
        };
        
        // Log ICE connection state changes
        peerConnection.oniceconnectionstatechange = () => {
            console.log('ICE connection state:', peerConnection.iceConnectionState);
        };
        
        // Create offer
        peerConnection.createOffer()
            .then(offer => peerConnection.setLocalDescription(offer))
            .then(() => {
                socket.emit('webrtc_signal', {
                    type: 'offer',
                    sdp: peerConnection.localDescription
                });
            })
            .catch(error => {
                console.error('Error creating WebRTC offer:', error);
                webrtcStatus.textContent = 'Failed to Connect';
                webrtcStatus.style.color = 'red';
            });
    }
    
    // Handle WebRTC signals from the server
    function handleWebRTCSignal(data) {
        if (!peerConnection) return;
        
        if (data.type === 'answer') {
            peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp))
                .catch(error => console.error('Error setting remote description:', error));
        } 
        else if (data.type === 'ice_candidate') {
            peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate))
                .catch(error => console.error('Error adding ICE candidate:', error));
        }
    }
    
    // Clean up WebRTC connection
    function cleanupWebRTC() {
        if (dataChannel) {
            dataChannel.close();
        }
        
        if (peerConnection) {
            peerConnection.close();
        }
        
        dataChannel = null;
        peerConnection = null;
        hasActiveConnection = false;
        webrtcStatus.textContent = 'Not Connected';
        webrtcStatus.style.color = 'red';
    }
    
    // Toggle audio listening
    function toggleListening() {
        if (isListening) {
            stopListening();
        } else {
            startListening();
        }
    }
    
    // Start listening for audio
    async function startListening() {
        if (!isConnected) return;
        
        try {
            await initAudio();
            isListening = true;
            startButton.textContent = 'Stop Listening';
            startButton.innerHTML = `
                <svg class="button-icon" viewBox="0 0 24 24" fill="white">
                    <path d="M6 6h12v12H6z"></path>
                </svg>
                Stop Listening
            `;
        } catch (error) {
            console.error('Error starting audio:', error);
            addInfoMessage('Error accessing microphone. Please check permissions.');
        }
    }
    
    // Stop listening for audio
    function stopListening() {
        cleanupAudio();
        isListening = false;
        startButton.innerHTML = `
            <svg class="button-icon" viewBox="0 0 24 24" fill="white">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z"></path>
            </svg>
            Start Listening
        `;
        showSpeakingIndicator(false);
    }
    
    // Initialize audio capture
    async function initAudio() {
        // Request microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: clientSampleRate,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        
        // Initialize AudioContext
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: clientSampleRate
        });
        
        // Create audio source from stream
        const source = audioContext.createMediaStreamSource(mediaStream);
        
        // Create ScriptProcessor for audio processing
        const bufferSize = 4096;
        audioProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
        
        // Process audio data
        audioProcessor.onaudioprocess = (event) => {
            if (!isListening || isAiSpeaking) return;
            
            const input = event.inputBuffer.getChannelData(0);
            const audioData = convertFloat32ToInt16(input);
            sendAudioChunk(audioData);
        };
        
        // Connect the nodes
        source.connect(audioProcessor);
        audioProcessor.connect(audioContext.destination);
    }
    
    // Clean up audio resources
    function cleanupAudio() {
        if (audioProcessor) {
            audioProcessor.disconnect();
            audioProcessor = null;
        }
        
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        
        if (audioContext && audioContext.state !== 'closed') {
            audioContext.close().catch(error => console.error('Error closing AudioContext:', error));
        }
        
        audioChunks.length = 0;
    }
    
    // Convert Float32Array to Int16Array for sending to server
    function convertFloat32ToInt16(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            // Convert float [-1.0, 1.0] to int16 [-32768, 32767]
            int16Array[i] = Math.max(-32768, Math.min(32767, Math.floor(float32Array[i] * 32768)));
        }
        return int16Array;
    }
    
    // Send audio chunk to server
    function sendAudioChunk(audioData) {
        if (!isConnected || !isListening) return;
        
        // Convert to base64 for transmission
        const base64Audio = arrayBufferToBase64(audioData.buffer);
        
        // Send via Socket.IO (could use WebRTC's DataChannel for lower latency in production)
        socket.emit('audio_stream', { audio: base64Audio });
    }
    
    // Play audio chunk received from server
    function playAudioChunk(base64Audio, isLast) {
        const audioData = base64ToArrayBuffer(base64Audio);
        
        // Add to queue
        audioQueue.push({
            data: audioData,
            isLast: isLast
        });
        
        // Start playing if not already playing
        if (!isPlaying) {
            playNextAudioChunk();
        }
    }
    
    // Play the next audio chunk in the queue
    function playNextAudioChunk() {
        if (audioQueue.length === 0) {
            isPlaying = false;
            return;
        }
        
        isPlaying = true;
        const chunk = audioQueue.shift();
        
        try {
            // Create audio context if needed
            if (!audioContext || audioContext.state === 'closed') {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            // Resume audio context if suspended
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
            
            // Decode the WAV data
            audioContext.decodeAudioData(chunk.data, (buffer) => {
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);
                
                // When playback ends, play the next chunk
                source.onended = () => {
                    playNextAudioChunk();
                };
                
                source.start(0);
                
                // If it's the last chunk, update UI
                if (chunk.isLast) {
                    setTimeout(() => {
                        isAiSpeaking = false;
                        interruptButton.disabled = true;
                    }, buffer.duration * 1000);
                }
            }, (error) => {
                console.error('Error decoding audio data:', error);
                playNextAudioChunk(); // Skip this chunk and try the next
            });
        } catch (error) {
            console.error('Error playing audio chunk:', error);
            playNextAudioChunk(); // Try the next chunk
        }
    }
    
    // Clear the audio queue (used when interrupting)
    function clearAudioQueue() {
        audioQueue.length = 0;
        isPlaying = false;
        
        // Stop any currently playing audio
        if (audioContext) {
            audioContext.suspend();
        }
    }
    
    // Send interrupt signal to server
    function interruptAI() {
        if (!isConnected || !isAiSpeaking) return;
        
        socket.emit('interrupt_ai');
        clearAudioQueue();
    }
    
    // Convert ArrayBuffer to Base64 string
    function arrayBufferToBase64(buffer) {
        const binary = new Uint8Array(buffer);
        let base64 = '';
        const len = binary.byteLength;
        for (let i = 0; i < len; i++) {
            base64 += String.fromCharCode(binary[i]);
        }
        return window.btoa(base64);
    }
    
    // Convert Base64 string to ArrayBuffer
    function base64ToArrayBuffer(base64) {
        const binaryString = window.atob(base64);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }
    
    // Add user message to conversation
    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.textContent = text;
        conversationDiv.appendChild(messageDiv);
        conversationDiv.scrollTop = conversationDiv.scrollHeight;
    }
    
    // Add AI message to conversation
    function addAIMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai-message';
        messageDiv.textContent = text;
        conversationDiv.appendChild(messageDiv);
        conversationDiv.scrollTop = conversationDiv.scrollHeight;
    }
    
    // Add info message to conversation
    function addInfoMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'info-message';
        messageDiv.textContent = text;
        conversationDiv.appendChild(messageDiv);
        conversationDiv.scrollTop = conversationDiv.scrollHeight;
    }
    
    // Show/hide speaking indicator
    function showSpeakingIndicator(show) {
        micAnimation.style.display = show ? 'flex' : 'none';
    }
    
    // Show loading indicator
    function showLoadingIndicator(text) {
        loadingText.textContent = text || 'Processing...';
        loadingDiv.style.display = 'block';
    }
    
    // Hide loading indicator
    function hideLoadingIndicator() {
        loadingDiv.style.display = 'none';
    }
});