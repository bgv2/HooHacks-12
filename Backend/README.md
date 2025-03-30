# csm-conversation-bot

## Overview
The CSM Conversation Bot is an application that utilizes advanced audio processing and language model technologies to facilitate real-time voice conversations with an AI assistant. The bot processes audio streams, converts spoken input into text, generates responses using the Llama 3.2 model, and converts the text back into audio for seamless interaction.

## Project Structure
```
csm-conversation-bot
├── api
│   ├── app.py                # Main entry point for the API
│   ├── routes.py             # Defines API routes
│   └── socket_handlers.py     # Manages Socket.IO events
├── src
│   ├── audio
│   │   ├── processor.py       # Audio processing functions
│   │   └── streaming.py       # Audio streaming management
│   ├── llm
│   │   ├── generator.py       # Response generation using Llama 3.2
│   │   └── tokenizer.py       # Text tokenization functions
│   ├── models
│   │   ├── audio_model.py     # Audio processing model
│   │   └── conversation.py     # Conversation state management
│   ├── services
│   │   ├── transcription_service.py # Audio to text conversion
│   │   └── tts_service.py     # Text to speech conversion
│   └── utils
│       ├── config.py          # Configuration settings
│       └── logger.py          # Logging utilities
├── static
│   ├── css
│   │   └── styles.css         # CSS styles for the web interface
│   ├── js
│   │   └── client.js          # Client-side JavaScript
│   └── index.html             # Main HTML file for the web interface
├── templates
│   └── index.html             # Template for rendering the main HTML page
├── config.py                  # Main configuration settings
├── requirements.txt           # Python dependencies
├── server.py                  # Entry point for running the application
└── README.md                  # Documentation for the project
```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/csm-conversation-bot.git
   cd csm-conversation-bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the application settings in `config.py` as needed.

## Usage
1. Start the server:
   ```
   python server.py
   ```

2. Open your web browser and navigate to `http://localhost:5000` to access the application.

3. Use the interface to start a conversation with the AI assistant.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.