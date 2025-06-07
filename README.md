# Voice Assistant with Fish TTS

Traditional STT → LLM → TTS pipeline using OpenAI + Fish Audio with WebRTC browser support.

## Features

- 🎤 **Browser Voice Chat**: WebRTC audio with echo cancellation
- 🎵 **Custom Voice**: Fish Audio TTS synthesis
- 🤖 **Smart Conversation**: OpenAI GPT-4o + Whisper STT
- 🌐 **Web Interface**: FastAPI server, no app installation needed

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Setup Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run
```bash
python openai_fish_demo.py
```

Visit `http://localhost:7860` and start talking!

## Configuration

Edit `.env` with your API keys:
- `OPENAI_API_KEY` - OpenAI API key  
- `FISH_API_KEY` - Fish Audio API key
- `FISH_MODEL_ID` - Your Fish Audio model ID
- `OPENAI_API_BASE` - Optional custom endpoint

## Architecture

```
Browser Mic → OpenAI Whisper → GPT-4o → Fish TTS → Browser Speaker
```

Built with [Pipecat](https://github.com/pipecat-ai/pipecat) framework.

## Troubleshooting

- **No audio**: Check browser microphone/speaker permissions
- **API errors**: Verify keys in `.env` file
- **Connection issues**: Try incognito mode or different browser