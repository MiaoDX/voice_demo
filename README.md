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

### Run (Recommended: Hypercorn)

For best reliability and clean shutdown (Ctrl+C), use [Hypercorn](https://pgjones.gitlab.io/hypercorn/):

```bash
pip install hypercorn
hypercorn openai_fish_demo:app --bind localhost:7860
```

- This will serve your app on http://localhost:7860
- Press Ctrl+C to stop the server cleanly
- Hypercorn handles signals properly (no custom signal handling needed)

### Run (Legacy: Direct Python)

You can also run directly for development (uses uvicorn internally):

```bash
python openai_fish_demo.py
```

Visit `http://localhost:7860` and start talking!

## Configuration

Edit `.env` with your API keys:
- `OPENAI_API_KEY` - OpenAI API key  
- `OPENAI_API_BASE` - Optional custom endpoint
- `FISH_API_KEY` - Fish Audio API key
- `FISH_MODEL_ID` - Your Fish Audio model ID

## Architecture

```
Browser Mic → OpenAI Whisper → GPT-4o → Fish TTS → Browser Speaker
```

Built with [Pipecat](https://github.com/pipecat-ai/pipecat) framework.

## ASGI Server Support

For production deployment, your `openai_fish_demo.py` exposes a top-level `app` variable:

```python
app = create_webrtc_server()
```

This allows ASGI servers (Hypercorn, Uvicorn, Gunicorn) to find and run your FastAPI app.

## Troubleshooting

- **No audio**: Check browser microphone/speaker permissions
- **API errors**: Verify keys in `.env` file
- **Connection issues**: Try incognito mode or different browser
- **Server won't stop with Ctrl+C**: Use Hypercorn as shown above for best results