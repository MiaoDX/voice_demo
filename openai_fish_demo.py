#!/usr/bin/env python3

"""
Working Voice-to-Voice Demo: Traditional STT + LLM + TTS Pipeline with WebRTC

A clean, working implementation using traditional speech-to-text, LLM, and text-to-speech
pipeline with Fish TTS for custom voice synthesis. Now includes FastAPI/WebRTC server
for browser-based audio with automatic echo cancellation.

Features:
- Traditional pipeline: STT → LLM → TTS
- Custom voice synthesis using Fish TTS
- WebRTC transport with browser-based echo cancellation
- FastAPI server for web interface
- Proper context aggregation and conversation management

Fixed Issues:
- AttributeError with LLMUserResponseAggregator (use modern API)
- Pipeline structure for proper conversation flow
- Audio output issues (proper sample rates and volume)
- StartFrame initialization problems
- WebRTC browser integration
"""

import asyncio
import os
import logging
from typing import Dict

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pipecat.frames.frames import LLMMessagesFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.fish.tts import FishAudioTTSService
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection, IceServer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from external libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class FrameLogger(FrameProcessor):
    """A simple pass-through processor that logs every frame for debugging."""
    def __init__(self, name: str = ""):
        super().__init__()
        self._name = name

    async def process_frame(self, frame, direction: FrameDirection):
        logger.info(f"[{self._name}] Frame: {frame}")
        await self.push_frame(frame, direction)


def validate_environment() -> bool:
    """Validate required environment variables"""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for STT and LLM",
        "FISH_API_KEY": "Fish Audio API key",  
        "FISH_MODEL_ID": "Fish Audio model ID"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        logger.info("Please set these in your .env file or environment")
        return False
    
    return True


async def run_voice_assistant(transport: SmallWebRTCTransport, args=None, started=False):
    """Voice assistant pipeline that runs with the given transport"""
    
    logger.info("🎤 Starting Voice Assistant with WebRTC transport")
    
    try:
        # Create OpenAI STT service
        logger.info("Initializing OpenAI STT...")
        openai_stt = OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),  # Optional
            model="whisper-1"
        )
        
        # Create OpenAI LLM service (traditional v1)
        logger.info("Initializing OpenAI LLM...")
        openai_llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),  # Optional
            model="gpt-4o"
        )
        
        # Create Fish TTS service with proper settings
        fish_model_id = os.getenv("FISH_MODEL_ID")
        logger.info(f"Initializing Fish TTS with model: {fish_model_id}")
        
        fish_tts = FishAudioTTSService(
            api_key=os.getenv("FISH_API_KEY"),
            model=fish_model_id,
            output_format="pcm",
            params=FishAudioTTSService.InputParams(language=Language.ZH_CN),
        )
        
        # Create conversation context
        context = OpenAILLMContext(
            messages=[
                {
                    "role": "system", 
                    "content": "你是一个有帮助的中文助理，请始终用简体中文与用户对话，回答要简洁自然。"
                }
            ]
        )
        
        # Create context aggregators using modern API
        context_aggregator = openai_llm.create_context_aggregator(context)
        
        # Build the traditional processing pipeline:
        # Audio Input → STT → User Context → LLM → Assistant Context → TTS → Audio Output
        logger.info("Building traditional pipeline...")
        pipeline = Pipeline([
            transport.input(),                   # WebRTC audio input with browser AEC
            openai_stt,                         # Speech-to-text
            context_aggregator.user(),           # User message aggregation
            openai_llm,                         # LLM processing
            fish_tts,                           # Fish TTS synthesis
            transport.output(),                  # WebRTC audio output
        ])
        
        # Create pipeline task with proper initialization
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=False,
                enable_usage_metrics=False
            )
        )
        
        # Initialize context
        await task.queue_frames([
            LLMMessagesFrame(context.messages)
        ])
        
        # Create and start runner
        runner = PipelineRunner()
        
        logger.info("🎙️ Voice assistant pipeline started, waiting for browser connection...")
        
        # Run the pipeline
        await runner.run(task)
        
    except Exception as e:
        logger.error(f"Voice assistant error: {e}", exc_info=True)
        raise


def create_webrtc_server():
    """Create FastAPI server with WebRTC support"""
    
    app = FastAPI(title="Pipecat Voice Assistant", description="WebRTC Voice Assistant with Fish TTS")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store connections by pc_id
    connections_map: Dict[str, SmallWebRTCConnection] = {}
    
    # ICE servers for WebRTC
    ice_servers = [
        IceServer(urls="stun:stun.l.google.com:19302")
    ]
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Serve the main page"""
        try:
            # Try to read HTML file from static directory
            html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return HTMLResponse(content=content)
            else:
                # Fallback: simple HTML if file doesn't exist
                return HTMLResponse(content="""
                    <html><body>
                    <h1>HTML file not found</h1>
                    <p>Please make sure static/index.html exists.</p>
                    </body></html>
                """)
        except Exception as e:
            logger.error(f"Error serving HTML: {e}")
            return HTMLResponse(content=f"<html><body><h1>Error: {e}</h1></body></html>")
    
    @app.post("/api/offer")
    async def offer(request: dict, background_tasks: BackgroundTasks):
        """Handle WebRTC offer and create answer"""
        
        logger.info("Received WebRTC offer")
        pc_id = request.get("pc_id", "default")
        
        try:
            if pc_id and pc_id in connections_map:
                # Reuse existing connection
                webrtc_connection = connections_map[pc_id]
                logger.info(f"Reusing existing connection for pc_id: {pc_id}")
                await webrtc_connection.renegotiate(
                    sdp=request["sdp"],
                    type=request["type"],
                    restart_pc=request.get("restart_pc", False),
                )
            else:
                # Create new connection
                webrtc_connection = SmallWebRTCConnection(ice_servers)
                await webrtc_connection.initialize(sdp=request["sdp"], type=request["type"])
                
                # Handle disconnection
                @webrtc_connection.event_handler("closed")
                async def handle_disconnected(conn: SmallWebRTCConnection):
                    logger.info(f"Cleaning up connection for pc_id: {conn.pc_id}")
                    connections_map.pop(conn.pc_id, None)
                
                # Create transport with this connection
                transport_params = TransportParams(
                    audio_out_enabled=True,
                    audio_in_enabled=True,
                    vad_enabled=True,
                    vad_analyzer=SileroVADAnalyzer()
                )
                transport = SmallWebRTCTransport(webrtc_connection=webrtc_connection, params=transport_params)
                
                # Start voice assistant in background
                background_tasks.add_task(run_voice_assistant, transport)
            
            # Get answer and store connection
            answer = webrtc_connection.get_answer()
            connections_map[answer["pc_id"]] = webrtc_connection
            
            logger.info(f"WebRTC connection established: {answer['pc_id']}")
            return answer
            
        except Exception as e:
            logger.error(f"Error handling WebRTC offer: {e}", exc_info=True)
            raise
    
    return app


def main():
    """Main function to start the WebRTC server"""
    
    print("\n" + "="*60)
    print("🎤 语音助手: WebRTC + Fish TTS")
    print("="*60)
    
    # Validate environment
    if not validate_environment():
        return
    
    try:
        # Create and start FastAPI server
        app = create_webrtc_server()
        
        print("🌐 Starting WebRTC server...")
        print("📱 Open your browser and visit: http://localhost:7860")
        print("🎙️  Click '连接并开始对话' to start voice conversation")
        print("🔊 Audio will be played through your browser with echo cancellation")
        print("⏹️  Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        uvicorn.run(app, host="localhost", port=7860, log_level="info")
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
    finally:
        print("✅ Server ended. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main()) if hasattr(asyncio, 'run') else main() 