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
- Signal handling for proper Ctrl+C shutdown
"""

import asyncio
import os
import logging
import signal
import sys
from typing import Dict, List
import unicodedata

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
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

# Global list to track running pipeline tasks
PIPELINE_TASKS: List[asyncio.Task] = []

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
        # Initialize services
        openai_stt = OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            model="whisper-1"
        )
        openai_llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            model="gpt-4o"
        )
        fish_tts = FishAudioTTSService(
            api_key=os.getenv("FISH_API_KEY"),
            model=os.getenv("FISH_MODEL_ID"),
            output_format="pcm",
            params=FishAudioTTSService.InputParams(language=Language.ZH_CN),
        )
        context = OpenAILLMContext(
            messages=[{
                "role": "system",
                "content": (
                    "你是一个有帮助的中文语音助理。请始终用简体中文与用户对话，"
                    "每次回答必须控制在40个汉字以内（或20秒语音），只回答用户最新的问题，"
                    "如果用户一次问多个问题，只回答最后一个。如果答案太长，"
                    "请简要总结或礼貌拒绝。不要重复未回答的问题。"
                    "所有回复都要简洁自然，适合语音播放。"
                )
            }]
        )
        context_aggregator = openai_llm.create_context_aggregator(context)

        # 自定义文本截断处理器，防止 TTS 超长
        class TruncateTextProcessor(FrameProcessor):
            def __init__(self, max_chars=100):
                super().__init__()
                self.max_chars = max_chars
                self.trunc_hint = "（已截断，问题请简短）"

            async def process_frame(self, frame, direction):
                # 只处理文本帧，其它帧直接透传
                if isinstance(frame, TextFrame) and hasattr(frame, 'text') and isinstance(frame.text, str):
                    text = frame.text.strip()
                    if len(text) == 0:
                        frame.text = "抱歉，我没有听清，请再说一遍。"
                    elif len(text) > self.max_chars:
                        frame.text = text[:self.max_chars] + self.trunc_hint
                await self.push_frame(frame, direction)

        pipeline = Pipeline([
            transport.input(),
            openai_stt,
            context_aggregator.user(),
            openai_llm,
            # TruncateTextProcessor(max_chars=100),
            fish_tts,
            transport.output(),
            context_aggregator.assistant(),
        ])
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=False,
                enable_usage_metrics=False
            )
        )
        await task.queue_frames([LLMMessagesFrame(context.messages)])

        runner = PipelineRunner()
        logger.info("🎙️ Voice assistant pipeline started, waiting for browser connection...")

        await runner.run(task)

    except asyncio.CancelledError:
        logger.info("Voice assistant task was cancelled")
        # No need to re-raise, expected on shutdown
    except Exception as e:
        logger.error(f"Voice assistant error: {e}", exc_info=True)
    finally:
        try:
            await runner.cleanup()
        except Exception as e:
            logger.warning(f"Error during runner cleanup: {e}")
        logger.info("Pipeline task finished.")


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
    async def offer(request: dict):
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
                
                # Start voice assistant in background and track the task
                task = asyncio.create_task(run_voice_assistant(transport))
                PIPELINE_TASKS.append(task)
                task.add_done_callback(PIPELINE_TASKS.remove)

            # Get answer and store connection
            answer = webrtc_connection.get_answer()
            connections_map[answer["pc_id"]] = webrtc_connection
            
            logger.info(f"WebRTC connection established: {answer['pc_id']}")
            return answer
            
        except Exception as e:
            logger.error(f"Error handling WebRTC offer: {e}", exc_info=True)
            raise
    
    # Add shutdown endpoint for programmatic termination
    @app.post("/shutdown")
    async def shutdown():
        """Programmatically shutdown the server"""
        logger.info("Shutdown request received")
        
        # Cancel all pipeline tasks
        for task in PIPELINE_TASKS[:]:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish cancellation
        if PIPELINE_TASKS:
            await asyncio.gather(*PIPELINE_TASKS, return_exceptions=True)
        
        # Send SIGTERM to self to shutdown uvicorn
        os.kill(os.getpid(), signal.SIGTERM)
        
        return {"message": "Shutting down..."}
    
    return app


async def main():
    """Main function to start the WebRTC server with proper signal handling."""
    
    print("\n" + "="*60)
    print("🎤 语音助手: WebRTC + Fish TTS")
    print("="*60)
    
    if not validate_environment():
        return
        
    app = create_webrtc_server()
    
    # Add shutdown event handler to FastAPI app
    @app.on_event("shutdown")
    async def shutdown_event():
        print("\n🛑 Shutting down gracefully...")
        print("Canceling all running pipeline tasks...")
        
        # Cancel all pipeline tasks
        for task in PIPELINE_TASKS[:]:  # Copy list to avoid modification during iteration
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish cancellation
        if PIPELINE_TASKS:
            await asyncio.gather(*PIPELINE_TASKS, return_exceptions=True)
        
        print("All pipeline tasks stopped.")
    
    print("🌐 Starting WebRTC server...")
    print("📱 Open your browser and visit: http://localhost:7860")
    print("🎙️  Click '连接并开始对话' to start voice conversation")
    print("🔊 Audio will be played through your browser with echo cancellation")
    print("⏹️  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Use uvicorn.run with its built-in signal handling
    config = uvicorn.Config(
        app,
        host="localhost",
        port=7860,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    
    # Install custom signal handler for SIGINT
    def signal_handler(signum, frame):
        print(f"\n🛑 Signal {signum} received, initiating shutdown...")
        # Send shutdown request to ourselves
        try:
            import requests
            requests.post("http://localhost:7860/shutdown", timeout=1)
        except:
            # If the requests fails, force shutdown with SIGTERM
            os.kill(os.getpid(), signal.SIGTERM)
    
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    await server.serve()


# Make sure this is the last top-level statement before __main__
app = create_webrtc_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    finally:
        print("✅ Server ended. Goodbye!") 