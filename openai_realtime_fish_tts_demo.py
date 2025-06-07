#!/usr/bin/env python3

"""
Working Voice-to-Voice Demo: Traditional STT + LLM + TTS Pipeline

A clean, working implementation using traditional speech-to-text, LLM, and text-to-speech
pipeline with Fish TTS for custom voice synthesis. This version incorporates all fixes
from debugging sessions.

Features:
- Traditional pipeline: STT → LLM → TTS
- Custom voice synthesis using Fish TTS
- Proper context aggregation and conversation management
- Echo prevention (use headphones recommended)

Fixed Issues:
- AttributeError with LLMUserResponseAggregator (use modern API)
- Pipeline structure for proper conversation flow
- Audio output issues (proper sample rates and volume)
- StartFrame initialization problems
"""

import asyncio
import os
import logging

from dotenv import load_dotenv

from pipecat.frames.frames import LLMMessagesFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.fish.tts import FishAudioTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transcriptions.language import Language

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


async def main():
    """Main function to set up and run the demo"""
    
    print("\n" + "="*60)
    print("🎤 Voice-to-Voice Demo: Traditional STT + LLM + TTS")
    print("="*60)
    
    # Validate environment
    if not validate_environment():
        return
    
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
        
        # Create audio transport with consistent sample rate
        logger.info("Setting up audio transport...")
        audio_params = LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(sample_rate=16000)  # VAD uses 16kHz
        )
        audio_transport = LocalAudioTransport(audio_params)
        
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
            audio_transport.input(),         # Audio input + VAD
            openai_stt,                     # Speech-to-text
            context_aggregator.user(),       # User message aggregation
            openai_llm,                     # LLM processing
            fish_tts,                       # Fish TTS synthesis
            audio_transport.output(),        # Audio output
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
        
        print("🎙️  Starting conversation... Speak into your microphone!")
        print("🔊 Audio will be played through your speakers")
        print("💡 Use headphones to prevent feedback!")
        print("⏹️  Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Run the pipeline
        await runner.run(task)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
    finally:
        print("✅ Demo ended. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main()) 