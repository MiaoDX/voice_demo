#!/usr/bin/env python3

"""
OpenAI Realtime API + Fish TTS Local Demo

A clean implementation that combines OpenAI's Realtime API with Fish TTS 
for custom voice synthesis using local audio transport.

Features:
- Real-time speech-to-speech conversation using local microphone/speakers
- Custom voice synthesis using Fish TTS
- Clean pipeline architecture for easy debugging
"""

import asyncio
import os
import logging

from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.services.openai_realtime_beta import (
    OpenAIRealtimeBetaLLMService,
    SessionProperties,
    InputAudioTranscription,
    TurnDetection,
)
from pipecat.services.fish.tts import FishAudioTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

# Load environment variables
load_dotenv()

# Configure logging for better debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Set more specific log levels for better debugging
logging.getLogger("pipecat").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.WARNING)


def validate_environment() -> bool:
    """Validate required environment variables"""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for Realtime API",
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


async def create_services():
    """Create and configure the services"""
    
    # Create OpenAI Realtime service with simplified configuration
    openai_realtime = OpenAIRealtimeBetaLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),  # Optional custom base URL
        session_properties=SessionProperties(
            modalities=["text", "audio"],
            instructions="""You are a helpful AI assistant with a warm, friendly personality. 
            Keep your responses conversational, concise, and engaging.
            Speak naturally as if having a friendly conversation.""",
            voice="alloy",  # This won't be used since we're using Fish TTS
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            input_audio_transcription=InputAudioTranscription(
                model="whisper-1"
            ),
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=800,
            ),
            temperature=0.8,
        ),
    )
    
    # Create Fish TTS service
    fish_model_id = os.getenv("FISH_MODEL_ID")
    logger.info(f"Initializing Fish TTS with model: {fish_model_id}")
    
    fish_tts = FishAudioTTSService(
        api_key=os.getenv("FISH_API_KEY"),
        model=fish_model_id,
        output_format="pcm",
        sample_rate=24000,
        params=FishAudioTTSService.InputParams(
            latency="normal",
            prosody_speed=1.0,
            prosody_volume=0,
        ),
    )
    
    return openai_realtime, fish_tts


async def create_audio_transport():
    """Create and configure audio transport"""
    
    audio_params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=16000
        )
    )
    
    return LocalAudioTransport(audio_params)


async def main():
    """Main function to set up and run the demo"""
    
    print("\n" + "="*60)
    print("🎤 OpenAI Realtime + Fish TTS Demo")
    print("="*60)
    
    # Validate environment
    if not validate_environment():
        return
    
    try:
        # Create services
        logger.info("Initializing services...")
        openai_realtime, fish_tts = await create_services()
        audio_transport = await create_audio_transport()
        
        # Create context for conversation management
        context = OpenAILLMContext()
        
        # Build the processing pipeline - simplified without custom processor
        logger.info("Building pipeline...")
        pipeline = Pipeline([
            audio_transport.input(),     # Audio input (microphone + VAD)
            openai_realtime,            # OpenAI Realtime API processing
            fish_tts,                   # Fish TTS service directly
            audio_transport.output(),    # Audio output (speakers)
        ])
        
        # Create and start pipeline
        task = PipelineTask(pipeline)
        runner = PipelineRunner()
        
        print("🎙️  Starting conversation... Speak into your microphone!")
        print("🔊 Audio will be played through your speakers")
        print("⏹️  Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Send initial context and start the pipeline
        await task.queue_frames([
            OpenAILLMContextFrame(context),
        ])
        
        # Run the pipeline - this will automatically send StartFrame to all processors
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