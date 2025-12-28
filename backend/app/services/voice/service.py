"""
OfflineRAG - Voice Service
==========================

Handles speech-to-text (ASR) and text-to-speech (TTS) using local models.
ASR: OpenAI Whisper
TTS: Piper TTS
"""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
import numpy as np

from app.core.config import settings


class ASRService:
    """
    Automatic Speech Recognition using OpenAI Whisper.
    Fully offline after model download.
    """
    
    def __init__(self):
        self._model = None
        self._initialized = False
    
    async def initialize(self):
        """Load the Whisper model."""
        if self._initialized:
            return
        
        logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL}")
        
        try:
            # Check if ffmpeg is available
            import shutil
            if not shutil.which("ffmpeg"):
                logger.warning("FFmpeg not found in PATH. Voice transcription may fail for non-WAV files.")
                logger.warning("Install FFmpeg: https://ffmpeg.org/download.html or 'choco install ffmpeg' (admin)")
            
            import whisper
            
            self._model = await asyncio.to_thread(
                whisper.load_model,
                settings.WHISPER_MODEL,
                device=settings.WHISPER_DEVICE
            )
            
            self._initialized = True
            logger.info("Whisper ASR model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    @property
    def is_available(self) -> bool:
        """Check if ASR is available."""
        return self._initialized and self._model is not None
    
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code
            
        Returns:
            Tuple of (transcription, detected_language, duration)
        """
        if not self._initialized:
            await self.initialize()
        
        language = language or settings.WHISPER_LANGUAGE
        
        # Check if file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Transcribe
            result = await asyncio.to_thread(
                self._model.transcribe,
                audio_path,
                language=language if language != "auto" else None,
                fp16=False  # CPU compatibility
            )
        except FileNotFoundError as e:
            # This usually means ffmpeg is not installed
            import shutil
            if not shutil.which("ffmpeg"):
                raise RuntimeError(
                    "FFmpeg is required for audio transcription. "
                    "Install it from https://ffmpeg.org/download.html or run 'choco install ffmpeg' as admin."
                ) from e
            raise
        
        # Estimate duration from transcription (Whisper provides segments)
        duration = 0.0
        if "segments" in result and result["segments"]:
            last_segment = result["segments"][-1]
            duration = last_segment.get("end", 0.0)
        
        return result["text"], result.get("language", language), duration
    
    async def transcribe_bytes(
        self,
        audio_data: bytes,
        file_format: str = "wav",
        language: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_data: Audio bytes
            file_format: Audio format (wav, mp3, etc.)
            language: Optional language code
            
        Returns:
            Tuple of (transcription, detected_language, duration)
        """
        # Save to temp file
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_format}",
            delete=False
        ) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        try:
            result = await self.transcribe(tmp_path, language)
            return result
        finally:
            os.unlink(tmp_path)


class TTSService:
    """
    Text-to-Speech using Piper TTS or fallback options.
    Fully offline with natural-sounding voices.
    
    Features:
    - Natural pacing
    - Interruptible playback
    - Multiple voice options
    """
    
    def __init__(self):
        self._initialized = False
        self._voice = None
        self._sample_rate = 22050
        self._engine = None
        self._current_playback = None
        self._should_stop = False
    
    async def initialize(self):
        """Initialize TTS engine."""
        if self._initialized:
            return
        
        if not settings.TTS_ENABLED:
            logger.info("TTS disabled in configuration")
            return
        
        logger.info(f"Initializing TTS with voice: {settings.TTS_MODEL}")
        
        try:
            # Try pyttsx3 as reliable offline TTS
            import pyttsx3
            self._engine = pyttsx3.init()
            
            # Set properties
            self._engine.setProperty('rate', int(150 * settings.TTS_RATE))
            
            # Get available voices
            voices = self._engine.getProperty('voices')
            if voices:
                # Try to find a good voice
                for voice in voices:
                    if 'zira' in voice.name.lower() or 'david' in voice.name.lower():
                        self._engine.setProperty('voice', voice.id)
                        break
            
            self._initialized = True
            logger.info("pyttsx3 TTS initialized")
        except Exception as e:
            logger.warning(f"TTS initialization failed: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self._initialized and self._engine is not None
    
    def stop_playback(self):
        """Stop any current playback."""
        self._should_stop = True
        if self._engine:
            try:
                self._engine.stop()
            except:
                pass
    
    async def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        rate: float = None
    ) -> Tuple[str, float]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path
            rate: Speech rate multiplier
            
        Returns:
            Tuple of (audio_path, duration)
        """
        if not self._initialized or not self._engine:
            raise RuntimeError("TTS not available")
        
        self._should_stop = False
        rate = rate or settings.TTS_RATE
        
        # Generate output path if not provided
        if not output_path:
            output_path = tempfile.mktemp(suffix=".wav")
        
        # Set rate
        current_rate = int(150 * rate)
        self._engine.setProperty('rate', current_rate)
        
        # Save to file
        await asyncio.to_thread(
            self._engine.save_to_file,
            text,
            output_path
        )
        await asyncio.to_thread(self._engine.runAndWait)
        
        # Calculate duration from file
        duration = len(text.split()) / (rate * 2.5)  # Rough estimate
        
        return output_path, duration
    
    async def synthesize_to_bytes(
        self,
        text: str,
        rate: float = None
    ) -> Tuple[bytes, float]:
        """
        Synthesize speech to bytes.
        
        Args:
            text: Text to synthesize
            rate: Speech rate multiplier
            
        Returns:
            Tuple of (audio_bytes, duration)
        """
        output_path, duration = await self.synthesize(text, rate=rate)
        
        try:
            with open(output_path, "rb") as f:
                audio_data = f.read()
            return audio_data, duration
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    async def speak_streaming(
        self,
        text: str,
        chunk_size: int = 50
    ):
        """
        Speak text with interruptible streaming.
        
        Yields audio chunks for progressive playback.
        """
        if not self._initialized or not self._engine:
            return
        
        self._should_stop = False
        
        # Split into sentences for natural pauses
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if self._should_stop:
                break
            
            if sentence.strip():
                audio_bytes, _ = await self.synthesize_to_bytes(sentence)
                yield audio_bytes


class VoiceService:
    """
    Unified voice service combining ASR and TTS.
    
    Features:
    - Async-first design
    - Interruptible operations
    - Cancel support
    - Status tracking
    """
    
    def __init__(self):
        self.asr = ASRService()
        self.tts = TTSService()
        self._initialized = False
        self._is_listening = False
        self._is_speaking = False
        self._current_operation = None
    
    async def initialize(self):
        """Initialize both ASR and TTS."""
        if self._initialized:
            return
        
        logger.info("Initializing voice services...")
        
        # Initialize ASR (Whisper)
        try:
            await self.asr.initialize()
        except Exception as e:
            logger.warning(f"ASR initialization failed: {e}")
        
        # Initialize TTS
        try:
            await self.tts.initialize()
        except Exception as e:
            logger.warning(f"TTS initialization failed: {e}")
        
        self._initialized = True
        logger.info("Voice services initialized")
    
    @property
    def asr_available(self) -> bool:
        """Check if ASR is available."""
        return self.asr.is_available
    
    @property
    def tts_available(self) -> bool:
        """Check if TTS is available."""
        return self.tts.is_available
    
    @property
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._is_listening
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking
    
    def get_status(self) -> dict:
        """Get voice service status."""
        return {
            "asr_available": self.asr_available,
            "tts_available": self.tts_available,
            "is_listening": self._is_listening,
            "is_speaking": self._is_speaking,
            "initialized": self._initialized
        }
    
    def cancel_operations(self):
        """Cancel any ongoing voice operations."""
        if self._is_speaking:
            self.tts.stop_playback()
            self._is_speaking = False
        self._is_listening = False
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Audio data in bytes
            language: Optional language code
            
        Returns:
            Transcribed text
        """
        if not self.asr_available:
            raise RuntimeError("ASR not available")
        
        self._is_listening = True
        try:
            return await self.asr.transcribe(audio_data, language)
        finally:
            self._is_listening = False
    
    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            file_path: Path to audio file
            language: Optional language code
            
        Returns:
            Transcribed text
        """
        if not self.asr_available:
            raise RuntimeError("ASR not available")
        
        self._is_listening = True
        try:
            return await self.asr.transcribe_file(file_path, language)
        finally:
            self._is_listening = False
    
    async def speak(
        self,
        text: str,
        rate: float = None
    ) -> Tuple[bytes, float]:
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
            rate: Speech rate multiplier
            
        Returns:
            Tuple of (audio_bytes, duration)
        """
        if not self.tts_available:
            raise RuntimeError("TTS not available")
        
        self._is_speaking = True
        try:
            return await self.tts.synthesize_to_bytes(text, rate)
        finally:
            self._is_speaking = False
    
    async def speak_streaming(self, text: str):
        """
        Stream speech synthesis for interruptible playback.
        
        Yields audio chunks that can be played progressively.
        """
        if not self.tts_available:
            return
        
        self._is_speaking = True
        try:
            async for chunk in self.tts.speak_streaming(text):
                yield chunk
        finally:
            self._is_speaking = False
    
    async def stop_speaking(self):
        """Stop any ongoing TTS playback."""
        self.tts.stop_playback()
        self._is_speaking = False


# Singleton instance
voice_service = VoiceService()
