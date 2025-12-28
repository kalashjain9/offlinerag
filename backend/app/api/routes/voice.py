"""
OfflineRAG - Voice API Routes
=============================

Endpoints for speech-to-text and text-to-speech.
"""

import io
import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse
from loguru import logger

from app.core.config import settings
from app.models.schemas import TranscriptionResponse, TTSRequest, TTSResponse
from app.services.voice import voice_service

router = APIRouter(prefix="/voice", tags=["Voice"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="en")
):
    """
    Transcribe audio to text using local Whisper model.
    
    Supports: WAV, MP3, FLAC, M4A, OGG
    """
    if not voice_service.asr_available:
        raise HTTPException(
            status_code=503,
            detail="Speech recognition not available"
        )
    
    # Validate file type
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in settings.SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {ext}"
        )
    
    # Save to temp file
    temp_path = settings.TEMP_DIR / f"{uuid.uuid4()}{ext}"
    
    try:
        async with aiofiles.open(temp_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        
        # Transcribe
        text, detected_lang, duration = await voice_service.asr.transcribe(
            str(temp_path),
            language=language if language != "auto" else None
        )
        
        return TranscriptionResponse(
            text=text,
            language=detected_lang,
            confidence=0.9,  # Whisper doesn't provide confidence
            duration=duration
        )
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


@router.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech using local TTS model.
    
    Returns: Audio file (WAV format)
    """
    if not voice_service.tts_available:
        raise HTTPException(
            status_code=503,
            detail="Text-to-speech not available"
        )
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")
    
    try:
        audio_data, duration = await voice_service.tts.synthesize_to_bytes(
            request.text,
            rate=request.rate
        )
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Audio-Duration": str(duration)
            }
        )
        
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def voice_status():
    """Get voice services status including current state."""
    return voice_service.get_status() | {
        "asr_model": settings.WHISPER_MODEL if voice_service.asr_available else None,
        "tts_model": settings.TTS_MODEL if voice_service.tts_available else None
    }


@router.post("/stop")
async def stop_voice():
    """Stop any ongoing voice operations (TTS playback, etc.)."""
    voice_service.cancel_operations()
    return {"status": "stopped"}


@router.post("/synthesize/stream")
async def synthesize_speech_streaming(request: TTSRequest):
    """
    Convert text to speech with streaming for interruptible playback.
    
    Returns: Chunked audio stream (WAV format)
    """
    if not voice_service.tts_available:
        raise HTTPException(
            status_code=503,
            detail="Text-to-speech not available"
        )
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    async def generate():
        async for chunk in voice_service.speak_streaming(request.text):
            yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech.wav"
        }
    )
