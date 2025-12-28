"""
OfflineRAG - System API Routes
==============================

Health checks, system info, and admin endpoints.
"""

import time
import psutil
from datetime import datetime
from fastapi import APIRouter, HTTPException
from loguru import logger

from app.core.config import settings
from app.models.schemas import HealthStatus, SystemInfo
from app.services.llm import llm_service
from app.services.rag import embedding_service, vector_store
from app.services.voice import voice_service

router = APIRouter(prefix="/system", tags=["System"])

# Track startup time
_start_time = time.time()


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Health check endpoint.
    
    Returns status of all services.
    """
    # Check services
    llm_ok = await llm_service.check_health() if llm_service.is_available else False
    embedding_ok = embedding_service.is_initialized
    vector_ok = vector_store.is_initialized
    voice_ok = voice_service.asr_available or voice_service.tts_available
    
    overall_status = "healthy" if (llm_ok and embedding_ok and vector_ok) else "degraded"
    
    return HealthStatus(
        status=overall_status,
        version=settings.APP_VERSION,
        llm_available=llm_ok,
        embedding_available=embedding_ok,
        vector_store_available=vector_ok,
        voice_available=voice_ok,
        uptime=time.time() - _start_time
    )


@router.get("/info", response_model=SystemInfo)
async def system_info():
    """Get system resource information."""
    # Get resource usage
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(settings.DATA_DIR))
    
    # Get document count
    doc_count = await vector_store.count() if vector_store.is_initialized else 0
    
    # Get loaded models
    models = []
    if embedding_service.is_initialized:
        models.append(f"embedding:{settings.EMBEDDING_MODEL}")
    if llm_service.is_available:
        models.append(f"llm:{settings.OLLAMA_MODEL}")
    if voice_service.asr_available:
        models.append(f"asr:{settings.WHISPER_MODEL}")
    
    return SystemInfo(
        cpu_usage=cpu_usage,
        memory_usage=memory.percent,
        disk_usage=disk.percent,
        active_sessions=0,  # Would need session tracking
        documents_indexed=doc_count,
        models_loaded=models
    )


@router.get("/config")
async def get_config():
    """Get public configuration values."""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "max_file_size_mb": settings.MAX_FILE_SIZE / 1024 / 1024,
        "supported_extensions": settings.all_supported_extensions,
        "llm_model": settings.OLLAMA_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "similarity_top_k": settings.SIMILARITY_TOP_K,
    }


@router.post("/initialize")
async def initialize_services():
    """
    Initialize all services.
    
    Call this on startup to pre-load models.
    """
    logger.info("Initializing all services...")
    
    results = {}
    
    try:
        await embedding_service.initialize()
        results["embedding"] = "ok"
    except Exception as e:
        results["embedding"] = f"error: {e}"
    
    try:
        await vector_store.initialize()
        results["vector_store"] = "ok"
    except Exception as e:
        results["vector_store"] = f"error: {e}"
    
    try:
        await llm_service.initialize()
        results["llm"] = "ok"
    except Exception as e:
        results["llm"] = f"error: {e}"
    
    try:
        await voice_service.initialize()
        results["voice"] = "ok"
    except Exception as e:
        results["voice"] = f"error: {e}"
    
    return {"status": "initialized", "services": results}


@router.post("/clear-index")
async def clear_index():
    """Clear all documents from the vector store."""
    if not vector_store.is_initialized:
        await vector_store.initialize()
    
    await vector_store.clear()
    
    # Also clear document store
    from app.api.routes.documents import documents_store
    documents_store.clear()
    
    return {"status": "cleared"}


@router.get("/debug/logs")
async def get_recent_logs(lines: int = 100):
    """Get recent log entries (debug mode only)."""
    if not settings.DEBUG:
        raise HTTPException(status_code=403, detail="Debug mode not enabled")
    
    log_file = settings.DATA_DIR / "logs" / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    if not log_file.exists():
        return {"logs": []}
    
    with open(log_file, "r") as f:
        all_lines = f.readlines()
        recent = all_lines[-lines:]
    
    return {"logs": recent}
