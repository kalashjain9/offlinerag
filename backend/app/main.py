"""
OfflineRAG - Main Application
=============================

FastAPI application entry point with all middleware and routes.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.config import settings
from app.api import api_router
from app.services import (
    document_processor,
    embedding_service,
    vector_store,
    llm_service,
    voice_service,
    chat_service
)
from app.services.documents.store import document_store
from app.services.chat.store import chat_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize services in parallel where possible
    try:
        # Initialize document store (persistent storage)
        await document_store.initialize()
        
        # Initialize chat store (persistent chat history)
        await chat_store.initialize()
        
        # Core services (required)
        await document_processor.initialize()
        await embedding_service.initialize()
        await vector_store.initialize()
        
        # LLM service
        try:
            await llm_service.initialize()
        except Exception as e:
            logger.warning(f"LLM initialization deferred: {e}")
        
        # Voice services (optional)
        try:
            await voice_service.initialize()
        except Exception as e:
            logger.warning(f"Voice services not available: {e}")
        
        # Chat service
        await chat_service.initialize()
        
        logger.info("All services initialized")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    # Cleanup tasks here if needed


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Fully offline RAG chatbot with multi-modal support",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else None
        }
    )


# Include API routes
app.include_router(api_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else None,
        "api": "/api/v1"
    }


# Simple health check at root level
@app.get("/health")
async def health():
    """Quick health check."""
    return {"status": "ok"}
