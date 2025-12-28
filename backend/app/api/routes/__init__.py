"""API routes initialization."""

from fastapi import APIRouter
from app.api.routes import chat, documents, voice, system

# Main API router
api_router = APIRouter(prefix="/api/v1")

# Include all route modules
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(voice.router)
api_router.include_router(system.router)

__all__ = ["api_router"]
