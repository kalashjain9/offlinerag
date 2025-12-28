"""
OfflineRAG - Chat API Routes
============================

REST and WebSocket endpoints for chat functionality.
"""

import asyncio
import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from app.models.schemas import (
    ChatRequest, ChatResponse, ChatSession, ChatMessage,
    StreamChunk, ErrorResponse
)
from app.services.chat import chat_service
from app.services.chat.store import chat_store

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a chat message and get a complete response.
    
    Use this for non-streaming responses.
    """
    try:
        response = await chat_service.chat(request)
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_message(request: ChatRequest):
    """
    Send a chat message and stream the response.
    
    Returns Server-Sent Events (SSE) stream.
    """
    async def generate():
        try:
            async for chunk in chat_service.stream_chat(request):
                data = chunk.model_dump_json()
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            yield f"data: {StreamChunk(type='done').model_dump_json()}\n\n"
        except Exception as e:
            error_chunk = StreamChunk(type="error", error=str(e))
            yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    
    Protocol:
    - Client sends: {"message": "...", "document_ids": [...], "session_id": "..."}
    - Server sends: {"type": "token|source|done|error", "content": "...", ...}
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            request = ChatRequest(
                message=data.get("message", ""),
                session_id=data.get("session_id"),
                document_ids=data.get("document_ids", []),
                stream=True,
                include_sources=data.get("include_sources", False)
            )
            
            # Stream response
            async for chunk in chat_service.stream_chat(request):
                await websocket.send_json(chunk.model_dump())
                
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass


# Session Management Endpoints

@router.post("/sessions", response_model=ChatSession)
async def create_session(title: Optional[str] = None):
    """Create a new chat session."""
    session = chat_service.create_session(title)
    return session


@router.get("/sessions", response_model=list[ChatSession])
async def list_sessions():
    """List all chat sessions."""
    return chat_service.list_sessions()


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    """Get a specific chat session."""
    session = chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if chat_service.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear message history for a session."""
    session = chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_service.clear_session_history(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/greeting")
async def get_greeting():
    """
    Get the initial greeting message for new chats.
    
    This greeting appears instantly on load and does not
    consume LLM tokens or affect RAG context.
    """
    return chat_service.get_greeting_message()


# Chat history sync models
class ChatSyncRequest(BaseModel):
    """Request to sync chat history from frontend."""
    sessions: List[Dict[str, Any]]
    messages: Dict[str, List[Dict[str, Any]]]


class ChatHistoryResponse(BaseModel):
    """Response with full chat history."""
    sessions: List[Dict[str, Any]]
    messages: Dict[str, List[Dict[str, Any]]]


@router.post("/sync")
async def sync_chat_history(request: ChatSyncRequest):
    """
    Sync chat history from frontend to backend.
    
    This allows persisting frontend state to the backend storage.
    """
    try:
        await chat_store.sync_from_frontend(request.sessions, request.messages)
        return {"status": "synced", "sessions_count": len(request.sessions)}
    except Exception as e:
        logger.error(f"Error syncing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history():
    """
    Get all persisted chat history.
    
    Returns sessions and messages for restoring frontend state.
    """
    try:
        sessions = await chat_store.list_sessions()
        messages = {}
        for session in sessions:
            session_id = session.get('id')
            if session_id:
                messages[session_id] = await chat_store.get_messages(session_id)
        
        return ChatHistoryResponse(sessions=sessions, messages=messages)
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
