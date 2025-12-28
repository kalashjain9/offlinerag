"""
OfflineRAG - Chat Service
=========================

Manages chat sessions, message history, and orchestrates the RAG pipeline.
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator
from loguru import logger
import uuid

from app.core.config import settings
from app.models.schemas import (
    ChatMessage, ChatSession, ChatRequest, ChatResponse,
    MessageRole, StreamChunk
)
from app.services.rag import rag_engine
from app.services.llm import llm_service


class ChatService:
    """
    Chat service for managing conversations and generating responses.
    """
    
    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize chat service and dependencies."""
        if self._initialized:
            return
        
        logger.info("Initializing chat service...")
        
        # Initialize dependencies
        await rag_engine.initialize()
        await llm_service.initialize()
        
        self._initialized = True
        logger.info("Chat service initialized")
    
    def create_session(self, title: Optional[str] = None) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(
            title=title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        self._sessions[session.id] = session
        logger.debug(f"Created session: {session.id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get an existing session."""
        return self._sessions.get(session_id)
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Get existing session or create new one."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session()
    
    def list_sessions(self) -> List[ChatSession]:
        """List all sessions."""
        return list(self._sessions.values())
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Deleted session: {session_id}")
            return True
        return False
    
    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        document_ids: List[str] = None,
        sources: List[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Add a message to a session."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        message = ChatMessage(
            role=role,
            content=content,
            document_ids=document_ids or [],
            sources=sources or []
        )
        
        session.messages.append(message)
        session.updated_at = datetime.utcnow()
        
        return message
    
    async def chat(
        self,
        request: ChatRequest
    ) -> ChatResponse:
        """
        Process a chat message and generate response (non-streaming).
        
        Args:
            request: Chat request with message and options
            
        Returns:
            ChatResponse with generated message
        """
        if not self._initialized:
            await self.initialize()
        
        # Get or create session
        session = self.get_or_create_session(request.session_id)
        
        # Add user message
        user_message = self.add_message(
            session.id,
            MessageRole.USER,
            request.message,
            document_ids=request.document_ids
        )
        
        # Build RAG context
        rag_context = await rag_engine.build_rag_context(
            query=request.message,
            document_ids=request.document_ids if request.document_ids else None,
            conversation_history=session.messages[:-1]  # Exclude current message
        )
        
        # Build prompt
        messages = rag_engine.build_prompt(
            query=request.message,
            context=rag_context.formatted_context,
            conversation_history=session.messages[:-1]
        )
        
        # Generate response
        response_text = await llm_service.generate(messages)
        
        # Extract sources
        sources = rag_engine.extract_sources(rag_context.retrieved_chunks)
        
        # Add assistant message
        assistant_message = self.add_message(
            session.id,
            MessageRole.ASSISTANT,
            response_text,
            sources=sources if request.include_sources else []
        )
        
        return ChatResponse(
            message=assistant_message,
            session_id=session.id,
            sources=sources if request.include_sources else []
        )
    
    async def stream_chat(
        self,
        request: ChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Process a chat message and stream the response.
        
        Args:
            request: Chat request with message and options
            
        Yields:
            StreamChunk objects with tokens, sources, or completion status
        """
        if not self._initialized:
            await self.initialize()
        
        # Get or create session
        session = self.get_or_create_session(request.session_id)
        
        # Add user message
        user_message = self.add_message(
            session.id,
            MessageRole.USER,
            request.message,
            document_ids=request.document_ids
        )
        
        # Generate request ID for cancellation
        request_id = str(uuid.uuid4())
        
        try:
            # Build RAG context
            rag_context = await rag_engine.build_rag_context(
                query=request.message,
                document_ids=request.document_ids if request.document_ids else None,
                conversation_history=session.messages[:-1]
            )
            
            # Extract sources early
            sources = rag_engine.extract_sources(rag_context.retrieved_chunks)
            
            # Send sources first if requested
            if request.include_sources and sources:
                yield StreamChunk(type="source", sources=sources)
            
            # Build prompt
            messages = rag_engine.build_prompt(
                query=request.message,
                context=rag_context.formatted_context,
                conversation_history=session.messages[:-1]
            )
            
            # Stream response
            full_response = ""
            async for token in llm_service.stream_generate(
                messages,
                request_id=request_id
            ):
                full_response += token
                yield StreamChunk(type="token", content=token)
            
            # Add assistant message to history
            self.add_message(
                session.id,
                MessageRole.ASSISTANT,
                full_response,
                sources=sources if request.include_sources else []
            )
            
            # Send completion
            yield StreamChunk(type="done")
            
        except asyncio.CancelledError:
            logger.info(f"Chat stream cancelled: {request_id}")
            yield StreamChunk(type="done")
            raise
        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield StreamChunk(type="error", error=str(e))
    
    def cancel_generation(self, session_id: str):
        """Cancel ongoing generation for a session."""
        # This would need more sophisticated tracking in production
        # For now, we rely on the request_id based cancellation
        pass
    
    def clear_session_history(self, session_id: str):
        """Clear message history for a session."""
        session = self._sessions.get(session_id)
        if session:
            session.messages = []
            session.updated_at = datetime.utcnow()
            logger.debug(f"Cleared history for session: {session_id}")
    
    def get_greeting_message(self) -> Dict[str, Any]:
        """
        Get the initial greeting message for new sessions.
        
        This greeting:
        - Appears instantly on load
        - Does not consume tokens
        - Does not affect RAG context
        - Feels natural, not scripted
        """
        greetings = [
            "Hello! I'm your AI assistant. How may I help you today?",
            "Hi there! I'm ready to help. What would you like to know?",
            "Welcome! I'm here to assist you. What can I help you with?",
            "Hello! I have access to your documents. Feel free to ask me anything!",
        ]
        import random
        
        return {
            "role": "assistant",
            "content": greetings[random.randint(0, len(greetings) - 1)],
            "is_greeting": True,
            "sources": []
        }


# Singleton instance
chat_service = ChatService()
