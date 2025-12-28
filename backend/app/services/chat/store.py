"""
Persistent Chat Store
=====================

Stores chat history to disk for persistence across restarts.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from app.core.config import settings


class ChatStore:
    """Persistent storage for chat sessions and messages."""
    
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._messages: Dict[str, List[Dict[str, Any]]] = {}  # session_id -> messages
        self._lock = asyncio.Lock()
        self._sessions_file = Path(settings.DATA_DIR) / "chat_sessions.json"
        self._messages_dir = Path(settings.DATA_DIR) / "chat_messages"
    
    async def initialize(self) -> None:
        """Initialize chat store and load existing data."""
        self._messages_dir.mkdir(parents=True, exist_ok=True)
        await self._load_sessions()
        logger.info(f"Chat store initialized with {len(self._sessions)} sessions")
    
    async def _load_sessions(self) -> None:
        """Load sessions from disk."""
        if self._sessions_file.exists():
            try:
                data = json.loads(self._sessions_file.read_text(encoding='utf-8'))
                self._sessions = data.get('sessions', {})
                logger.info(f"Loaded {len(self._sessions)} chat sessions")
            except Exception as e:
                logger.error(f"Error loading chat sessions: {e}")
                self._sessions = {}
    
    async def _save_sessions(self) -> None:
        """Save sessions to disk."""
        try:
            data = {'sessions': self._sessions}
            self._sessions_file.write_text(
                json.dumps(data, indent=2, default=str),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Error saving chat sessions: {e}")
    
    async def _load_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Load messages for a session from disk."""
        msg_file = self._messages_dir / f"{session_id}.json"
        if msg_file.exists():
            try:
                data = json.loads(msg_file.read_text(encoding='utf-8'))
                return data.get('messages', [])
            except Exception as e:
                logger.error(f"Error loading messages for {session_id}: {e}")
        return []
    
    async def _save_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save messages for a session to disk."""
        try:
            msg_file = self._messages_dir / f"{session_id}.json"
            data = {'session_id': session_id, 'messages': messages}
            msg_file.write_text(
                json.dumps(data, indent=2, default=str),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Error saving messages for {session_id}: {e}")
    
    async def create_session(self, session_id: str, title: str = "New Chat") -> Dict[str, Any]:
        """Create a new chat session."""
        async with self._lock:
            session = {
                'id': session_id,
                'title': title,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }
            self._sessions[session_id] = session
            self._messages[session_id] = []
            await self._save_sessions()
            await self._save_messages(session_id, [])
            return session
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    async def update_session(self, session_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Update session properties."""
        async with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].update(kwargs)
                self._sessions[session_id]['updated_at'] = datetime.now().isoformat()
                await self._save_sessions()
                return self._sessions[session_id]
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._messages.pop(session_id, None)
                
                # Delete messages file
                msg_file = self._messages_dir / f"{session_id}.json"
                if msg_file.exists():
                    msg_file.unlink()
                
                await self._save_sessions()
                return True
            return False
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        sessions = list(self._sessions.values())
        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.get('updated_at', ''), reverse=True)
        return sessions
    
    async def add_message(self, session_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Add a message to a session."""
        async with self._lock:
            # Create session if it doesn't exist
            if session_id not in self._sessions:
                await self.create_session(session_id)
            
            # Load messages if not in memory
            if session_id not in self._messages:
                self._messages[session_id] = await self._load_messages(session_id)
            
            # Add message
            message['timestamp'] = message.get('timestamp', datetime.now().isoformat())
            self._messages[session_id].append(message)
            
            # Update session timestamp
            if session_id in self._sessions:
                self._sessions[session_id]['updated_at'] = datetime.now().isoformat()
                await self._save_sessions()
            
            await self._save_messages(session_id, self._messages[session_id])
            return message
    
    async def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session."""
        if session_id not in self._messages:
            self._messages[session_id] = await self._load_messages(session_id)
        return self._messages.get(session_id, [])
    
    async def clear_messages(self, session_id: str) -> bool:
        """Clear all messages for a session."""
        async with self._lock:
            self._messages[session_id] = []
            await self._save_messages(session_id, [])
            return True
    
    async def sync_from_frontend(self, sessions: List[Dict[str, Any]], messages: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Sync chat data from frontend.
        Used to persist frontend state to backend.
        """
        async with self._lock:
            # Update sessions
            for session in sessions:
                session_id = session.get('id')
                if session_id:
                    self._sessions[session_id] = session
            
            # Update messages
            for session_id, msgs in messages.items():
                self._messages[session_id] = msgs
                await self._save_messages(session_id, msgs)
            
            await self._save_sessions()
            logger.info(f"Synced {len(sessions)} sessions and {sum(len(m) for m in messages.values())} messages from frontend")


# Global instance
chat_store = ChatStore()
