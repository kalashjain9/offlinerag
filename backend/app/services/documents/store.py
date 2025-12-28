"""
OfflineRAG - Persistent Document Store
======================================

Stores document metadata and manages uploaded files persistently.
"""

import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.models.schemas import Document, DocumentStatus, DocumentMetadata


def serialize_value(obj: Any) -> Any:
    """Serialize values for JSON, handling datetime and other types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, 'value'):  # Enum
        return obj.value
    if hasattr(obj, 'model_dump'):  # Pydantic model
        return {k: serialize_value(v) for k, v in obj.model_dump().items()}
    if isinstance(obj, dict):
        return {k: serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_value(v) for v in obj]
    return obj


class DocumentStore:
    """
    Persistent document store using JSON file storage.
    
    Features:
    - Saves document metadata to JSON file
    - Manages uploaded files in a permanent uploads directory
    - Auto-loads on startup
    - Thread-safe operations
    """
    
    def __init__(self):
        self._documents: Dict[str, Document] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        
        # Storage paths
        self._store_path = settings.DATA_DIR / "documents.json"
        self._uploads_dir = settings.DATA_DIR / "uploads"
    
    async def initialize(self):
        """Initialize the document store."""
        if self._initialized:
            return
        
        # Create uploads directory
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing documents
        await self._load()
        
        self._initialized = True
        logger.info(f"Document store initialized with {len(self._documents)} documents")
    
    async def _load(self):
        """Load documents from JSON file."""
        if not self._store_path.exists():
            return
        
        try:
            async with aiofiles.open(self._store_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                
                for doc_data in data.get('documents', []):
                    try:
                        # Convert to Document object
                        doc = Document(
                            id=doc_data['id'],
                            filename=doc_data['filename'],
                            file_type=doc_data['file_type'],
                            file_size=doc_data.get('file_size', 0),
                            status=DocumentStatus(doc_data.get('status', 'ready')),
                            error_message=doc_data.get('error_message'),
                            metadata=DocumentMetadata(**doc_data.get('metadata', {})) if doc_data.get('metadata') else None,
                            chunks=doc_data.get('chunks', []),
                            created_at=datetime.fromisoformat(doc_data['created_at']) if doc_data.get('created_at') else None,
                            updated_at=datetime.fromisoformat(doc_data['updated_at']) if doc_data.get('updated_at') else None,
                        )
                        self._documents[doc.id] = doc
                    except Exception as e:
                        logger.warning(f"Failed to load document {doc_data.get('id')}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to load document store: {e}")
    
    async def _save(self):
        """Save documents to JSON file."""
        try:
            data = {
                'documents': [
                    {
                        'id': doc.id,
                        'filename': doc.filename,
                        'file_type': doc.file_type.value if hasattr(doc.file_type, 'value') else str(doc.file_type),
                        'file_size': doc.file_size,
                        'status': doc.status.value if hasattr(doc.status, 'value') else str(doc.status),
                        'error_message': doc.error_message,
                        'metadata': serialize_value(doc.metadata.model_dump()) if doc.metadata else None,
                        'chunks': [],  # Don't save chunks to reduce file size
                        'created_at': doc.created_at.isoformat() if doc.created_at else None,
                        'updated_at': doc.updated_at.isoformat() if doc.updated_at else None,
                    }
                    for doc in self._documents.values()
                ],
                'updated_at': datetime.now().isoformat()
            }
            
            async with aiofiles.open(self._store_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, default=str))
                
        except Exception as e:
            logger.error(f"Failed to save document store: {e}")
    
    def get_upload_path(self, document_id: str, extension: str) -> Path:
        """Get the permanent path for an uploaded file."""
        return self._uploads_dir / f"{document_id}{extension}"
    
    async def add(self, document: Document) -> Document:
        """Add a document to the store."""
        async with self._lock:
            self._documents[document.id] = document
            await self._save()
        return document
    
    async def get(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(document_id)
    
    async def update(self, document_id: str, **updates) -> Optional[Document]:
        """Update a document."""
        async with self._lock:
            if document_id not in self._documents:
                return None
            
            doc = self._documents[document_id]
            for key, value in updates.items():
                if hasattr(doc, key):
                    setattr(doc, key, value)
            doc.updated_at = datetime.now()
            
            await self._save()
            return doc
    
    async def delete(self, document_id: str) -> bool:
        """Delete a document and its file."""
        async with self._lock:
            if document_id not in self._documents:
                return False
            
            doc = self._documents.pop(document_id)
            
            # Delete the uploaded file
            for ext in ['.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls', 
                       '.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp3', '.wav', '.mp4']:
                file_path = self._uploads_dir / f"{document_id}{ext}"
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete file {file_path}: {e}")
            
            await self._save()
            return True
    
    async def list_all(self) -> List[Document]:
        """List all documents."""
        return list(self._documents.values())
    
    async def list_ready(self) -> List[Document]:
        """List all ready documents."""
        return [doc for doc in self._documents.values() if doc.status == DocumentStatus.READY]
    
    def __contains__(self, document_id: str) -> bool:
        """Check if a document exists."""
        return document_id in self._documents
    
    def __len__(self) -> int:
        """Get the number of documents."""
        return len(self._documents)


# Singleton instance
document_store = DocumentStore()
