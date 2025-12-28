"""
OfflineRAG - Document API Routes
================================

Endpoints for document upload, processing, and management.
"""

import os
import uuid
import asyncio
import aiofiles
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.config import settings
from app.models.schemas import (
    Document, DocumentUploadResponse, DocumentStatusResponse,
    DocumentStatus, DocumentType
)
from app.services.documents import document_processor
from app.services.documents.store import document_store
from app.services.rag import vector_store, embedding_service

router = APIRouter(prefix="/documents", tags=["Documents"])

# Track in-progress processing tasks
processing_tasks: dict[str, asyncio.Task] = {}


async def process_document_task(
    document_id: str,
    file_path: Path,
    filename: str,
    permanent_path: Path
):
    """Background task for document processing."""
    try:
        # Update status
        await document_store.update(document_id, status=DocumentStatus.PROCESSING)
        
        # Process the file
        async def progress_callback(progress: float, message: str):
            logger.debug(f"Document {document_id}: {progress*100:.0f}% - {message}")
        
        raw_text, chunks, metadata = await document_processor.process_file(
            file_path=file_path,
            filename=filename,
            document_id=document_id,
            progress_callback=progress_callback
        )
        
        # Update document
        await document_store.update(document_id, status=DocumentStatus.INDEXING, metadata=metadata)
        
        # Generate embeddings and store
        if chunks:
            texts = [chunk.content for chunk in chunks]
            embeddings = await embedding_service.embed_texts(texts, show_progress=True)
            await vector_store.add_chunks(chunks, embeddings)
        
        # Copy file to permanent location
        try:
            shutil.copy2(file_path, permanent_path)
        except Exception as e:
            logger.warning(f"Failed to copy file to permanent storage: {e}")
        
        # Mark as ready
        await document_store.update(document_id, status=DocumentStatus.READY)
        logger.info(f"Document {document_id} processed: {len(chunks)} chunks")
        
    except asyncio.CancelledError:
        await document_store.update(document_id, status=DocumentStatus.CANCELLED)
        logger.info(f"Document processing cancelled: {document_id}")
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        await document_store.update(document_id, status=DocumentStatus.ERROR, error_message=str(e))
    
    finally:
        # Cleanup temp file
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass
        
        # Remove from processing tasks
        processing_tasks.pop(document_id, None)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document for processing.
    
    Supports: PDF, Word, Excel, CSV, Images, Audio, Video, Text files
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")
    
    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.all_supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {ext}"
        )
    
    # Check file size (read content length if available)
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
        )
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Detect file type
    file_type = document_processor.detect_type(file.filename)
    
    # Save to temp file
    temp_path = settings.TEMP_DIR / f"{document_id}{ext}"
    permanent_path = document_store.get_upload_path(document_id, ext)
    
    try:
        async with aiofiles.open(temp_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        
        file_size = len(content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Create document record
    from app.models.schemas import DocumentMetadata
    from datetime import datetime
    
    document = Document(
        id=document_id,
        filename=file.filename,
        file_type=file_type,
        file_size=file_size,
        status=DocumentStatus.UPLOADING,
        metadata=DocumentMetadata(
            filename=file.filename,
            file_type=file_type,
            file_size=file_size
        ),
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Add to persistent store
    await document_store.add(document)
    
    # Start background processing
    task = asyncio.create_task(
        process_document_task(document_id, temp_path, file.filename, permanent_path)
    )
    processing_tasks[document_id] = task
    
    return DocumentUploadResponse(
        id=document_id,
        filename=file.filename,
        status=DocumentStatus.UPLOADING,
        message="Document uploaded, processing started"
    )


@router.post("/upload/multiple")
async def upload_multiple_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload multiple documents at once."""
    if len(files) > settings.MAX_CONCURRENT_UPLOADS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.MAX_CONCURRENT_UPLOADS} files at once"
        )
    
    results = []
    for file in files:
        try:
            result = await upload_document(background_tasks, file)
            results.append(result)
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "error": e.detail
            })
    
    return {"documents": results}


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str):
    """Get document details."""
    document = await document_store.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Get document processing status."""
    document = await document_store.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Calculate progress based on status
    progress_map = {
        DocumentStatus.UPLOADING: 0.1,
        DocumentStatus.PROCESSING: 0.5,
        DocumentStatus.INDEXING: 0.8,
        DocumentStatus.READY: 1.0,
        DocumentStatus.ERROR: 0.0,
        DocumentStatus.CANCELLED: 0.0,
    }
    
    return DocumentStatusResponse(
        id=document_id,
        filename=document.filename,
        status=document.status,
        progress=progress_map.get(document.status, 0),
        error_message=document.error_message
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its indexed data."""
    document = await document_store.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Cancel processing if in progress
    if document_id in processing_tasks:
        processing_tasks[document_id].cancel()
        document_processor.request_cancel(document_id)
    
    # Remove from vector store
    await vector_store.delete_document(document_id)
    
    # Remove from persistent store (also deletes file)
    await document_store.delete(document_id)
    
    return {"status": "deleted", "document_id": document_id}


@router.post("/{document_id}/cancel")
async def cancel_processing(document_id: str):
    """Cancel document processing."""
    if document_id not in processing_tasks:
        raise HTTPException(status_code=400, detail="Document not being processed")
    
    # Request cancellation
    document_processor.request_cancel(document_id)
    processing_tasks[document_id].cancel()
    
    await document_store.update(document_id, status=DocumentStatus.CANCELLED)
    
    return {"status": "cancelled", "document_id": document_id}


@router.get("/")
async def list_documents(
    status: Optional[DocumentStatus] = Query(None),
    limit: int = Query(50, ge=1, le=100)
):
    """List all documents."""
    docs = await document_store.list_all()
    
    if status:
        docs = [d for d in docs if d.status == status]
    
    # Sort by created_at descending
    docs.sort(key=lambda d: d.created_at if d.created_at else datetime.min, reverse=True)
    
    return {"documents": docs[:limit], "total": len(docs)}


@router.get("/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """Get all chunks for a document."""
    document = await document_store.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = await vector_store.get_document_chunks(document_id)
    
    return {"document_id": document_id, "chunks": chunks}
