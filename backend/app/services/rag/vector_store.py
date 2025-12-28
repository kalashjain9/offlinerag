"""
OfflineRAG - Vector Store Service
=================================

ChromaDB-based vector storage for document embeddings.
Supports persistent storage, incremental updates, and hybrid search.
"""

import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.models.schemas import DocumentChunk, RetrievalResult
from app.services.rag.embedding import embedding_service


class VectorStore:
    """ChromaDB-based vector store for RAG."""
    
    def __init__(self):
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize ChromaDB client and collection."""
        if self._initialized:
            return
        
        logger.info("Initializing vector store...")
        
        try:
            # Create persistent client
            self._client = chromadb.PersistentClient(
                path=str(settings.CHROMA_PERSIST_DIR),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            self._initialized = True
            logger.info(f"Vector store initialized. Documents: {self._collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    @property
    def is_initialized(self) -> bool:
        """Check if store is ready."""
        return self._initialized
    
    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[List[List[float]]] = None
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            Number of chunks added
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return 0
        
        # Generate embeddings if not provided
        if embeddings is None:
            texts = [chunk.content for chunk in chunks]
            embeddings = await embedding_service.embed_texts(texts)
        
        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        # Add to collection
        await asyncio.to_thread(
            self._collection.add,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        return len(chunks)
    
    async def search(
        self,
        query: str,
        top_k: int = None,
        filter_document_ids: Optional[List[str]] = None,
        threshold: float = None
    ) -> List[RetrievalResult]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_document_ids: Filter by specific document IDs
            threshold: Minimum similarity threshold
            
        Returns:
            List of retrieval results
        """
        if not self._initialized:
            await self.initialize()
        
        top_k = top_k or settings.SIMILARITY_TOP_K
        threshold = threshold or settings.SIMILARITY_THRESHOLD
        
        # Generate query embedding
        query_embedding = await embedding_service.embed_text(query)
        
        # Build filter
        where_filter = None
        if filter_document_ids:
            where_filter = {"document_id": {"$in": filter_document_ids}}
        
        # Search
        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        retrieval_results = []
        
        if results and results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                # ChromaDB returns distances, convert to similarity
                distance = results['distances'][0][i] if results['distances'] else 0
                # For cosine distance: similarity = 1 - distance
                similarity = 1 - distance
                
                if similarity >= threshold:
                    retrieval_results.append(RetrievalResult(
                        chunk_id=chunk_id,
                        document_id=results['metadatas'][0][i].get('document_id', ''),
                        content=results['documents'][0][i],
                        score=similarity,
                        metadata=results['metadatas'][0][i]
                    ))
        
        return retrieval_results
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        filter_document_ids: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_document_ids: Filter by specific document IDs
            
        Returns:
            List of retrieval results with combined scores
        """
        top_k = top_k or settings.SIMILARITY_TOP_K
        
        # Semantic search
        semantic_results = await self.search(
            query,
            top_k=top_k * 2,  # Get more for reranking
            filter_document_ids=filter_document_ids,
            threshold=0.1  # Lower threshold for hybrid
        )
        
        # Keyword boost
        query_terms = set(query.lower().split())
        
        for result in semantic_results:
            content_terms = set(result.content.lower().split())
            keyword_overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            
            # Combine scores
            result.score = (
                settings.RAG_SEMANTIC_WEIGHT * result.score +
                settings.RAG_KEYWORD_WEIGHT * keyword_overlap
            )
        
        # Sort by combined score and return top_k
        semantic_results.sort(key=lambda x: x.score, reverse=True)
        
        return semantic_results[:top_k]
    
    async def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Number of chunks deleted
        """
        if not self._initialized:
            await self.initialize()
        
        # Get chunks for this document
        results = await asyncio.to_thread(
            self._collection.get,
            where={"document_id": document_id},
            include=[]
        )
        
        if results and results['ids']:
            await asyncio.to_thread(
                self._collection.delete,
                ids=results['ids']
            )
            logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            return len(results['ids'])
        
        return 0
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        if not self._initialized:
            await self.initialize()
        
        results = await asyncio.to_thread(
            self._collection.get,
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if results and results['ids']:
            for i, chunk_id in enumerate(results['ids']):
                chunks.append({
                    "id": chunk_id,
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })
        
        return chunks
    
    async def count(self) -> int:
        """Get total number of chunks in store."""
        if not self._initialized:
            await self.initialize()
        
        return self._collection.count()
    
    async def clear(self):
        """Clear all data from the store."""
        if not self._initialized:
            await self.initialize()
        
        # Reset collection
        self._client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        self._collection = self._client.create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Vector store cleared")


# Singleton instance
vector_store = VectorStore()
