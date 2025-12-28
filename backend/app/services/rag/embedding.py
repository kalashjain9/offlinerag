"""
OfflineRAG - Embedding Service
==============================

Handles text embedding using local models (sentence-transformers).
"""

import asyncio
from typing import List, Optional
from loguru import logger
import numpy as np

from app.core.config import settings


class EmbeddingService:
    """Local embedding service using sentence-transformers."""
    
    def __init__(self):
        self._model = None
        self._initialized = False
        self._dimension = 384  # Default for MiniLM
    
    async def initialize(self):
        """Load the embedding model."""
        if self._initialized:
            return
        
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model in thread pool to not block
            self._model = await asyncio.to_thread(
                SentenceTransformer,
                settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE
            )
            
            # Get embedding dimension
            test_embedding = self._model.encode(["test"])[0]
            self._dimension = len(test_embedding)
            
            self._initialized = True
            logger.info(f"Embedding model loaded. Dimension: {self._dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    @property
    def is_initialized(self) -> bool:
        """Check if model is loaded."""
        return self._initialized
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not self._initialized:
            await self.initialize()
        
        embedding = await asyncio.to_thread(
            self._model.encode,
            text,
            normalize_embeddings=True
        )
        
        return embedding.tolist()
    
    async def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()
        
        if not texts:
            return []
        
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        embeddings = await asyncio.to_thread(
            self._model.encode,
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )
        
        return embeddings.tolist()
    
    async def compute_similarity(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embeddings
            
        Returns:
            List of similarity scores
        """
        query = np.array(query_embedding)
        docs = np.array(document_embeddings)
        
        # Cosine similarity (embeddings are normalized)
        similarities = np.dot(docs, query)
        
        return similarities.tolist()


# Singleton instance
embedding_service = EmbeddingService()
