"""
OfflineRAG - Cross-Encoder Re-Ranker Service
=============================================

Two-stage RAG re-ranking using local cross-encoder models.
Runs after initial vector retrieval to improve precision.
"""

import asyncio
from typing import List, Optional, Tuple
from loguru import logger
import torch

from app.core.config import settings
from app.models.schemas import RetrievalResult


class RerankerService:
    """
    Cross-encoder re-ranking service.
    
    Features:
    - Local cross-encoder model
    - Joint query-document scoring
    - Batch processing
    - Memory-efficient inference
    """
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._max_length = 512
        self._batch_size = 16
        self._device = "cpu"
    
    async def initialize(self):
        """Initialize the cross-encoder model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
            
            logger.info(f"Loading cross-encoder model: {self._model_name}")
            
            # Load cross-encoder model
            self._model = await asyncio.to_thread(
                CrossEncoder,
                self._model_name,
                max_length=self._max_length,
                device=self._device
            )
            
            self._initialized = True
            logger.info(f"Cross-encoder initialized on device: {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            self._initialized = False
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_n: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Re-rank retrieval results using cross-encoder.
        
        Args:
            query: User query
            results: Initial retrieval results from vector search
            top_n: Number of top results to return (default: half of input)
            
        Returns:
            Re-ranked results sorted by cross-encoder score
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._model or not results:
            return results
        
        if top_n is None:
            top_n = max(3, len(results) // 2)
        
        try:
            # Create query-document pairs
            pairs = [(query, result.content) for result in results]
            
            # Score pairs with cross-encoder (in batches)
            scores = await self._score_pairs(pairs)
            
            # Combine results with scores
            scored_results = list(zip(results, scores))
            
            # Sort by cross-encoder score (higher is better)
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Update scores and return top N
            reranked_results = []
            for result, score in scored_results[:top_n]:
                # Create new result with updated score
                reranked_result = RetrievalResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=float(score),  # Use cross-encoder score
                    metadata={
                        **result.metadata,
                        'original_score': result.score,
                        'reranked': True
                    }
                )
                reranked_results.append(reranked_result)
            
            logger.debug(
                f"Re-ranked {len(results)} results to top {len(reranked_results)} "
                f"(scores: {[f'{r.score:.3f}' for r in reranked_results[:3]]})"
            )
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            # Return original results on failure
            return results[:top_n] if top_n else results
    
    async def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score query-document pairs in batches."""
        all_scores = []
        
        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i:i + self._batch_size]
            batch_scores = await asyncio.to_thread(
                self._model.predict,
                batch,
                show_progress_bar=False
            )
            all_scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else list(batch_scores))
        
        return all_scores
    
    def is_available(self) -> bool:
        """Check if re-ranker is available."""
        return self._initialized and self._model is not None


# Singleton instance
reranker_service = RerankerService()
