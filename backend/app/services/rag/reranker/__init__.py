"""
OfflineRAG - Cross-Encoder Re-Ranking Module
=============================================

Local cross-encoder for two-stage RAG re-ranking.
"""

from app.services.rag.reranker.reranker import reranker_service, RerankerService

__all__ = ['reranker_service', 'RerankerService']
