"""
OfflineRAG - Semantic Cache Module
==================================

Redis-based semantic caching for query optimization.
"""

from app.services.cache.semantic_cache import semantic_cache, SemanticCache

__all__ = ['semantic_cache', 'SemanticCache']
