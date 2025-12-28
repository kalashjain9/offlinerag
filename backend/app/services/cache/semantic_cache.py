"""
OfflineRAG - Semantic Cache Service
====================================

Redis-based semantic caching with cosine similarity matching.
Caches query embeddings, retrieval results, and final answers.
"""

import hashlib
import json
import pickle
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger
import numpy as np

from app.core.config import settings


class SemanticCache:
    """
    Semantic cache using Redis for fast query response caching.
    
    Features:
    - Query embedding similarity matching
    - TTL-based expiration
    - Session-aware isolation
    - Graceful fallback when Redis unavailable
    """
    
    def __init__(self):
        self._redis_client = None
        self._embedding_service = None
        self._initialized = False
        self._cache_enabled = True
        self._similarity_threshold = 0.92  # High threshold for cache hits
        self._ttl_seconds = 3600  # 1 hour default TTL
        self._max_cache_entries = 10000
        self._local_cache: Dict[str, Any] = {}  # Fallback in-memory cache
    
    async def initialize(self):
        """Initialize Redis connection and embedding service."""
        if self._initialized:
            return
        
        # Import here to avoid circular imports
        from app.services.rag.embedding import embedding_service
        self._embedding_service = embedding_service
        await self._embedding_service.initialize()
        
        # Try to connect to Redis
        try:
            import redis.asyncio as redis
            self._redis_client = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=getattr(settings, 'REDIS_DB', 0),
                decode_responses=False,  # We'll handle encoding ourselves
            )
            # Test connection
            await self._redis_client.ping()
            logger.info("Redis semantic cache connected")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache: {e}")
            self._redis_client = None
        
        self._initialized = True
        logger.info("Semantic cache initialized")
    
    def _generate_cache_key(self, query: str, session_id: Optional[str] = None) -> str:
        """Generate a unique cache key for a query."""
        key_data = f"{query.lower().strip()}"
        if session_id:
            key_data = f"{session_id}:{key_data}"
        return f"rag_cache:{hashlib.sha256(key_data.encode()).hexdigest()[:32]}"
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    async def get_cached_response(
        self,
        query: str,
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a semantically similar query exists in cache.
        
        Args:
            query: User query
            session_id: Optional session ID for isolation
            document_ids: Document IDs to filter by
            
        Returns:
            Cached response dict if found, None otherwise
        """
        if not self._cache_enabled or not self._initialized:
            return None
        
        try:
            # Generate query embedding
            query_embedding = await self._embedding_service.embed_text(query)
            query_embedding = np.array(query_embedding)
            
            # First, try exact match
            exact_key = self._generate_cache_key(query, session_id)
            cached = await self._get_from_cache(exact_key)
            if cached:
                cached['cache_hit'] = 'exact'
                logger.debug(f"Exact cache hit for query: {query[:50]}...")
                return cached
            
            # Search for semantic matches in recent queries
            similar_entry = await self._find_similar_query(
                query_embedding, 
                session_id,
                document_ids
            )
            
            if similar_entry:
                similar_entry['cache_hit'] = 'semantic'
                logger.debug(f"Semantic cache hit (similarity: {similar_entry.get('similarity', 0):.3f})")
                return similar_entry
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            return None
    
    async def _find_similar_query(
        self,
        query_embedding: np.ndarray,
        session_id: Optional[str],
        document_ids: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Find a semantically similar cached query."""
        try:
            # Get all cached query embeddings
            if self._redis_client:
                keys = await self._redis_client.keys("rag_cache:emb:*")
                for key in keys[:100]:  # Limit search
                    data = await self._redis_client.get(key)
                    if data:
                        entry = pickle.loads(data)
                        cached_embedding = np.array(entry['embedding'])
                        similarity = self._compute_similarity(query_embedding, cached_embedding)
                        
                        if similarity >= self._similarity_threshold:
                            # Check document filter match
                            if document_ids:
                                cached_docs = set(entry.get('document_ids', []))
                                if cached_docs and not cached_docs.intersection(set(document_ids)):
                                    continue
                            
                            # Retrieve full response
                            response_key = entry['response_key']
                            response_data = await self._redis_client.get(response_key)
                            if response_data:
                                response = pickle.loads(response_data)
                                response['similarity'] = similarity
                                return response
            else:
                # Use in-memory cache
                for key, entry in list(self._local_cache.items())[:100]:
                    if not key.startswith("emb:"):
                        continue
                    cached_embedding = np.array(entry['embedding'])
                    similarity = self._compute_similarity(query_embedding, cached_embedding)
                    
                    if similarity >= self._similarity_threshold:
                        response_key = entry['response_key']
                        if response_key in self._local_cache:
                            response = self._local_cache[response_key].copy()
                            response['similarity'] = similarity
                            return response
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
        
        return None
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from cache (Redis or local)."""
        try:
            if self._redis_client:
                data = await self._redis_client.get(key)
                if data:
                    return pickle.loads(data)
            elif key in self._local_cache:
                entry = self._local_cache[key]
                if isinstance(entry, dict) and entry.get('expires_at'):
                    if datetime.utcnow() > entry['expires_at']:
                        del self._local_cache[key]
                        return None
                return entry.get('data') if isinstance(entry, dict) else entry
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None
    
    async def cache_response(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        retrieval_results: Optional[List[Any]] = None
    ):
        """
        Cache a query response with its embedding.
        
        Args:
            query: Original query
            response: Generated response
            sources: Source documents
            session_id: Session ID
            document_ids: Document IDs used
            retrieval_results: Retrieved chunks
        """
        if not self._cache_enabled or not self._initialized:
            return
        
        try:
            # Generate query embedding
            query_embedding = await self._embedding_service.embed_text(query)
            
            # Create cache entry
            response_key = self._generate_cache_key(query, session_id)
            embedding_key = f"rag_cache:emb:{response_key.split(':')[1]}"
            
            response_data = {
                'query': query,
                'response': response,
                'sources': sources,
                'document_ids': document_ids or [],
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            embedding_data = {
                'embedding': query_embedding,
                'response_key': response_key,
                'document_ids': document_ids or [],
            }
            
            if self._redis_client:
                # Store in Redis
                await self._redis_client.setex(
                    response_key,
                    self._ttl_seconds,
                    pickle.dumps(response_data)
                )
                await self._redis_client.setex(
                    embedding_key,
                    self._ttl_seconds,
                    pickle.dumps(embedding_data)
                )
            else:
                # Store in local cache
                expires_at = datetime.utcnow() + timedelta(seconds=self._ttl_seconds)
                self._local_cache[response_key] = {
                    'data': response_data,
                    'expires_at': expires_at
                }
                self._local_cache[f"emb:{response_key}"] = {
                    **embedding_data,
                    'expires_at': expires_at
                }
                
                # Cleanup old entries if needed
                if len(self._local_cache) > self._max_cache_entries:
                    self._cleanup_local_cache()
            
            logger.debug(f"Cached response for query: {query[:50]}...")
            
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")
    
    def _cleanup_local_cache(self):
        """Remove expired and oldest entries from local cache."""
        now = datetime.utcnow()
        
        # Remove expired
        expired = [
            k for k, v in self._local_cache.items()
            if isinstance(v, dict) and v.get('expires_at') and v['expires_at'] < now
        ]
        for k in expired:
            del self._local_cache[k]
        
        # Remove oldest if still too many
        if len(self._local_cache) > self._max_cache_entries * 0.8:
            sorted_keys = sorted(
                [k for k in self._local_cache.keys() if not k.startswith('emb:')],
                key=lambda k: self._local_cache[k].get('timestamp', ''),
            )
            for k in sorted_keys[:len(sorted_keys) // 2]:
                self._local_cache.pop(k, None)
                self._local_cache.pop(f"emb:{k}", None)
    
    async def invalidate_cache(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern."""
        try:
            if self._redis_client:
                if pattern:
                    keys = await self._redis_client.keys(f"rag_cache:*{pattern}*")
                else:
                    keys = await self._redis_client.keys("rag_cache:*")
                if keys:
                    await self._redis_client.delete(*keys)
            else:
                if pattern:
                    to_delete = [k for k in self._local_cache if pattern in k]
                    for k in to_delete:
                        del self._local_cache[k]
                else:
                    self._local_cache.clear()
            
            logger.info(f"Cache invalidated (pattern: {pattern or 'all'})")
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self._redis_client:
                info = await self._redis_client.info('memory')
                keys = await self._redis_client.keys("rag_cache:*")
                return {
                    'type': 'redis',
                    'entries': len(keys),
                    'memory_used': info.get('used_memory_human', 'unknown'),
                    'enabled': self._cache_enabled,
                }
            else:
                return {
                    'type': 'in-memory',
                    'entries': len(self._local_cache),
                    'enabled': self._cache_enabled,
                }
        except Exception as e:
            return {'error': str(e)}
    
    def set_enabled(self, enabled: bool):
        """Enable or disable caching."""
        self._cache_enabled = enabled
        logger.info(f"Semantic cache {'enabled' if enabled else 'disabled'}")
    
    def set_similarity_threshold(self, threshold: float):
        """Set the similarity threshold for cache hits."""
        self._similarity_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Cache similarity threshold set to {self._similarity_threshold}")


# Singleton instance
semantic_cache = SemanticCache()
