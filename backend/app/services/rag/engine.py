"""
OfflineRAG - RAG Engine
=======================

Core retrieval-augmented generation engine.
Orchestrates retrieval, cross-encoder re-ranking, caching, and generation.
"""

import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from loguru import logger

from app.core.config import settings
from app.models.schemas import (
    RetrievalResult, RAGContext, ChatMessage, MessageRole
)
from app.services.rag.vector_store import vector_store
from app.services.rag.embedding import embedding_service
from app.services.rag.reranker import reranker_service
from app.services.cache import semantic_cache


class RAGEngine:
    """
    RAG Engine for retrieval-augmented generation.
    
    Handles:
    - Semantic caching for fast responses
    - Context retrieval from vector store
    - Cross-encoder re-ranking for improved precision
    - Context formatting and optimization
    - Source tracking for citations
    """
    
    def __init__(self):
        self._initialized = False
        self._use_reranking = True
        self._use_caching = True
    
    async def initialize(self):
        """Initialize the RAG engine and dependencies."""
        if self._initialized:
            return
        
        logger.info("Initializing RAG engine...")
        
        # Initialize dependencies
        await embedding_service.initialize()
        await vector_store.initialize()
        
        # Initialize re-ranker (optional, graceful failure)
        try:
            await reranker_service.initialize()
            logger.info("Cross-encoder re-ranker initialized")
        except Exception as e:
            logger.warning(f"Cross-encoder not available: {e}")
            self._use_reranking = False
        
        # Initialize semantic cache
        try:
            await semantic_cache.initialize()
            logger.info("Semantic cache initialized")
        except Exception as e:
            logger.warning(f"Semantic cache not available: {e}")
            self._use_caching = False
        
        self._initialized = True
        logger.info("RAG engine initialized")
    
    async def retrieve_context(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = None,
        use_hybrid: bool = None,
        use_reranking: bool = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant context for a query with optional re-ranking.
        
        Two-stage retrieval:
        1. Fast vector similarity search
        2. Cross-encoder re-ranking for precision
        
        Args:
            query: User query
            document_ids: Filter by specific documents
            top_k: Number of chunks to retrieve
            use_hybrid: Use hybrid search
            use_reranking: Apply cross-encoder re-ranking
            
        Returns:
            List of retrieval results
        """
        if not self._initialized:
            await self.initialize()
        
        top_k = top_k or settings.SIMILARITY_TOP_K
        use_hybrid = use_hybrid if use_hybrid is not None else settings.RAG_HYBRID_SEARCH
        use_reranking = use_reranking if use_reranking is not None else self._use_reranking
        
        # Stage 1: Fast vector retrieval
        # Retrieve more candidates for re-ranking
        retrieval_k = top_k * 3 if use_reranking and reranker_service.is_available() else top_k
        
        if use_hybrid:
            results = await vector_store.hybrid_search(
                query=query,
                top_k=retrieval_k,
                filter_document_ids=document_ids
            )
        else:
            results = await vector_store.search(
                query=query,
                top_k=retrieval_k,
                filter_document_ids=document_ids
            )
        
        # Stage 2: Cross-encoder re-ranking
        if use_reranking and reranker_service.is_available() and len(results) > 1:
            results = await reranker_service.rerank(
                query=query,
                results=results,
                top_n=top_k
            )
        
        logger.debug(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
        
        return results
    
    def format_context(
        self,
        results: List[RetrievalResult],
        max_tokens: int = None
    ) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            results: Retrieved chunks
            max_tokens: Maximum context tokens
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        max_tokens = max_tokens or settings.RAG_CONTEXT_WINDOW
        
        # Estimate tokens (rough: 4 chars per token)
        max_chars = max_tokens * 4
        
        context_parts = []
        current_chars = 0
        
        for i, result in enumerate(results):
            # Format chunk with source info
            source_info = f"[Source {i+1}: {result.metadata.get('filename', 'Unknown')}]"
            chunk_text = f"{source_info}\n{result.content}"
            
            # Check if adding this chunk exceeds limit
            if current_chars + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            current_chars += len(chunk_text)
        
        return "\n\n---\n\n".join(context_parts)
    
    async def build_rag_context(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> RAGContext:
        """
        Build complete RAG context for generation.
        
        Args:
            query: User query
            document_ids: Filter by documents
            conversation_history: Previous messages
            
        Returns:
            RAGContext ready for LLM
        """
        # Retrieve relevant chunks
        results = await self.retrieve_context(
            query=query,
            document_ids=document_ids
        )
        
        # Format context
        formatted_context = self.format_context(results)
        
        # Estimate tokens
        total_chars = len(formatted_context) + len(query)
        if conversation_history:
            total_chars += sum(len(m.content) for m in conversation_history)
        estimated_tokens = total_chars // 4
        
        return RAGContext(
            query=query,
            retrieved_chunks=results,
            formatted_context=formatted_context,
            total_tokens=estimated_tokens
        )
    
    def build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        system_prompt: str = None
    ) -> List[Dict[str, str]]:
        """
        Build the prompt messages for LLM.
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous messages
            system_prompt: Custom system prompt
            
        Returns:
            List of message dicts for LLM
        """
        messages = []
        
        # System message with context
        system = system_prompt or settings.RAG_SYSTEM_PROMPT
        
        if context:
            system += f"\n\n## Retrieved Context:\n{context}"
        else:
            system += "\n\nNo relevant context was found in the knowledge base."
        
        messages.append({"role": "system", "content": system})
        
        # Add conversation history (limited)
        if conversation_history:
            history = conversation_history[-settings.MAX_CONVERSATION_HISTORY:]
            for msg in history:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def extract_sources(
        self,
        results: List[RetrievalResult]
    ) -> List[Dict[str, Any]]:
        """
        Extract source information for citations.
        
        Args:
            results: Retrieved chunks
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen_docs = set()
        
        for result in results:
            doc_id = result.document_id
            
            if doc_id not in seen_docs:
                sources.append({
                    "document_id": doc_id,
                    "filename": result.metadata.get("filename", "Unknown"),
                    "chunk_id": result.chunk_id,
                    "score": round(result.score, 3),
                    "preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
                })
                seen_docs.add(doc_id)
        
        return sources[:settings.RAG_MAX_SOURCES]
    
    async def check_cache(
        self,
        query: str,
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check semantic cache for a similar query.
        
        Returns cached response if found with high similarity.
        """
        if not self._use_caching:
            return None
        
        return await semantic_cache.get_cached_response(
            query=query,
            session_id=session_id,
            document_ids=document_ids
        )
    
    async def cache_response(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        retrieval_results: Optional[List[RetrievalResult]] = None
    ):
        """Cache a query response for future semantic matching."""
        if not self._use_caching:
            return
        
        await semantic_cache.cache_response(
            query=query,
            response=response,
            sources=sources,
            session_id=session_id,
            document_ids=document_ids,
            retrieval_results=retrieval_results
        )
    
    def set_reranking_enabled(self, enabled: bool):
        """Enable or disable cross-encoder re-ranking."""
        self._use_reranking = enabled
        logger.info(f"Re-ranking {'enabled' if enabled else 'disabled'}")
    
    def set_caching_enabled(self, enabled: bool):
        """Enable or disable semantic caching."""
        self._use_caching = enabled
        semantic_cache.set_enabled(enabled)


# Singleton instance
rag_engine = RAGEngine()
