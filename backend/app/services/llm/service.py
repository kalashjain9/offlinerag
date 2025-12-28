"""
OfflineRAG - LLM Service
========================

Local LLM integration using Ollama or LlamaCpp.
Supports streaming, cancellation, and multiple models.
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from abc import ABC, abstractmethod
from loguru import logger
import httpx

from app.core.config import settings


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def initialize(self):
        """Initialize the provider."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate a complete response."""
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate response tokens."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local inference."""
    
    def __init__(self):
        self._base_url = settings.OLLAMA_HOST
        self._model = settings.OLLAMA_MODEL
        self._timeout = settings.OLLAMA_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self):
        """Initialize the Ollama client."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout, connect=10.0)
        )
        
        # Check if model is available
        if await self.is_available():
            logger.info(f"Ollama provider initialized with model: {self._model}")
        else:
            logger.warning(f"Ollama model {self._model} not available")
    
    async def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                return self._model.split(":")[0] in model_names
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
        return False
    
    async def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
        except Exception:
            pass
        return []
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """Generate a complete response."""
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": settings.LLM_TOP_P,
                "repeat_penalty": settings.LLM_REPEAT_PENALTY,
            }
        }
        
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get("message", {}).get("content", "")
    
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate response tokens."""
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": settings.LLM_TOP_P,
                "repeat_penalty": settings.LLM_REPEAT_PENALTY,
            }
        }
        
        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        import json
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()


class LlamaCppProvider(BaseLLMProvider):
    """LlamaCpp provider for direct model loading."""
    
    def __init__(self):
        self._model = None
        self._model_path = settings.LLAMACPP_MODEL_PATH
    
    async def initialize(self):
        """Load the model."""
        if not self._model_path:
            logger.warning("LlamaCpp model path not configured")
            return
        
        try:
            from llama_cpp import Llama
            
            self._model = await asyncio.to_thread(
                Llama,
                model_path=self._model_path,
                n_ctx=settings.LLAMACPP_N_CTX,
                n_gpu_layers=settings.LLAMACPP_N_GPU_LAYERS,
                verbose=False
            )
            logger.info(f"LlamaCpp model loaded: {self._model_path}")
        except Exception as e:
            logger.error(f"Failed to load LlamaCpp model: {e}")
    
    async def is_available(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """Generate response."""
        if not self._model:
            raise RuntimeError("Model not loaded")
        
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        # Format messages into prompt
        prompt = self._format_prompt(messages)
        
        result = await asyncio.to_thread(
            self._model,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=settings.LLM_TOP_P,
            repeat_penalty=settings.LLM_REPEAT_PENALTY,
        )
        
        return result["choices"][0]["text"]
    
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate response."""
        if not self._model:
            raise RuntimeError("Model not loaded")
        
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        prompt = self._format_prompt(messages)
        
        # Run in thread with streaming
        for output in self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        ):
            text = output["choices"][0]["text"]
            if text:
                yield text
                await asyncio.sleep(0)  # Yield control
    
    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single prompt."""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        prompt += "<|assistant|>\n"
        return prompt


class LLMService:
    """
    Main LLM service that manages providers and handles generation.
    """
    
    def __init__(self):
        self._provider: Optional[BaseLLMProvider] = None
        self._initialized = False
        self._cancel_flags: Dict[str, bool] = {}
    
    async def initialize(self):
        """Initialize the LLM service with configured provider."""
        if self._initialized:
            return
        
        logger.info(f"Initializing LLM service with provider: {settings.LLM_PROVIDER}")
        
        if settings.LLM_PROVIDER == "ollama":
            self._provider = OllamaProvider()
        elif settings.LLM_PROVIDER == "llamacpp":
            self._provider = LlamaCppProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
        
        await self._provider.initialize()
        self._initialized = True
        
        logger.info("LLM service initialized")
    
    @property
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self._initialized and self._provider is not None
    
    def request_cancel(self, request_id: str):
        """Request cancellation of a generation."""
        self._cancel_flags[request_id] = True
    
    def is_cancelled(self, request_id: str) -> bool:
        """Check if generation is cancelled."""
        return self._cancel_flags.get(request_id, False)
    
    def clear_cancel(self, request_id: str):
        """Clear cancellation flag."""
        self._cancel_flags.pop(request_id, None)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate a complete response."""
        if not self._initialized:
            await self.initialize()
        
        return await self._provider.generate(messages, **kwargs)
    
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        request_id: str = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate response tokens with cancellation support."""
        if not self._initialized:
            await self.initialize()
        
        try:
            async for token in self._provider.stream_generate(messages, **kwargs):
                # Check for cancellation
                if request_id and self.is_cancelled(request_id):
                    logger.info(f"Generation cancelled: {request_id}")
                    break
                yield token
        finally:
            if request_id:
                self.clear_cancel(request_id)
    
    async def check_health(self) -> bool:
        """Check LLM health."""
        if not self._provider:
            return False
        return await self._provider.is_available()


# Singleton instance
llm_service = LLMService()
