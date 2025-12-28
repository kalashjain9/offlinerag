"""
OfflineRAG - Core Configuration Module
======================================

Centralized configuration management for the entire application.
All settings can be overridden via environment variables.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Any


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # ============================================
    # Application Settings
    # ============================================
    APP_NAME: str = "OfflineRAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Optional[Path] = None
    MODELS_DIR: Optional[Path] = None
    TEMP_DIR: Optional[Path] = None
    
    # ============================================
    # Server Settings
    # ============================================
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 1
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]  # Add your Vercel domain in .env: CORS_ORIGINS=["https://your-app.vercel.app"]
    
    # ============================================
    # LLM Settings (Ollama)
    # ============================================
    LLM_PROVIDER: str = "ollama"  # ollama, llamacpp
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT: int = 120
    
    # LlamaCpp settings (alternative)
    LLAMACPP_MODEL_PATH: Optional[str] = None
    LLAMACPP_N_CTX: int = 4096
    LLAMACPP_N_GPU_LAYERS: int = -1  # -1 = auto
    
    # Generation settings
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_TOP_P: float = 0.9
    LLM_REPEAT_PENALTY: float = 1.1
    
    # ============================================
    # Embedding Settings
    # ============================================
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"  # cpu, cuda, mps
    EMBEDDING_BATCH_SIZE: int = 32
    
    # ============================================
    # Vector Store Settings (ChromaDB)
    # ============================================
    CHROMA_PERSIST_DIR: Optional[Path] = None
    CHROMA_COLLECTION_NAME: str = "documents"
    SIMILARITY_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    
    # ============================================
    # Document Processing Settings
    # ============================================
    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 100
    
    # File size limits (in bytes)
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    MAX_TOTAL_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB
    
    # Supported file types
    SUPPORTED_TEXT_EXTENSIONS: List[str] = [".txt", ".md", ".rst", ".log", ".json", ".xml", ".yaml", ".yml"]
    SUPPORTED_DOC_EXTENSIONS: List[str] = [".pdf", ".docx", ".doc", ".pptx", ".ppt"]
    SUPPORTED_DATA_EXTENSIONS: List[str] = [".csv", ".xlsx", ".xls"]
    SUPPORTED_IMAGE_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg", ".ico"]
    SUPPORTED_AUDIO_EXTENSIONS: List[str] = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".webm", ".aac"]
    SUPPORTED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv"]
    
    # OCR Settings
    OCR_ENABLED: bool = True
    OCR_LANGUAGE: str = "eng"
    TESSERACT_CMD: Optional[str] = None  # Auto-detect on Windows
    
    def get_tesseract_cmd(self) -> Optional[str]:
        """Get Tesseract command, trying common Windows install paths."""
        if self.TESSERACT_CMD:
            return self.TESSERACT_CMD
        
        import os
        import platform
        
        if platform.system() == "Windows":
            # Common Windows install paths
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
            ]
            for path in common_paths:
                if os.path.isfile(path):
                    return path
        
        return None  # Use system PATH
    
    # ============================================
    # Voice Settings
    # ============================================
    # ASR (Whisper)
    WHISPER_MODEL: str = "base"  # tiny, base, small, medium, large
    WHISPER_DEVICE: str = "cpu"
    WHISPER_LANGUAGE: str = "en"
    
    # TTS (Piper)
    TTS_ENABLED: bool = True
    TTS_MODEL: str = "en_US-amy-medium"
    TTS_RATE: float = 1.0
    
    # ============================================
    # RAG Settings
    # ============================================
    RAG_CONTEXT_WINDOW: int = 4096
    RAG_MAX_SOURCES: int = 5
    RAG_HYBRID_SEARCH: bool = True
    RAG_KEYWORD_WEIGHT: float = 0.3
    RAG_SEMANTIC_WEIGHT: float = 0.7
    
    # Cross-encoder re-ranking
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_N: int = 5
    
    # System prompt
    RAG_SYSTEM_PROMPT: str = """You are a helpful AI assistant with access to a knowledge base. 
Answer questions based on the provided context. If the context doesn't contain relevant information, 
say so clearly. Be concise, accurate, and helpful. Never make up information not present in the context."""

    # ============================================
    # Redis Cache Settings
    # ============================================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600
    CACHE_SIMILARITY_THRESHOLD: float = 0.92

    # ============================================
    # Session Settings
    # ============================================
    SESSION_TIMEOUT: int = 3600  # 1 hour
    MAX_CONVERSATION_HISTORY: int = 20
    
    # ============================================
    # Performance Settings
    # ============================================
    MAX_CONCURRENT_UPLOADS: int = 5
    PROCESSING_QUEUE_SIZE: int = 100
    MEMORY_LIMIT_MB: int = 4096
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True
    }
    
    def model_post_init(self, __context: Any) -> None:
        # Set default paths
        if self.DATA_DIR is None:
            object.__setattr__(self, 'DATA_DIR', self.BASE_DIR / "data")
        if self.MODELS_DIR is None:
            object.__setattr__(self, 'MODELS_DIR', self.BASE_DIR / "models")
        if self.TEMP_DIR is None:
            object.__setattr__(self, 'TEMP_DIR', self.BASE_DIR / "temp")
        if self.CHROMA_PERSIST_DIR is None:
            object.__setattr__(self, 'CHROMA_PERSIST_DIR', self.DATA_DIR / "chroma")
        
        # Create directories
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.TEMP_DIR, self.CHROMA_PERSIST_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def all_supported_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        return (
            self.SUPPORTED_TEXT_EXTENSIONS +
            self.SUPPORTED_DOC_EXTENSIONS +
            self.SUPPORTED_DATA_EXTENSIONS +
            self.SUPPORTED_IMAGE_EXTENSIONS +
            self.SUPPORTED_AUDIO_EXTENSIONS +
            self.SUPPORTED_VIDEO_EXTENSIONS
        )


# Global settings instance
settings = Settings()
