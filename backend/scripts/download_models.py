"""
OfflineRAG - Model Download Script
==================================

Downloads required models for offline operation.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings


def download_embedding_model():
    """Download the sentence-transformer embedding model."""
    print(f"[*] Downloading embedding model: {settings.EMBEDDING_MODEL}")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print(f"    [OK] Embedding model downloaded")
        return True
    except Exception as e:
        print(f"    [X] Failed: {e}")
        return False


def download_whisper_model():
    """Download the Whisper ASR model."""
    print(f"[*] Downloading Whisper model: {settings.WHISPER_MODEL}")
    
    try:
        import whisper
        model = whisper.load_model(settings.WHISPER_MODEL)
        print(f"    [OK] Whisper model downloaded")
        return True
    except Exception as e:
        print(f"    [X] Failed: {e}")
        return False


def check_ollama():
    """Check if Ollama is available and pull the model."""
    print(f"[*] Checking Ollama model: {settings.OLLAMA_MODEL}")
    
    try:
        import httpx
        
        # Check if Ollama is running
        response = httpx.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            if settings.OLLAMA_MODEL.split(":")[0] in model_names:
                print(f"    [OK] Model already available")
                return True
            else:
                print(f"    [!] Model not found. Run: ollama pull {settings.OLLAMA_MODEL}")
                return False
        else:
            print(f"    [X] Ollama not responding")
            return False
            
    except Exception as e:
        print(f"    [X] Ollama not available: {e}")
        print(f"        Make sure Ollama is installed and running: ollama serve")
        return False


def create_directories():
    """Create required directories."""
    print("[*] Creating directories...")
    
    dirs = [
        settings.DATA_DIR,
        settings.MODELS_DIR,
        settings.TEMP_DIR,
        settings.CHROMA_PERSIST_DIR,
        settings.DATA_DIR / "logs",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"    Created: {d}")
    
    print("    [OK] Directories created")


def main():
    print("")
    print("=" * 50)
    print("  OfflineRAG Model Setup")
    print("=" * 50)
    print("")
    
    # Create directories
    create_directories()
    print("")
    
    # Download models
    results = []
    
    results.append(("Embedding", download_embedding_model()))
    print("")
    
    results.append(("Whisper", download_whisper_model()))
    print("")
    
    results.append(("Ollama", check_ollama()))
    print("")
    
    # Summary
    print("=" * 50)
    print("  Setup Summary")
    print("=" * 50)
    
    all_ok = True
    for name, success in results:
        status = "[OK]" if success else "[X]"
        color_status = status
        print(f"  {name}: {color_status}")
        if not success:
            all_ok = False
    
    print("")
    
    if all_ok:
        print("All models downloaded successfully!")
        print("You can now start the application with: .\\start.ps1")
    else:
        print("Some models failed to download.")
        print("The application may work with reduced functionality.")
    
    print("")


if __name__ == "__main__":
    main()
