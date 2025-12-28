# 🤖 OfflineRAG - Production-Grade Offline RAG Chatbot

A fully offline, production-grade Retrieval-Augmented Generation (RAG) chatbot with a ChatGPT-like interface, supporting multi-modal attachments, voice interaction, and robust local processing.

## ✨ Features

- **100% Offline** - No internet dependency at runtime
- **ChatGPT-Like UI** - Modern, clean, responsive interface
- **Multi-Modal Support** - PDFs, Word, Excel, CSV, Images, Audio, Video
- **Voice Interaction** - Speech-to-text and text-to-speech
- **Advanced RAG** - Hybrid retrieval with semantic + keyword search
- **Cancel-Safe** - Interrupt any operation safely
- **Memory Efficient** - Optimized for local hardware

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Chat UI    │  │ Attachments │  │  Voice Controls         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket + REST
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Chat Service │  │ RAG Engine   │  │  Voice Service       │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Doc Processor│  │ Vector Store │  │  LLM Service         │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Local Models & Storage                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Ollama  │  │ Whisper  │  │  Piper   │  │  ChromaDB        │ │
│  │  (LLM)   │  │  (ASR)   │  │  (TTS)   │  │  (Vector Store)  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Node.js 18+**
3. **Ollama** - For local LLM inference
4. **Tesseract OCR** - For document scanning

### Installation

```bash
# Clone and navigate
cd RAG

# Install backend dependencies
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Download models
python scripts/download_models.py

# Install frontend dependencies
cd ../frontend
npm install

# Start the application
cd ..
.\start.ps1  # Windows
```

### First Run

1. Start Ollama: `ollama serve`
2. Pull a model: `ollama pull llama3.2`
3. Run the app: `.\start.ps1`
4. Open: `http://localhost:3000`

## 📁 Project Structure

```
RAG/
├── backend/
│   ├── app/
│   │   ├── api/              # API routes
│   │   ├── core/             # Core configuration
│   │   ├── models/           # Data models
│   │   ├── services/         # Business logic
│   │   │   ├── chat/         # Chat service
│   │   │   ├── rag/          # RAG engine
│   │   │   ├── documents/    # Document processing
│   │   │   ├── voice/        # ASR/TTS
│   │   │   └── llm/          # LLM integration
│   │   └── utils/            # Utilities
│   ├── data/                 # Local storage
│   ├── models/               # Downloaded models
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── hooks/            # Custom hooks
│   │   ├── services/         # API services
│   │   ├── store/            # State management
│   │   └── styles/           # CSS/Tailwind
│   └── package.json
└── start.ps1                 # Launch script
```

## 🔧 Configuration

Edit `backend/app/core/config.py` for:
- Model paths
- Chunk sizes
- Retrieval parameters
- Voice settings

## 📝 License

MIT License - Use freely for any purpose.
