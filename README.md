# 🤖 OfflineRAG - Production-Grade Offline RAG Chatbot

A fully offline, production-grade Retrieval-Augmented Generation (RAG) chatbot with a ChatGPT-like interface, supporting multi-modal attachments, voice interaction, and robust local processing.

![Status](https://img.shields.io/badge/Status-Active-green) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![React](https://img.shields.io/badge/React-18-61dafb) ![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **100% Offline** - No internet dependency at runtime
- **ChatGPT-Like UI** - Modern, clean, responsive interface with dark/light themes
- **Multi-Modal Support** - PDFs, Word, Excel, CSV, PowerPoint, Images, Audio, Video
- **Voice Interaction** - Speech-to-text (Whisper) and text-to-speech
- **Image Understanding** - BLIP-powered image captioning
- **Advanced RAG** - Hybrid retrieval with semantic + keyword search
- **Persistent Storage** - Chat history and documents saved locally
- **Re-ranking** - Cross-encoder re-ranking for better results
- **Cancel-Safe** - Interrupt any operation safely

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
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Ollama     │  │   Whisper    │  │     ChromaDB         │   │
│  │   (LLM)      │  │   (ASR)      │  │   (Vector Store)     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Download |
|-------------|---------|----------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| Ollama | Latest | [ollama.ai](https://ollama.ai/) |
| Tesseract OCR | 5.0+ | [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) |
| FFmpeg | Latest | [ffmpeg.org](https://ffmpeg.org/download.html) |

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/kalashjain9/offlinerag.git
cd offlinerag
```

#### 2. Set Up Ollama (LLM)

```bash
# Start Ollama service
ollama serve

# In a new terminal, pull a model (choose one)
ollama pull llama3.2        # Recommended - 3B parameters
ollama pull mistral         # Alternative - 7B parameters
ollama pull phi3            # Lightweight - 3.8B parameters
```

#### 3. Set Up Backend

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 4. Set Up Frontend (New Terminal)

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

#### 5. Open the App

Open your browser and navigate to: **http://localhost:3000**

---

### 🖥️ Quick Run (After Installation)

**Windows (PowerShell):**
```powershell
# Terminal 1: Backend
cd backend; .\venv\Scripts\Activate.ps1; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend; npm run dev
```

**macOS/Linux:**
```bash
# Terminal 1: Backend
cd backend && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

> **Note:** Make sure Ollama is running (`ollama serve`) before starting the backend.

## 📁 Project Structure

```
offlinerag/
├── backend/
│   ├── app/
│   │   ├── api/              # API routes (chat, documents, voice)
│   │   ├── core/             # Configuration & settings
│   │   ├── models/           # Pydantic data models
│   │   └── services/         # Business logic
│   │       ├── chat/         # Chat management & history
│   │       ├── rag/          # RAG engine & retrieval
│   │       ├── documents/    # Document processing & storage
│   │       ├── voice/        # Whisper ASR & TTS
│   │       └── llm/          # Ollama LLM integration
│   ├── data/                 # Persistent storage
│   │   ├── documents/        # Uploaded documents
│   │   ├── chats/            # Chat history
│   │   └── chroma/           # Vector database
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── Chat/         # Chat interface
│   │   │   ├── Sidebar/      # Navigation & history
│   │   │   └── Settings/     # Configuration UI
│   │   ├── hooks/            # Custom React hooks
│   │   ├── services/         # API client services
│   │   ├── store/            # Zustand state management
│   │   └── styles/           # Tailwind CSS
│   └── package.json
└── README.md
```

## 📚 Supported File Types

| Category | Formats |
|----------|---------|
| Documents | PDF, DOCX, TXT, MD, RTF |
| Spreadsheets | XLSX, XLS, CSV |
| Presentations | PPTX, PPT |
| Images | PNG, JPG, JPEG, GIF, WebP |
| Audio | MP3, WAV, M4A, FLAC, OGG |
| Video | MP4, MKV, AVI, MOV, WebM |

## 🔧 Configuration

Edit `backend/app/core/config.py` for:
- Model paths and LLM settings
- Chunk sizes and retrieval parameters
- Voice/Whisper settings
- CORS origins

## 🛠️ Troubleshooting

**Ollama not connecting:**
```bash
curl http://localhost:11434/api/tags  # Check if running
ollama serve                           # Restart if needed
```

**Tesseract OCR not found:**
- Windows: Default path `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

**Port already in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## 🔒 Privacy

- **No data leaves your machine** - Everything runs locally
- **No API keys required** - Uses local Ollama models
- **No telemetry** - Zero tracking or analytics

## 📝 License

MIT License - Use freely for any purpose.

---

**Made with ❤️ for privacy-conscious users who want powerful AI without the cloud.**
