/**
 * OfflineRAG - Type Definitions
 */

// Message types
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  documentIds?: string[];
  sources?: Source[];
  isStreaming?: boolean;
}

export interface Source {
  document_id: string;
  filename: string;
  chunk_id: string;
  score: number;
  preview: string;
}

// Document types
export interface Document {
  id: string;
  filename: string;
  fileType: DocumentType;
  fileSize: number;
  status: DocumentStatus;
  progress: number;
  errorMessage?: string;
  createdAt: Date;
}

export type DocumentType =
  | 'text'
  | 'pdf'
  | 'word'
  | 'excel'
  | 'csv'
  | 'image'
  | 'audio'
  | 'video'
  | 'powerpoint'
  | 'unknown';

export type DocumentStatus =
  | 'uploading'
  | 'processing'
  | 'indexing'
  | 'ready'
  | 'error'
  | 'cancelled';

// Session types
export interface Session {
  id: string;
  title: string;
  messages: Message[];
  documentIds: string[];
  createdAt: Date;
  updatedAt: Date;
}

// API types
export interface ChatRequest {
  message: string;
  session_id?: string;
  document_ids?: string[];
  stream?: boolean;
  include_sources?: boolean;
}

export interface StreamChunk {
  type: 'token' | 'source' | 'done' | 'error';
  content?: string;
  sources?: Source[];
  error?: string;
}

// Voice types
export interface TranscriptionResult {
  text: string;
  language: string;
  confidence: number;
  duration: number;
}

// System types
export interface HealthStatus {
  status: string;
  version: string;
  llm_available: boolean;
  embedding_available: boolean;
  vector_store_available: boolean;
  voice_available: boolean;
  uptime: number;
}

export interface SystemInfo {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  active_sessions: number;
  documents_indexed: number;
  models_loaded: string[];
}

// UI State types
export interface AppState {
  // Chat state
  sessions: Session[];
  currentSessionId: string | null;
  messages: Message[];
  isGenerating: boolean;
  
  // Document state
  documents: Document[];
  pendingDocuments: Document[];
  
  // Voice state
  isRecording: boolean;
  isSpeaking: boolean;
  
  // UI state
  sidebarOpen: boolean;
  showSources: boolean;
  darkMode: boolean;
  
  // System state
  isConnected: boolean;
  systemHealth: HealthStatus | null;
}

// File icon mapping
export const FILE_ICONS: Record<DocumentType, string> = {
  text: '📄',
  pdf: '📕',
  word: '📘',
  excel: '📊',
  csv: '📈',
  image: '🖼️',
  audio: '🎵',
  video: '🎬',
  powerpoint: '📙',
  unknown: '📁',
};

// File type to extension mapping
export const EXTENSION_TO_TYPE: Record<string, DocumentType> = {
  '.txt': 'text',
  '.md': 'text',
  '.json': 'text',
  '.xml': 'text',
  '.yaml': 'text',
  '.yml': 'text',
  '.pdf': 'pdf',
  '.doc': 'word',
  '.docx': 'word',
  '.xls': 'excel',
  '.xlsx': 'excel',
  '.csv': 'csv',
  '.png': 'image',
  '.jpg': 'image',
  '.jpeg': 'image',
  '.gif': 'image',
  '.webp': 'image',
  '.bmp': 'image',
  '.tiff': 'image',
  '.svg': 'image',
  '.ico': 'image',
  '.mp3': 'audio',
  '.wav': 'audio',
  '.flac': 'audio',
  '.m4a': 'audio',
  '.ogg': 'audio',
  '.aac': 'audio',
  '.mp4': 'video',
  '.mkv': 'video',
  '.avi': 'video',
  '.mov': 'video',
  '.webm': 'video',
  '.ppt': 'powerpoint',
  '.pptx': 'powerpoint',
};
