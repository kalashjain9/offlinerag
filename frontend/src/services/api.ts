/**
 * OfflineRAG - API Service
 * 
 * Handles all communication with the backend API.
 */

import { ChatRequest, StreamChunk, Document, Session, HealthStatus, TranscriptionResult } from '@/types';

// Use environment variable for API base URL, fallback to relative path for local dev
const API_BASE = import.meta.env.VITE_API_URL 
  ? `${import.meta.env.VITE_API_URL}/api/v1`
  : '/api/v1';

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, error.detail || error.error || 'Request failed');
  }
  return response.json();
}

// ============================================
// Chat API
// ============================================

export async function sendMessage(request: ChatRequest): Promise<ReadableStream<StreamChunk>> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new ApiError(response.status, 'Failed to send message');
  }

  if (!response.body) {
    throw new Error('No response body');
  }

  // Return a transformed stream of StreamChunk objects
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  return new ReadableStream<StreamChunk>({
    async pull(controller) {
      const { done, value } = await reader.read();
      
      if (done) {
        controller.close();
        return;
      }

      const text = decoder.decode(value, { stream: true });
      const lines = text.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6)) as StreamChunk;
            controller.enqueue(data);
            
            if (data.type === 'done' || data.type === 'error') {
              controller.close();
              return;
            }
          } catch (e) {
            // Skip invalid JSON
          }
        }
      }
    },
  });
}

export async function createSession(title?: string): Promise<Session> {
  const response = await fetch(`${API_BASE}/chat/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  return handleResponse<Session>(response);
}

export async function getSessions(): Promise<Session[]> {
  const response = await fetch(`${API_BASE}/chat/sessions`);
  return handleResponse<Session[]>(response);
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new ApiError(response.status, 'Failed to delete session');
  }
}

export async function getGreeting(): Promise<{ role: string; content: string; is_greeting: boolean }> {
  const response = await fetch(`${API_BASE}/chat/greeting`);
  return handleResponse(response);
}

export interface ChatHistoryData {
  sessions: Array<{
    id: string;
    title: string;
    created_at?: string;
    updated_at?: string;
  }>;
  messages: Record<string, Array<{
    id: string;
    role: string;
    content: string;
    timestamp?: string;
    sources?: unknown[];
  }>>;
}

export async function syncChatHistory(
  sessions: unknown[],
  messages: Record<string, unknown[]>
): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE}/chat/sync`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessions, messages }),
  });
  return handleResponse(response);
}

export async function getChatHistory(): Promise<ChatHistoryData> {
  const response = await fetch(`${API_BASE}/chat/history`);
  return handleResponse(response);
}

// ============================================
// Document API
// ============================================

export async function uploadDocument(file: File): Promise<Document> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse<Document>(response);
}

export async function uploadMultipleDocuments(files: File[]): Promise<Document[]> {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  const response = await fetch(`${API_BASE}/documents/upload/multiple`, {
    method: 'POST',
    body: formData,
  });

  const result = await handleResponse<{ documents: Document[] }>(response);
  return result.documents;
}

export async function getDocumentStatus(documentId: string): Promise<Document> {
  const response = await fetch(`${API_BASE}/documents/${documentId}/status`);
  return handleResponse<Document>(response);
}

export async function getDocuments(): Promise<Document[]> {
  const response = await fetch(`${API_BASE}/documents/`);
  const result = await handleResponse<{ documents: Document[] }>(response);
  return result.documents;
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${documentId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new ApiError(response.status, 'Failed to delete document');
  }
}

export async function cancelDocumentProcessing(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${documentId}/cancel`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new ApiError(response.status, 'Failed to cancel processing');
  }
}

// ============================================
// Voice API
// ============================================

export async function transcribeAudio(audioBlob: Blob, language = 'en', ext = 'wav'): Promise<TranscriptionResult> {
  const formData = new FormData();
  formData.append('file', audioBlob, `recording.${ext}`);
  formData.append('language', language);

  const response = await fetch(`${API_BASE}/voice/transcribe`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse<TranscriptionResult>(response);
}

export async function synthesizeSpeech(text: string, rate = 1.0): Promise<Blob> {
  const response = await fetch(`${API_BASE}/voice/synthesize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, rate }),
  });

  if (!response.ok) {
    throw new ApiError(response.status, 'Failed to synthesize speech');
  }

  return response.blob();
}

export async function getVoiceStatus(): Promise<{ 
  asr_available: boolean; 
  tts_available: boolean;
  is_listening: boolean;
  is_speaking: boolean;
}> {
  const response = await fetch(`${API_BASE}/voice/status`);
  return handleResponse(response);
}

export async function stopVoice(): Promise<void> {
  const response = await fetch(`${API_BASE}/voice/stop`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new ApiError(response.status, 'Failed to stop voice');
  }
}

// ============================================
// System API
// ============================================

export async function getHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE}/system/health`);
  return handleResponse<HealthStatus>(response);
}

export async function initializeServices(): Promise<{ status: string; services: Record<string, string> }> {
  const response = await fetch(`${API_BASE}/system/initialize`, {
    method: 'POST',
  });
  return handleResponse(response);
}

export async function getConfig(): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE}/system/config`);
  return handleResponse(response);
}

// WebSocket connection for real-time chat
export function createChatWebSocket(): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  return new WebSocket(`${protocol}//${host}${API_BASE}/chat/ws`);
}

export { ApiError };
