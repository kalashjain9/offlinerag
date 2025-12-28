/**
 * OfflineRAG - Zustand Store
 * 
 * Global state management for the application.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { v4 as uuidv4 } from 'uuid';
import { Message, Document, Session, HealthStatus } from '@/types';
import { syncChatHistory } from '@/services/api';

interface ChatStore {
  // Sessions
  sessions: Session[];
  currentSessionId: string | null;
  
  // Messages
  messages: Message[];
  isGenerating: boolean;
  streamingContent: string;
  
  // Documents
  documents: Document[];
  pendingDocuments: Document[];
  
  // Voice
  isRecording: boolean;
  isSpeaking: boolean;
  
  // UI
  sidebarOpen: boolean;
  showSources: boolean;
  darkMode: boolean;
  
  // System
  isConnected: boolean;
  systemHealth: HealthStatus | null;
  
  // Actions - Sessions
  createSession: () => string;
  selectSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  updateSessionTitle: (sessionId: string, title: string) => void;
  
  // Actions - Messages
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => string;
  updateMessage: (messageId: string, updates: Partial<Message>) => void;
  appendToMessage: (messageId: string, content: string) => void;
  setStreaming: (isGenerating: boolean, content?: string) => void;
  clearMessages: () => void;
  
  // Actions - Documents
  addDocument: (doc: Document) => void;
  updateDocument: (docId: string, updates: Partial<Document>) => void;
  removeDocument: (docId: string) => void;
  addPendingDocument: (doc: Document) => void;
  removePendingDocument: (docId: string) => void;
  clearPendingDocuments: () => void;
  
  // Actions - Voice
  setRecording: (isRecording: boolean) => void;
  setSpeaking: (isSpeaking: boolean) => void;
  
  // Actions - UI
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  toggleSources: () => void;
  toggleDarkMode: () => void;
  
  // Actions - System
  setConnected: (connected: boolean) => void;
  setSystemHealth: (health: HealthStatus | null) => void;
  
  // Actions - Persistence
  syncToBackend: () => Promise<void>;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      // Initial state
      sessions: [],
      currentSessionId: null,
      messages: [],
      isGenerating: false,
      streamingContent: '',
      documents: [],
      pendingDocuments: [],
      isRecording: false,
      isSpeaking: false,
      sidebarOpen: true,
      showSources: false,
      darkMode: true,
      isConnected: false,
      systemHealth: null,
      
      // Session actions
      createSession: () => {
        const session: Session = {
          id: uuidv4(),
          title: 'New Chat',
          messages: [],
          documentIds: [],
          createdAt: new Date(),
          updatedAt: new Date(),
        };
        
        set(state => ({
          sessions: [session, ...state.sessions],
          currentSessionId: session.id,
          messages: [],
        }));
        
        return session.id;
      },
      
      selectSession: (sessionId) => {
        const session = get().sessions.find(s => s.id === sessionId);
        if (session) {
          set({
            currentSessionId: sessionId,
            messages: session.messages,
          });
        }
      },
      
      deleteSession: (sessionId) => {
        set(state => {
          const newSessions = state.sessions.filter(s => s.id !== sessionId);
          const newCurrentId = state.currentSessionId === sessionId
            ? (newSessions[0]?.id || null)
            : state.currentSessionId;
          
          return {
            sessions: newSessions,
            currentSessionId: newCurrentId,
            messages: newCurrentId
              ? (newSessions.find(s => s.id === newCurrentId)?.messages || [])
              : [],
          };
        });
      },
      
      updateSessionTitle: (sessionId, title) => {
        set(state => ({
          sessions: state.sessions.map(s =>
            s.id === sessionId ? { ...s, title, updatedAt: new Date() } : s
          ),
        }));
      },
      
      // Message actions
      addMessage: (message) => {
        const id = uuidv4();
        const fullMessage: Message = {
          ...message,
          id,
          timestamp: new Date(),
        };
        
        set(state => {
          const newMessages = [...state.messages, fullMessage];
          
          // Update session
          const updatedSessions = state.sessions.map(s =>
            s.id === state.currentSessionId
              ? { ...s, messages: newMessages, updatedAt: new Date() }
              : s
          );
          
          return {
            messages: newMessages,
            sessions: updatedSessions,
          };
        });
        
        // Debounced sync to backend after adding a message
        setTimeout(() => get().syncToBackend(), 1000);
        
        return id;
      },
      
      updateMessage: (messageId, updates) => {
        set(state => {
          const newMessages = state.messages.map(m =>
            m.id === messageId ? { ...m, ...updates } : m
          );
          
          const updatedSessions = state.sessions.map(s =>
            s.id === state.currentSessionId
              ? { ...s, messages: newMessages, updatedAt: new Date() }
              : s
          );
          
          return {
            messages: newMessages,
            sessions: updatedSessions,
          };
        });
      },
      
      appendToMessage: (messageId, content) => {
        set(state => ({
          messages: state.messages.map(m =>
            m.id === messageId ? { ...m, content: m.content + content } : m
          ),
        }));
      },
      
      setStreaming: (isGenerating, content = '') => {
        set({ isGenerating, streamingContent: content });
      },
      
      clearMessages: () => {
        set(state => {
          const updatedSessions = state.sessions.map(s =>
            s.id === state.currentSessionId
              ? { ...s, messages: [], updatedAt: new Date() }
              : s
          );
          
          return {
            messages: [],
            sessions: updatedSessions,
          };
        });
      },
      
      // Document actions
      addDocument: (doc) => {
        set(state => ({
          documents: [...state.documents, doc],
        }));
      },
      
      updateDocument: (docId, updates) => {
        set(state => ({
          documents: state.documents.map(d =>
            d.id === docId ? { ...d, ...updates } : d
          ),
        }));
      },
      
      removeDocument: (docId) => {
        set(state => ({
          documents: state.documents.filter(d => d.id !== docId),
        }));
      },
      
      addPendingDocument: (doc) => {
        set(state => ({
          pendingDocuments: [...state.pendingDocuments, doc],
        }));
      },
      
      removePendingDocument: (docId) => {
        set(state => ({
          pendingDocuments: state.pendingDocuments.filter(d => d.id !== docId),
        }));
      },
      
      clearPendingDocuments: () => {
        set({ pendingDocuments: [] });
      },
      
      // Voice actions
      setRecording: (isRecording) => set({ isRecording }),
      setSpeaking: (isSpeaking) => set({ isSpeaking }),
      
      // UI actions
      toggleSidebar: () => set(state => ({ sidebarOpen: !state.sidebarOpen })),
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleSources: () => set(state => ({ showSources: !state.showSources })),
      toggleDarkMode: () => {
        set(state => {
          const newMode = !state.darkMode;
          document.documentElement.classList.toggle('dark', newMode);
          return { darkMode: newMode };
        });
      },
      
      // System actions
      setConnected: (connected) => set({ isConnected: connected }),
      setSystemHealth: (health) => set({ systemHealth: health }),
      
      // Persistence actions
      syncToBackend: async () => {
        const state = get();
        try {
          // Build messages map by session
          const messagesMap: Record<string, unknown[]> = {};
          for (const session of state.sessions) {
            messagesMap[session.id] = session.messages.map(m => ({
              id: m.id,
              role: m.role,
              content: m.content,
              timestamp: m.timestamp?.toISOString?.() ?? m.timestamp,
              sources: m.sources,
            }));
          }
          
          // Sync to backend
          await syncChatHistory(
            state.sessions.map(s => ({
              id: s.id,
              title: s.title,
              created_at: s.createdAt?.toISOString?.() ?? s.createdAt,
              updated_at: s.updatedAt?.toISOString?.() ?? s.updatedAt,
            })),
            messagesMap
          );
        } catch (error) {
          console.error('Failed to sync to backend:', error);
        }
      },
    }),
    {
      name: 'offlinerag-storage',
      partialize: (state) => ({
        sessions: state.sessions,
        currentSessionId: state.currentSessionId,
        messages: state.messages,
        documents: state.documents,
        darkMode: state.darkMode,
        sidebarOpen: state.sidebarOpen,
      }),
    }
  )
);
