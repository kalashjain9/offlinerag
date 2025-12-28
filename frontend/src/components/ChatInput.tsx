/**
 * OfflineRAG - Chat Input Component
 * 
 * Message input with file upload and voice recording.
 */

import { useState, useRef, useCallback, useEffect, KeyboardEvent, ChangeEvent } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  Paperclip, 
  X, 
  Loader2,
  FileText,
  Image as ImageIcon,
  Music,
  Video,
  Table
} from 'lucide-react';
import { useChatStore } from '@/store/chatStore';
import { sendMessage, uploadDocument, cancelDocumentProcessing, getDocumentStatus } from '@/services/api';
import { Document, DocumentType, EXTENSION_TO_TYPE } from '@/types';
import { VoiceInput } from './VoiceInput';
import { clsx } from 'clsx';

export default function ChatInput() {
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    isGenerating,
    pendingDocuments,
    currentSessionId,
    darkMode,
    addMessage,
    updateMessage,
    setStreaming,
    addPendingDocument,
    removePendingDocument,
    clearPendingDocuments,
  } = useChatStore();

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  // Handle send message
  const handleSend = useCallback(async () => {
    if (!input.trim() && pendingDocuments.length === 0) return;
    if (isGenerating) return;

    const message = input.trim();
    const documentIds = pendingDocuments.map(d => d.id);
    
    setInput('');
    clearPendingDocuments();

    // Add user message
    addMessage({
      role: 'user',
      content: message,
      documentIds,
    });

    // Add placeholder for assistant response
    const assistantMessageId = addMessage({
      role: 'assistant',
      content: '',
      isStreaming: true,
    });

    setStreaming(true);

    try {
      // Send message and stream response
      const stream = await sendMessage({
        message,
        session_id: currentSessionId || undefined,
        document_ids: documentIds.length > 0 ? documentIds : undefined,
        stream: true,
        include_sources: true,
      });

      const reader = stream.getReader();
      let fullContent = '';
      let sources: any[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        if (value.type === 'token' && value.content) {
          fullContent += value.content;
          updateMessage(assistantMessageId, { content: fullContent });
        } else if (value.type === 'source' && value.sources) {
          sources = value.sources;
        } else if (value.type === 'error') {
          throw new Error(value.error || 'Unknown error');
        }
      }

      // Finalize message
      updateMessage(assistantMessageId, {
        content: fullContent,
        sources,
        isStreaming: false,
      });

    } catch (error) {
      console.error('Chat error:', error);
      updateMessage(assistantMessageId, {
        content: 'Sorry, an error occurred while generating the response. Please try again.',
        isStreaming: false,
      });
    } finally {
      setStreaming(false);
    }
  }, [input, pendingDocuments, isGenerating, currentSessionId, addMessage, updateMessage, setStreaming, clearPendingDocuments]);

  // Handle keyboard shortcuts
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Handle file upload
  const handleFileSelect = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);

    for (const file of Array.from(files)) {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      const fileType = EXTENSION_TO_TYPE[ext] || 'unknown';

      // Add as pending with uploading status
      const pendingDoc: Document = {
        id: `pending-${Date.now()}-${Math.random()}`,
        filename: file.name,
        fileType: fileType as DocumentType,
        fileSize: file.size,
        status: 'uploading',
        progress: 0,
        createdAt: new Date(),
      };
      addPendingDocument(pendingDoc);

      try {
        // Upload to server
        const result = await uploadDocument(file);
        
        // Update with real ID - keep as 'processing' while backend processes
        removePendingDocument(pendingDoc.id);
        const uploadedDoc = {
          ...pendingDoc,
          id: result.id,
          status: 'processing' as const,
        };
        addPendingDocument(uploadedDoc);

        // Poll for document status until ready
        const pollStatus = async () => {
          try {
            const status = await getDocumentStatus(result.id);
            if (status.status === 'ready') {
              removePendingDocument(result.id);
              addPendingDocument({
                ...uploadedDoc,
                status: 'ready',
              });
            } else if (status.status === 'error') {
              removePendingDocument(result.id);
            } else {
              // Still processing, poll again
              setTimeout(pollStatus, 1000);
            }
          } catch (e) {
            // If status check fails, assume ready (file was uploaded)
            removePendingDocument(result.id);
            addPendingDocument({
              ...uploadedDoc,
              status: 'ready',
            });
          }
        };
        
        // Start polling after a short delay
        setTimeout(pollStatus, 500);
        
      } catch (error) {
        console.error('Upload failed:', error);
        removePendingDocument(pendingDoc.id);
      }
    }

    setIsUploading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Handle cancelling a pending document
  const handleCancelUpload = async (docId: string) => {
    removePendingDocument(docId);
    if (!docId.startsWith('pending-')) {
      try {
        await cancelDocumentProcessing(docId);
      } catch (e) {
        // Ignore errors
      }
    }
  };

  // Handle voice transcription
  const handleVoiceTranscription = useCallback((text: string) => {
    setInput(prev => prev + (prev ? ' ' : '') + text);
    // Focus the textarea
    textareaRef.current?.focus();
  }, []);

  // Get file icon
  const getFileIcon = (type: DocumentType) => {
    switch (type) {
      case 'image':
        return <ImageIcon size={14} />;
      case 'audio':
        return <Music size={14} />;
      case 'video':
        return <Video size={14} />;
      case 'excel':
      case 'csv':
        return <Table size={14} />;
      default:
        return <FileText size={14} />;
    }
  };

  return (
    <div className={`border-t p-4 transition-colors duration-300 ${
      darkMode 
        ? 'border-gray-700 bg-gray-900' 
        : 'border-gray-200 bg-white'
    }`}>
      <div className="max-w-3xl mx-auto">
        {/* Pending Documents */}
        <AnimatePresence>
          {pendingDocuments.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="flex flex-wrap gap-2 mb-3"
            >
              {pendingDocuments.map((doc) => (
                <motion.div
                  key={doc.id}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm ${
                    darkMode
                      ? 'bg-gray-700 text-gray-200'
                      : 'bg-gray-100 text-gray-700 border border-gray-200'
                  } ${doc.status === 'uploading' ? 'animate-pulse' : ''}`}
                >
                  {doc.status === 'uploading' ? (
                    <Loader2 size={14} className="animate-spin text-blue-500" />
                  ) : doc.status === 'processing' ? (
                    <Loader2 size={14} className="animate-spin text-yellow-500" />
                  ) : doc.status === 'ready' ? (
                    <span className="text-green-500">✓</span>
                  ) : (
                    getFileIcon(doc.fileType)
                  )}
                  <span className="truncate max-w-[150px]">{doc.filename}</span>
                  {doc.status === 'ready' && (
                    <span className="text-xs text-green-500">Ready</span>
                  )}
                  <button
                    onClick={() => handleCancelUpload(doc.id)}
                    className={`p-0.5 rounded hover:bg-red-500/20 transition-colors ${
                      darkMode ? 'text-gray-400 hover:text-red-400' : 'text-gray-500 hover:text-red-500'
                    }`}
                    title="Remove file"
                  >
                    <X size={12} />
                  </button>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input Area */}
        <div className={`relative flex items-end gap-2 rounded-2xl border transition-colors ${
          darkMode
            ? 'bg-gray-800 border-gray-600 focus-within:border-gray-500'
            : 'bg-gray-50 border-gray-300 focus-within:border-gray-400'
        }`}>
          {/* File Upload Button */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={handleFileSelect}
            accept=".txt,.md,.json,.xml,.yaml,.yml,.pdf,.docx,.doc,.xlsx,.xls,.csv,.png,.jpg,.jpeg,.gif,.webp,.bmp,.tiff,.svg,.ico,.mp3,.wav,.flac,.m4a,.ogg,.mp4,.mkv,.avi,.mov,.pptx,.ppt"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className={`p-3 transition-colors disabled:opacity-50 ${
              darkMode
                ? 'text-gray-400 hover:text-gray-200'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            title="Attach files"
          >
            {isUploading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Paperclip size={20} />
            )}
          </button>

          {/* Text Input */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message OfflineRAG..."
            rows={1}
            className={`flex-1 bg-transparent py-3 pr-2 resize-none focus:outline-none max-h-[200px] ${
              darkMode
                ? 'text-gray-100 placeholder-gray-500'
                : 'text-gray-900 placeholder-gray-400'
            }`}
            disabled={isGenerating}
          />

          {/* Voice Button */}
          <VoiceInput
            onTranscription={handleVoiceTranscription}
            darkMode={darkMode}
            disabled={isGenerating}
          />

          {/* Send Button */}
          <button
            onClick={handleSend}
            disabled={(!input.trim() && pendingDocuments.length === 0) || isGenerating}
            className={clsx(
              'p-3 mr-1 rounded-xl transition-colors',
              input.trim() || pendingDocuments.length > 0
                ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white hover:from-purple-600 hover:to-blue-600'
                : darkMode
                  ? 'text-gray-500 cursor-not-allowed'
                  : 'text-gray-400 cursor-not-allowed'
            )}
          >
            {isGenerating ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>

        {/* Disclaimer */}
        <p className={`text-xs text-center mt-2 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          OfflineRAG runs 100% locally. Your data never leaves your device.
        </p>
      </div>
    </div>
  );
}
