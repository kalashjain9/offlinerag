/**
 * OfflineRAG - Sidebar Component
 * 
 * Session list, new chat button, and settings.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, 
  MessageSquare, 
  Trash2,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { useChatStore } from '@/store/chatStore';
import { clsx } from 'clsx';

export default function Sidebar() {
  const {
    sessions,
    currentSessionId,
    sidebarOpen,
    darkMode,
    isConnected,
    systemHealth,
    createSession,
    selectSession,
    deleteSession,
    toggleSidebar,
  } = useChatStore();

  const [hoveredSession, setHoveredSession] = useState<string | null>(null);

  const handleNewChat = () => {
    createSession();
  };

  const formatDate = (date: Date) => {
    const d = new Date(date);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - d.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return d.toLocaleDateString();
  };

  return (
    <>
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 260, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            className={`flex flex-col h-full border-r overflow-hidden transition-colors ${
              darkMode 
                ? 'bg-gray-950 border-gray-800' 
                : 'bg-gray-50 border-gray-200'
            }`}
          >
            {/* New Chat Button */}
            <div className="p-3">
              <button
                onClick={handleNewChat}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg border transition-colors ${
                  darkMode
                    ? 'border-gray-700 hover:bg-gray-800'
                    : 'border-gray-300 hover:bg-gray-100'
                }`}
              >
                <Plus size={18} />
                <span className="font-medium">New Chat</span>
              </button>
            </div>

            {/* Session List */}
            <div className="flex-1 overflow-y-auto px-2 pb-2">
              <div className="space-y-1">
                {sessions.map((session) => (
                  <motion.div
                    key={session.id}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={clsx(
                      'group relative flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors',
                      session.id === currentSessionId
                        ? darkMode ? 'bg-gray-800' : 'bg-gray-200'
                        : darkMode ? 'hover:bg-gray-800/50' : 'hover:bg-gray-100'
                    )}
                    onClick={() => selectSession(session.id)}
                    onMouseEnter={() => setHoveredSession(session.id)}
                    onMouseLeave={() => setHoveredSession(null)}
                  >
                    <MessageSquare size={16} className={`flex-shrink-0 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm truncate">
                        {session.title || 'New Chat'}
                      </p>
                      <p className={`text-xs truncate ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                        {formatDate(session.updatedAt)}
                      </p>
                    </div>
                    
                    {/* Delete button */}
                    <AnimatePresence>
                      {hoveredSession === session.id && (
                        <motion.button
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.8 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteSession(session.id);
                          }}
                          className={`absolute right-2 p-1.5 rounded transition-colors ${
                            darkMode
                              ? 'hover:bg-red-500/20 text-gray-400 hover:text-red-400'
                              : 'hover:bg-red-100 text-gray-500 hover:text-red-500'
                          }`}
                        >
                          <Trash2 size={14} />
                        </motion.button>
                      )}
                    </AnimatePresence>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Bottom Section */}
            <div className={`border-t p-3 space-y-2 ${darkMode ? 'border-gray-800' : 'border-gray-200'}`}>
              {/* System Status */}
              <div className={`flex items-center gap-2 px-3 py-2 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <div className={clsx(
                  'w-2 h-2 rounded-full',
                  isConnected ? 'bg-green-500' : 'bg-red-500'
                )} />
                <span>
                  {isConnected 
                    ? `Connected • ${systemHealth?.version || 'v1.0'}` 
                    : 'Disconnected'
                  }
                </span>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Toggle Button */}
      <button
        onClick={toggleSidebar}
        className={clsx(
          'fixed top-4 z-50 p-2 rounded-lg border transition-all',
          darkMode 
            ? 'bg-gray-900 border-gray-700 hover:bg-gray-800' 
            : 'bg-white border-gray-300 hover:bg-gray-100',
          sidebarOpen ? 'left-[268px]' : 'left-4'
        )}
      >
        {sidebarOpen ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
      </button>
    </>
  );
}
