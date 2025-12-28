/**
 * OfflineRAG - Main App Component
 */

import { useEffect, useRef } from 'react';
import { useChatStore } from '@/store/chatStore';
import Sidebar from '@/components/Sidebar';
import ChatArea from '@/components/ChatArea';
import ThemeToggle from '@/components/ThemeToggle';
import { getHealth, getGreeting } from '@/services/api';

function App() {
  const { 
    darkMode, 
    setConnected, 
    setSystemHealth,
    createSession,
    currentSessionId,
    sessions,
    addMessage,
    messages
  } = useChatStore();
  
  // Track if greeting has been shown for this app session
  const greetingShownRef = useRef<Set<string>>(new Set());

  // Initialize dark mode
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    // Also update body background for smooth transitions
    document.body.className = darkMode 
      ? 'bg-gray-900 text-gray-100' 
      : 'bg-white text-gray-900';
  }, [darkMode]);

  // Check system health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await getHealth();
        setSystemHealth(health);
        setConnected(health.status === 'healthy' || health.status === 'degraded');
      } catch (error) {
        setConnected(false);
        setSystemHealth(null);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, [setConnected, setSystemHealth]);

  // Create initial session if none exists
  useEffect(() => {
    if (sessions.length === 0 || !currentSessionId) {
      createSession();
    }
  }, [sessions.length, currentSessionId, createSession]);

  // Fetch greeting message when session is created and no messages exist
  useEffect(() => {
    const fetchGreeting = async () => {
      // Only show greeting once per session and only if no messages
      if (currentSessionId && messages.length === 0 && !greetingShownRef.current.has(currentSessionId)) {
        greetingShownRef.current.add(currentSessionId);
        try {
          const greeting = await getGreeting();
          addMessage({
            role: 'assistant',
            content: greeting.content,
            sources: [],
          });
        } catch (error) {
          // Fallback greeting if API fails
          addMessage({
            role: 'assistant',
            content: "Hello! I'm your AI assistant. How may I help you today?",
            sources: [],
          });
        }
      }
    };

    fetchGreeting();
  }, [currentSessionId, messages.length, addMessage]); // Include all dependencies

  return (
    <div className={`flex h-screen transition-colors duration-300 ${
      darkMode 
        ? 'bg-gray-900 text-gray-100' 
        : 'bg-gray-50 text-gray-900'
    }`}>
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Chat Area */}
      <ChatArea />
      
      {/* Theme Toggle */}
      <ThemeToggle />
    </div>
  );
}

export default App;
