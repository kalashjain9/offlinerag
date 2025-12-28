/**
 * OfflineRAG - Welcome Screen Component
 * 
 * Shown when there are no messages in the current session.
 */

import { motion } from 'framer-motion';
import { 
  FileText, 
  Mic, 
  Search, 
  Zap,
  Shield,
  Database,
  MessageSquare
} from 'lucide-react';
import { useChatStore } from '@/store/chatStore';

const features = [
  {
    icon: FileText,
    title: 'Multi-Modal Documents',
    description: 'Upload PDFs, Word, Excel, images, audio, and video files',
  },
  {
    icon: Search,
    title: 'Intelligent Search',
    description: 'Semantic search across all your documents',
  },
  {
    icon: Mic,
    title: 'Voice Interaction',
    description: 'Speak your questions and listen to responses',
  },
  {
    icon: Shield,
    title: '100% Offline',
    description: 'Your data never leaves your device',
  },
];

const suggestions = [
  'Summarize the key points from my uploaded documents',
  'What are the main topics discussed in my files?',
  'Find information about specific topics in my knowledge base',
  'Compare and contrast different documents',
];

export default function WelcomeScreen() {
  const { darkMode } = useChatStore();
  
  return (
    <div className="flex flex-col items-center justify-center min-h-full px-4 py-12">
      {/* Logo and Title */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-accent flex items-center justify-center">
          <MessageSquare size={32} className="text-white" />
        </div>
        <h1 className={`text-3xl font-bold mb-2 ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
          OfflineRAG
        </h1>
        <p className={`max-w-md ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          Your fully offline AI assistant with document intelligence. 
          Upload files, ask questions, and get accurate answers.
        </p>
      </motion.div>

      {/* Features Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl w-full mb-12"
      >
        {features.map((feature, index) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 + index * 0.05 }}
            className={`flex items-start gap-4 p-4 rounded-xl border transition-colors ${
              darkMode 
                ? 'bg-gray-800/50 border-gray-700 hover:border-gray-600' 
                : 'bg-gray-50 border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="p-2 rounded-lg bg-accent/20 text-accent">
              <feature.icon size={20} />
            </div>
            <div>
              <h3 className={`font-medium mb-1 ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                {feature.title}
              </h3>
              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {feature.description}
              </p>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Suggestions */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="w-full max-w-2xl"
      >
        <h2 className={`text-sm font-medium mb-3 flex items-center gap-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          <Zap size={14} />
          Try asking
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              className={`text-left p-3 rounded-lg border transition-colors text-sm ${
                darkMode
                  ? 'bg-gray-800/30 border-gray-700 hover:bg-gray-700/50 hover:border-gray-600 text-gray-300'
                  : 'bg-gray-50 border-gray-200 hover:bg-gray-100 hover:border-gray-300 text-gray-700'
              }`}
            >
              "{suggestion}"
            </button>
          ))}
        </div>
      </motion.div>

      {/* Status Bar */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className={`mt-12 flex items-center gap-6 text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}
      >
        <div className="flex items-center gap-2">
          <Database size={12} />
          <span>Local Vector Store</span>
        </div>
        <div className="flex items-center gap-2">
          <Shield size={12} />
          <span>No Cloud Required</span>
        </div>
        <div className="flex items-center gap-2">
          <Zap size={12} />
          <span>GPU Accelerated</span>
        </div>
      </motion.div>
    </div>
  );
}
