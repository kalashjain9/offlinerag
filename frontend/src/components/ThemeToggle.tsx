/**
 * OfflineRAG - Theme Toggle Component
 * 
 * Floating button to toggle between dark and light modes.
 */

import { Moon, Sun } from 'lucide-react';
import { motion } from 'framer-motion';
import { useChatStore } from '@/store/chatStore';

export default function ThemeToggle() {
  const { darkMode, toggleDarkMode } = useChatStore();

  return (
    <motion.button
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      onClick={toggleDarkMode}
      className={`fixed bottom-6 right-6 p-3 rounded-full shadow-lg transition-colors z-50 ${
        darkMode
          ? 'bg-gray-700 hover:bg-gray-600 text-yellow-400'
          : 'bg-white hover:bg-gray-100 text-gray-700 border border-gray-200'
      }`}
      title={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
    >
      <motion.div
        initial={false}
        animate={{ rotate: darkMode ? 0 : 180 }}
        transition={{ duration: 0.3 }}
      >
        {darkMode ? <Sun size={22} /> : <Moon size={22} />}
      </motion.div>
    </motion.button>
  );
}
