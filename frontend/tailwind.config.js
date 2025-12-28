/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // ChatGPT-inspired color palette
        'chat-bg': {
          light: '#ffffff',
          dark: '#343541',
        },
        'sidebar-bg': {
          light: '#f7f7f8',
          dark: '#202123',
        },
        'user-bubble': {
          light: '#f7f7f8',
          dark: '#343541',
        },
        'assistant-bubble': {
          light: '#ffffff',
          dark: '#444654',
        },
        'accent': {
          DEFAULT: '#10a37f',
          hover: '#0d8a6c',
        },
        'border': {
          light: '#e5e5e5',
          dark: '#4a4b53',
        }
      },
      fontFamily: {
        sans: ['Inter', 'Söhne', 'system-ui', 'sans-serif'],
        mono: ['Söhne Mono', 'Monaco', 'Consolas', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'typing': 'typing 1.5s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        typing: {
          '0%, 100%': { opacity: '0.3' },
          '50%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
