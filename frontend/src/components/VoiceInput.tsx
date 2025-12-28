/**
 * OfflineRAG - Voice Input Component
 * 
 * Push-to-talk microphone button with listening indicator.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, Loader2, Square } from 'lucide-react';
import { transcribeAudio, stopVoice } from '@/services/api';
import { useChatStore } from '@/store/chatStore';

interface VoiceInputProps {
  onTranscription: (text: string) => void;
  darkMode?: boolean;
  disabled?: boolean;
}

export const VoiceInput: React.FC<VoiceInputProps> = ({
  onTranscription,
  darkMode = true,
  disabled = false
}) => {
  const { isRecording, setRecording } = useChatStore();
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number | null>(null);
  const isRecordingRef = useRef(false);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      
      streamRef.current = stream;
      
      // Set up audio level monitoring
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;
      
      // Start monitoring audio levels
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateLevel = () => {
        if (!isRecordingRef.current) return;
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        setAudioLevel(average / 255);
        animationRef.current = requestAnimationFrame(updateLevel);
      };
      
      // Create media recorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm') 
          ? 'audio/webm' 
          : 'audio/mp4'
      });
      
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
        
        // Get the mime type that was actually used
        const mimeType = mediaRecorder.mimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        
        if (audioBlob.size > 0) {
          setIsProcessing(true);
          try {
            // Determine extension from mime type for backend
            const ext = mimeType.includes('webm') ? 'webm' 
                      : mimeType.includes('mp4') ? 'm4a' 
                      : mimeType.includes('ogg') ? 'ogg' 
                      : 'wav';
            const result = await transcribeAudio(audioBlob, 'en', ext);
            if (result.text.trim()) {
              onTranscription(result.text);
            }
          } catch (err) {
            setError('Transcription failed. Please try again.');
            console.error('Transcription error:', err);
          } finally {
            setIsProcessing(false);
          }
        }
        
        // Cleanup stream
        stream.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100); // Collect data every 100ms
      isRecordingRef.current = true;
      setRecording(true);
      
      // Start audio level animation
      updateLevel();
      
    } catch (err) {
      console.error('Failed to start recording:', err);
      setError('Microphone access denied. Please allow access and try again.');
    }
  }, [onTranscription, setRecording]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    isRecordingRef.current = false;
    setRecording(false);
    setAudioLevel(0);
  }, [setRecording]);

  const cancelRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      audioChunksRef.current = []; // Clear chunks so nothing gets transcribed
      if (mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    isRecordingRef.current = false;
    setRecording(false);
    setAudioLevel(0);
    stopVoice(); // Stop backend voice operations
  }, [setRecording]);

  const handleClick = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else if (!isProcessing && !disabled) {
      startRecording();
    }
  }, [isRecording, isProcessing, disabled, startRecording, stopRecording]);

  return (
    <div className="relative flex items-center gap-2">
      <AnimatePresence mode="wait">
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className={`absolute bottom-full mb-2 left-0 px-3 py-1 rounded text-sm ${
              darkMode ? 'bg-red-900/80 text-red-200' : 'bg-red-100 text-red-700'
            }`}
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {isRecording && (
        <motion.button
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          onClick={cancelRecording}
          className={`p-2 rounded-lg transition-colors ${
            darkMode
              ? 'hover:bg-gray-700 text-gray-400'
              : 'hover:bg-gray-200 text-gray-500'
          }`}
          title="Cancel recording"
        >
          <Square className="w-4 h-4" />
        </motion.button>
      )}

      <motion.button
        onClick={handleClick}
        disabled={isProcessing || disabled}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className={`relative p-3 rounded-full transition-all duration-200 ${
          isRecording
            ? darkMode
              ? 'bg-red-600 text-white shadow-lg shadow-red-600/30'
              : 'bg-red-500 text-white shadow-lg shadow-red-500/30'
            : isProcessing
              ? darkMode
                ? 'bg-gray-700 text-gray-400'
                : 'bg-gray-300 text-gray-500'
              : darkMode
                ? 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                : 'bg-gray-200 hover:bg-gray-300 text-gray-600'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        title={isRecording ? 'Stop recording' : 'Start voice input'}
      >
        {/* Pulsing ring when recording */}
        {isRecording && (
          <motion.div
            className="absolute inset-0 rounded-full bg-red-500"
            initial={{ scale: 1, opacity: 0.5 }}
            animate={{
              scale: [1, 1.2 + audioLevel * 0.3, 1],
              opacity: [0.5, 0.2, 0.5]
            }}
            transition={{
              duration: 0.5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        )}
        
        {isProcessing ? (
          <Loader2 className="w-5 h-5 animate-spin" />
        ) : isRecording ? (
          <Mic className="w-5 h-5 relative z-10" />
        ) : (
          <Mic className="w-5 h-5" />
        )}
      </motion.button>

      {/* Recording indicator */}
      <AnimatePresence>
        {isRecording && (
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            className={`flex items-center gap-2 text-sm ${
              darkMode ? 'text-red-400' : 'text-red-600'
            }`}
          >
            <motion.div
              className="w-2 h-2 rounded-full bg-red-500"
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            />
            Listening...
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
