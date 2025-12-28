/**
 * OfflineRAG - Chat Message Component
 * 
 * Renders individual chat messages with markdown support.
 */

import { memo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check, User, Bot, FileText } from 'lucide-react';
import { useState } from 'react';
import { Message, Source } from '@/types';
import { useChatStore } from '@/store/chatStore';
import { clsx } from 'clsx';

interface ChatMessageProps {
  message: Message;
  isLast?: boolean;
}

function ChatMessage({ message }: ChatMessageProps) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  const { darkMode } = useChatStore();
  const isUser = message.role === 'user';

  const copyToClipboard = async (text: string, id: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div
      className={clsx(
        'flex gap-4 py-4 px-4 rounded-lg mb-4 transition-colors',
        isUser 
          ? 'bg-transparent' 
          : darkMode 
            ? 'bg-gray-800/50' 
            : 'bg-gray-100'
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
          isUser 
            ? 'bg-blue-600' 
            : 'bg-gradient-to-br from-purple-500 to-blue-500'
        )}
      >
        {isUser ? (
          <User size={18} className="text-white" />
        ) : (
          <Bot size={18} className="text-white" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 overflow-hidden">
        {/* Role label */}
        <p className={`text-sm font-medium mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {isUser ? 'You' : 'Assistant'}
        </p>

        {/* Attached documents (for user messages) */}
        {isUser && message.documentIds && message.documentIds.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {message.documentIds.map((docId) => (
              <div
                key={docId}
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm ${
                  darkMode 
                    ? 'bg-gray-700/50 text-gray-300' 
                    : 'bg-gray-200 text-gray-600'
                }`}
              >
                <FileText size={14} />
                <span className="truncate max-w-[150px]">Document</span>
              </div>
            ))}
          </div>
        )}

        {/* Message content with markdown */}
        <div className={`prose max-w-none ${darkMode ? 'prose-invert' : ''} prose-sm`}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              code({ className, children, ...props }: any) {
                const inline = !(props.node?.tagName === 'code' && props.node?.properties?.className);
                const match = /language-(\w+)/.exec(className || '');
                const codeString = String(children).replace(/\n$/, '');
                const codeId = `code-${Math.random().toString(36).slice(2)}`;

                if (!inline && match) {
                  return (
                    <div className="relative group">
                      <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={() => copyToClipboard(codeString, codeId)}
                          className={`p-1.5 rounded transition-colors ${
                            darkMode 
                              ? 'bg-gray-700 hover:bg-gray-600' 
                              : 'bg-gray-200 hover:bg-gray-300'
                          }`}
                        >
                          {copiedCode === codeId ? (
                            <Check size={14} className="text-green-400" />
                          ) : (
                            <Copy size={14} className={darkMode ? 'text-gray-400' : 'text-gray-500'} />
                          )}
                        </button>
                      </div>
                      <SyntaxHighlighter
                        style={(darkMode ? oneDark : oneLight) as any}
                        language={match[1]}
                        PreTag="div"
                        className="rounded-lg !mt-0"
                        {...props}
                      >
                        {codeString}
                      </SyntaxHighlighter>
                    </div>
                  );
                }

                return (
                  <code
                    className={`px-1.5 py-0.5 rounded text-sm ${
                      darkMode ? 'bg-gray-700/50' : 'bg-gray-200'
                    }`}
                    {...props}
                  >
                    {children}
                  </code>
                );
              },
              a({ href, children }) {
                return (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-500 hover:underline"
                  >
                    {children}
                  </a>
                );
              },
              table({ children }) {
                return (
                  <div className="overflow-x-auto">
                    <table className="border-collapse w-full">{children}</table>
                  </div>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className={`mt-4 pt-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <p className={`text-sm font-medium mb-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Sources:</p>
            <div className="space-y-2">
              {message.sources.map((source, idx) => (
                <SourceCard key={idx} source={source} index={idx + 1} darkMode={darkMode} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface SourceCardProps {
  source: Source;
  index: number;
  darkMode: boolean;
}

function SourceCard({ source, index, darkMode }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`p-3 rounded-lg cursor-pointer transition-colors ${
        darkMode 
          ? 'bg-gray-700/30 hover:bg-gray-700/50' 
          : 'bg-gray-100 hover:bg-gray-200'
      }`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-blue-500 bg-blue-500/20 px-2 py-0.5 rounded">
          [{index}]
        </span>
        <span className={`text-sm truncate flex-1 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          {source.filename}
        </span>
        <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          {(source.score * 100).toFixed(0)}% match
        </span>
      </div>
      {expanded && (
        <p className={`mt-2 text-sm line-clamp-3 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {source.preview}
        </p>
      )}
    </div>
  );
}

export default memo(ChatMessage);
