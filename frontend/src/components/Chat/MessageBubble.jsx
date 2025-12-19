import { forwardRef } from 'react'
import { motion } from 'framer-motion'
import { HiUser } from 'react-icons/hi'
import { BsRobot } from 'react-icons/bs'
import ReactMarkdown from 'react-markdown'
import PolicyBadge from './PolicyBadge'

const MessageBubble = forwardRef(({ message, index }, ref) => {
  const isUser = message.role === 'user'
  const metadata = message.metadata || {}

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ delay: index * 0.03, duration: 0.3 }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
    >
      {/* Avatar */}
      <div className={`flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center shadow-lg ${
        isUser
          ? 'bg-gradient-to-br from-blue-500 to-blue-600'
          : 'bg-gradient-to-br from-gray-700 to-gray-800 border border-gray-600'
      }`}>
        {isUser ? (
          <HiUser className="w-5 h-5 text-white" />
        ) : (
          <BsRobot className="w-5 h-5 text-blue-400" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex flex-col gap-2 max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Sender Label */}
        <span className="text-xs text-gray-500 px-1">
          {isUser ? 'You' : 'QuestRAG AI'}
        </span>

        {/* Message Bubble */}
        <div className={`px-4 py-3 rounded-2xl shadow-md ${
          isUser
            ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-tr-md'
            : 'bg-gray-800 text-gray-100 border border-gray-700 rounded-tl-md'
        }`}>
          <ReactMarkdown
            className="text-sm leading-relaxed prose prose-sm prose-invert max-w-none"
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              ul: ({ children }) => <ul className="list-disc ml-4 mb-2 space-y-1">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal ml-4 mb-2 space-y-1">{children}</ol>,
              li: ({ children }) => <li className="text-sm">{children}</li>,
              code: ({ children, inline }) => (
                inline 
                  ? <code className="px-1.5 py-0.5 rounded bg-black/20 text-xs font-mono">{children}</code>
                  : <code className="block p-3 rounded-lg bg-black/20 text-xs font-mono overflow-x-auto">{children}</code>
              ),
              strong: ({ children }) => <strong className="font-semibold text-blue-300">{children}</strong>,
              a: ({ children, href }) => (
                <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
                  {children}
                </a>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Metadata (for assistant messages) */}
        {!isUser && metadata.policy_action && (
          <div className="flex flex-wrap items-center gap-2 text-xs px-1">
            <PolicyBadge 
              action={metadata.policy_action} 
              confidence={metadata.policy_confidence}
            />
            {metadata.documents_retrieved > 0 && (
              <span className="px-2 py-1 bg-blue-900/30 text-blue-300 border border-blue-500/30 rounded-full">
                {metadata.documents_retrieved} docs
              </span>
            )}
            {metadata.total_time_ms && (
              <span className="text-gray-500">
                {metadata.total_time_ms}ms
              </span>
            )}
          </div>
        )}

        {/* Timestamp */}
        <span className="text-xs text-gray-600 px-1">
          {new Date(message.timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
          })}
        </span>
      </div>
    </motion.div>
  )
})

MessageBubble.displayName = 'MessageBubble'

export default MessageBubble