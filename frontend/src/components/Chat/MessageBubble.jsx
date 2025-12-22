import { forwardRef } from 'react'
import { motion } from 'framer-motion'
import { HiUser } from 'react-icons/hi'
import { BsRobot } from 'react-icons/bs'
import ReactMarkdown from 'react-markdown'
import PolicyBadge from './PolicyBadge'
import { cn } from '../../lib/utils'

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
      className={cn("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}
    >
      {/* Avatar */}
      <div className={cn(
        "flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center shadow-lg",
        isUser
          ? "bg-gradient-to-br from-violet-500 to-blue-500"
          : "bg-gradient-to-br from-violet-600 to-cyan-600"
      )}>
        {isUser ? (
          <HiUser className="w-5 h-5 text-white" />
        ) : (
          <BsRobot className="w-5 h-5 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={cn("flex flex-col gap-2 max-w-[75%]", isUser ? "items-end" : "items-start")}>
        {/* Sender Label */}
        <span className="text-xs text-muted-foreground px-1">
          {isUser ? 'You' : 'QuestRAG AI'}
        </span>

        {/* Message Bubble */}
        <div className={cn(
          "px-4 py-3 rounded-2xl shadow-md",
          isUser
            ? "bg-gradient-to-br from-violet-600 to-blue-600 text-white rounded-tr-md"
            : "bg-accent text-foreground border border-border/50 rounded-tl-md"
        )}>
          <ReactMarkdown
            className={cn(
              "text-sm leading-relaxed prose prose-sm max-w-none",
              isUser ? "prose-invert" : "dark:prose-invert"
            )}
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              ul: ({ children }) => <ul className="list-disc ml-4 mb-2 space-y-1">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal ml-4 mb-2 space-y-1">{children}</ol>,
              li: ({ children }) => <li className="text-sm">{children}</li>,
              code: ({ children, inline }) => (
                inline 
                  ? <code className={cn("px-1.5 py-0.5 rounded text-xs font-mono", isUser ? "bg-white/20" : "bg-muted")}>{children}</code>
                  : <code className={cn("block p-3 rounded-lg text-xs font-mono overflow-x-auto", isUser ? "bg-white/10" : "bg-muted")}>{children}</code>
              ),
              strong: ({ children }) => <strong className={cn("font-semibold", isUser ? "text-white" : "text-violet-600 dark:text-violet-400")}>{children}</strong>,
              a: ({ children, href }) => (
                <a href={href} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">
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
              <span className="px-2 py-1 bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 rounded-full">
                {metadata.documents_retrieved} docs
              </span>
            )}
            {metadata.total_time_ms && (
              <span className="text-muted-foreground">
                {metadata.total_time_ms}ms
              </span>
            )}
          </div>
        )}

        {/* Timestamp */}
        <span className="text-xs text-muted-foreground px-1">
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