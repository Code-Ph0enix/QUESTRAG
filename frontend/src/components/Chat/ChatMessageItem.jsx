import { forwardRef, useState } from 'react'
import { motion } from 'framer-motion'
import { HiUser, HiClipboard, HiCheck, HiThumbUp, HiThumbDown } from 'react-icons/hi'
import { BsRobot, BsCpu } from 'react-icons/bs'
import ReactMarkdown from 'react-markdown'
import { cn } from '../../lib/utils'
import { useChat } from '../../context/ChatContext'

// Policy Badge Component
const PolicyBadge = ({ action, confidence }) => {
  const isFetch = action === 'FETCH'
  
  return (
    <span className={cn(
      "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
      isFetch 
        ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20" 
        : "bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20"
    )}>
      <span className={cn(
        "w-1.5 h-1.5 rounded-full",
        isFetch ? "bg-emerald-500" : "bg-amber-500"
      )} />
      {action}
      {confidence && (
        <span className="text-muted-foreground">
          ({(confidence * 100).toFixed(0)}%)
        </span>
      )}
    </span>
  )
}

const ChatMessageItem = forwardRef(({ message, index }, ref) => {
  const [copied, setCopied] = useState(false)
  const [feedback, setFeedback] = useState(null)
  const { selectedModel } = useChat()
  
  const isUser = message.role === 'user'
  const metadata = message.metadata || {}

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleFeedback = (type) => {
    setFeedback(type)
    // TODO: Send feedback to backend
  }

  const formattedTime = new Date(message.timestamp).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit'
  })

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ delay: index * 0.02, duration: 0.3 }}
      className={cn(
        "flex gap-3",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div className={cn(
        "flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center",
        isUser
          ? "bg-gradient-to-br from-violet-500 to-blue-500"
          : "bg-gradient-to-br from-violet-600 to-cyan-600"
      )}>
        {isUser ? (
          <HiUser className="w-4 h-4 text-white" />
        ) : (
          <BsRobot className="w-4 h-4 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={cn(
        "flex flex-col gap-1 max-w-[80%]",
        isUser ? "items-end" : "items-start"
      )}>
        {/* Sender & Time */}
        <div className={cn(
          "flex items-center gap-2 text-xs text-muted-foreground px-1",
          isUser && "flex-row-reverse"
        )}>
          <span className="font-medium">{isUser ? 'You' : 'QuestRAG'}</span>
          <span>•</span>
          <span>{formattedTime}</span>
        </div>

        {/* Message Bubble */}
        <div className={cn(
          "relative group px-4 py-3 rounded-2xl",
          isUser
            ? "bg-gradient-to-br from-violet-600 to-blue-600 text-white rounded-tr-md"
            : "bg-accent text-foreground rounded-tl-md"
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
                  ? <code className={cn(
                      "px-1.5 py-0.5 rounded text-xs font-mono",
                      isUser ? "bg-white/20" : "bg-muted"
                    )}>{children}</code>
                  : <code className={cn(
                      "block p-3 rounded-lg text-xs font-mono overflow-x-auto my-2",
                      isUser ? "bg-white/10" : "bg-muted"
                    )}>{children}</code>
              ),
              strong: ({ children }) => (
                <strong className={cn(
                  "font-semibold",
                  isUser ? "text-white" : "text-violet-600 dark:text-violet-400"
                )}>{children}</strong>
              ),
              a: ({ children, href }) => (
                <a 
                  href={href} 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="text-cyan-400 hover:underline"
                >
                  {children}
                </a>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>

          {/* Copy Button - Show on hover for assistant messages */}
          {!isUser && (
            <div className="absolute -bottom-8 left-0 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
              <button
                onClick={handleCopy}
                className={cn(
                  "p-1.5 rounded-lg",
                  "hover:bg-accent transition-colors",
                  "text-muted-foreground hover:text-foreground"
                )}
                title="Copy message"
              >
                {copied ? (
                  <HiCheck className="w-4 h-4 text-green-500" />
                ) : (
                  <HiClipboard className="w-4 h-4" />
                )}
              </button>
              <button
                onClick={() => handleFeedback('up')}
                className={cn(
                  "p-1.5 rounded-lg transition-colors",
                  feedback === 'up' 
                    ? "text-green-500 bg-green-500/10" 
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"
                )}
                title="Good response"
              >
                <HiThumbUp className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleFeedback('down')}
                className={cn(
                  "p-1.5 rounded-lg transition-colors",
                  feedback === 'down' 
                    ? "text-red-500 bg-red-500/10" 
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"
                )}
                title="Poor response"
              >
                <HiThumbDown className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>

        {/* Metadata (for assistant messages) */}
        {!isUser && (
          <div className="flex flex-wrap items-center gap-2 text-xs px-1 mt-1">
            {/* LLM Model Badge */}
            {/* OLD CODE (commented for rollback):
            <span className="inline-flex items-center gap-1 px-2 py-1 bg-violet-500/10 text-violet-600 dark:text-violet-400 border border-violet-500/20 rounded-full">
              <BsCpu className="w-3 h-3" />
              {selectedModel?.llmModel || 'Llama 3.1 8B'}
            </span>
            */}
            {/* NEW CODE - Removed fallback, using model from context */}
            <span className="inline-flex items-center gap-1 px-2 py-1 bg-violet-500/10 text-violet-600 dark:text-violet-400 border border-violet-500/20 rounded-full">
              <BsCpu className="w-3 h-3" />
              {selectedModel?.llmModel}
            </span>
            
            {metadata.policy_action && (
              <PolicyBadge 
                action={metadata.policy_action} 
                confidence={metadata.policy_confidence}
              />
            )}
            {metadata.documents_retrieved > 0 && (
              <span className="px-2 py-1 bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 rounded-full">
                {metadata.documents_retrieved} docs retrieved
              </span>
            )}
            {metadata.total_time_ms && (
              <span className="text-muted-foreground">
                {metadata.total_time_ms.toFixed(0)}ms
              </span>
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
})

ChatMessageItem.displayName = 'ChatMessageItem'

export default ChatMessageItem
