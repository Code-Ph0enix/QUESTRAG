import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useChat } from '../../context/ChatContext'
import MessageBubble from './MessageBubble'
import MessageInput from './MessageInput'
import TypingIndicator from './TypingIndicator'
import { HiSparkles, HiLightningBolt } from 'react-icons/hi'
import { BsRobot } from 'react-icons/bs'
import { cn } from '../../lib/utils'

const ChatContainer = ({ conversationId }) => {
  const { messages, isLoading, sendMessage, error } = useChat()
  const messagesEndRef = useRef(null)
  const messagesContainerRef = useRef(null)
  const [showScrollButton, setShowScrollButton] = useState(false)

  const scrollToBottom = (behavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  // Handle scroll to show/hide scroll button
  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current
      setShowScrollButton(scrollHeight - scrollTop - clientHeight > 100)
    }
  }

  const handleSendMessage = async (message) => {
    try {
      await sendMessage(message, conversationId)
    } catch (error) {
      console.error('Failed to send message:', error)
    }
  }

  return (
    <div className="flex flex-col h-full bg-background relative">
      {/* Messages Area */}
      <div 
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 md:px-6 py-6 min-h-0"
      >
        {messages.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="h-full flex flex-col items-center justify-center"
          >
            {/* Hero Section for Empty State */}
            <div className="text-center max-w-md mx-auto space-y-6">
              {/* Logo/Icon */}
              <motion.div
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', damping: 15 }}
                className="relative inline-block"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-violet-500 to-cyan-500 rounded-2xl blur-xl opacity-30 animate-pulse" />
                <div className="relative w-20 h-20 mx-auto bg-gradient-to-br from-violet-600 to-cyan-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <BsRobot className="w-10 h-10 text-white" />
                </div>
              </motion.div>

              {/* Title */}
              <div>
                <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-violet-600 via-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">
                  QuestRAG Banking AI
                </h1>
                <p className="text-muted-foreground">
                  Your intelligent banking assistant powered by RAG + RL
                </p>
              </div>

              {/* Feature Pills */}
              <div className="flex flex-wrap justify-center gap-2">
                <span className="inline-flex items-center gap-1 px-3 py-1.5 bg-violet-500/10 text-violet-600 dark:text-violet-400 rounded-full text-sm border border-violet-500/20">
                  <HiLightningBolt className="w-4 h-4" />
                  Fast Responses
                </span>
                <span className="inline-flex items-center gap-1 px-3 py-1.5 bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 rounded-full text-sm border border-cyan-500/20">
                  <HiSparkles className="w-4 h-4" />
                  AI-Powered
                </span>
              </div>

              {/* Prompt */}
              <p className="text-sm text-muted-foreground pt-4">
                Type your banking question below to get started
              </p>
            </div>
          </motion.div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-4">
            <AnimatePresence mode="popLayout">
              {messages.map((message, index) => (
                <MessageBubble key={`msg-${index}`} message={message} index={index} />
              ))}
            </AnimatePresence>
            
            {isLoading && <TypingIndicator />}
            
            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center p-3 bg-red-500/10 text-red-500 border border-red-500/20 rounded-lg text-sm"
              >
                {error}
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Scroll to Bottom Button */}
      <AnimatePresence>
        {showScrollButton && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={() => scrollToBottom()}
            className={cn(
              "absolute bottom-24 right-6",
              "p-2 rounded-full",
              "bg-accent border border-border",
              "shadow-lg hover:shadow-xl",
              "transition-shadow z-10"
            )}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </motion.button>
        )}
      </AnimatePresence>

      {/* Input Area */}
      <MessageInput onSend={handleSendMessage} disabled={isLoading} />
    </div>
  )
}

export default ChatContainer
