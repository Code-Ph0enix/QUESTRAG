import { useRef, useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link } from 'react-router-dom'
import { HiMenuAlt2, HiArrowDown, HiSparkles, HiLightningBolt, HiSun, HiMoon } from 'react-icons/hi'
import { BsRobot } from 'react-icons/bs'
import { useChat } from '../../context/ChatContext'
import { useTheme } from '../../context/ThemeContext'
import { cn } from '../../lib/utils'
import ChatMessageItem from './ChatMessageItem'
import ChatInput from './ChatInput'
import Logo from '../shared/Logo'

// Suggestions for new chat
const SUGGESTIONS = [
  "What are the different types of bank accounts available?",
  "How do I open a savings account?",
  "Explain the process of getting a home loan",
  "What is the difference between FD and RD?",
]

const ChatWindow = ({ onToggleSidebar, sidebarOpen }) => {
  const { 
    messages, 
    isLoading, 
    isStreaming,
    error, 
    sendMessage,
    currentConversationId,
    selectedModel
  } = useChat()
  
  // ROLLBACK: Theme toggle was not present before
  const { theme, toggleTheme } = useTheme()
  
  const messagesEndRef = useRef(null)
  const messagesContainerRef = useRef(null)
  const [showScrollButton, setShowScrollButton] = useState(false)
  const [inputValue, setInputValue] = useState('')

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
    if (!message.trim()) return
    
    try {
      setInputValue('')
      await sendMessage(message, currentConversationId)
    } catch (error) {
      console.error('Failed to send message:', error)
    }
  }

  const handleSuggestionClick = (suggestion) => {
    handleSendMessage(suggestion)
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <header className={cn(
        "flex items-center justify-between",
        "px-4 py-3 border-b border-border/50",
        "bg-background/80 backdrop-blur-xl"
      )}>
        <div className="flex items-center gap-3">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onToggleSidebar}
            className={cn(
              "p-2 rounded-lg",
              "hover:bg-accent transition-colors"
            )}
          >
            <HiMenuAlt2 className="w-5 h-5" />
          </motion.button>
          
          {/* OLD CODE (commented for rollback):
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-cyan-600 flex items-center justify-center">
              <BsRobot className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-semibold">QuestRAG AI</h1>
              <p className="text-xs text-muted-foreground">
                {selectedModel.name} • <span className="text-violet-500">{selectedModel.llmModel || 'Llama 3.1 8B'}</span>
              </p>
            </div>
          </div>
          */}
          
          {/* NEW CODE - Clickable Logo redirects to home */}
          <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-cyan-600 flex items-center justify-center">
              <BsRobot className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-semibold">QuestRAG AI</h1>
              {/* ROLLBACK: Original showed selectedModel.name and selectedModel.llmModel */}
              {/* NEW: Show type (Chat/Eval) and model ID */}
              <p className="text-xs text-muted-foreground">
                {selectedModel.type || 'Chat'} • <span className="text-violet-500">{selectedModel.id}</span>
              </p>
            </div>
          </Link>
        </div>

        <div className="flex items-center gap-2">
          {isStreaming && (
            <span className="flex items-center gap-1 text-xs text-violet-500">
              <span className="w-2 h-2 bg-violet-500 rounded-full animate-pulse" />
              Generating...
            </span>
          )}
          
          {/* ROLLBACK: Theme toggle was not present before */}
          {/* NEW: Theme Toggle Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleTheme}
            className={cn(
              "p-2 rounded-lg",
              "hover:bg-accent transition-colors"
            )}
            aria-label="Toggle theme"
          >
            {theme === 'light' ? (
              <HiMoon className="w-5 h-5 text-violet-500" />
            ) : (
              <HiSun className="w-5 h-5 text-amber-400" />
            )}
          </motion.button>
        </div>
      </header>

      {/* Messages Area */}
      <div 
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto"
      >
        {isEmpty ? (
          <div className="h-full flex flex-col items-center justify-center p-6">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center max-w-lg mx-auto space-y-8"
            >
              {/* Logo */}
              <div className="flex justify-center">
                <Logo size="xl" showText={false} animated={true} />
              </div>

              {/* Welcome Text */}
              <div>
                <h2 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-violet-600 via-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">
                  How can I help you today?
                </h2>
                <p className="text-muted-foreground">
                  Ask me anything about banking, finance, or account management
                </p>
              </div>

              {/* Feature Pills */}
              <div className="flex flex-wrap justify-center gap-2">
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-violet-500/10 text-violet-600 dark:text-violet-400 rounded-full text-sm border border-violet-500/20">
                  <HiLightningBolt className="w-4 h-4" />
                  Fast Responses
                </span>
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 rounded-full text-sm border border-cyan-500/20">
                  <HiSparkles className="w-4 h-4" />
                  RAG Powered
                </span>
              </div>

              {/* Suggestions */}
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">Try asking:</p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {SUGGESTIONS.map((suggestion, index) => (
                    <motion.button
                      key={index}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className={cn(
                        "p-3 rounded-xl text-left text-sm",
                        "bg-accent/50 hover:bg-accent",
                        "border border-border/50 hover:border-violet-500/30",
                        "transition-all duration-200"
                      )}
                    >
                      {suggestion}
                    </motion.button>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-4">
            <AnimatePresence mode="popLayout">
              {messages.map((message, index) => (
                <ChatMessageItem
                  key={message.id || index}
                  message={message}
                  index={index}
                />
              ))}
            </AnimatePresence>

            {/* Typing Indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-3"
              >
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-cyan-600 flex items-center justify-center flex-shrink-0">
                  <BsRobot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-accent rounded-2xl rounded-tl-md px-4 py-3">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-violet-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Error Message */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="px-4 py-2"
          >
            <div className="max-w-3xl mx-auto p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-sm">
              {error}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

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
              "transition-shadow"
            )}
          >
            <HiArrowDown className="w-5 h-5" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Input Area */}
      {/* OLD CODE (commented for rollback):
      <ChatInput
        value={inputValue}
        onChange={setInputValue}
        onSend={handleSendMessage}
        disabled={isLoading}
        placeholder="Ask me anything about banking..."
        modelName={selectedModel.llmModel || 'Llama 3.1 8B'}
      />
      */}
      {/* NEW CODE - Updated placeholder, removed fallback */}
      <ChatInput
        value={inputValue}
        onChange={setInputValue}
        onSend={handleSendMessage}
        disabled={isLoading}
        placeholder="Type your question here..."
        modelName={selectedModel.llmModel}
      />
    </div>
  )
}

export default ChatWindow
