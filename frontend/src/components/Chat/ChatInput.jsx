import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HiPaperAirplane, HiStop, HiMicrophone, HiPaperClip, HiExclamationCircle } from 'react-icons/hi'
import { cn } from '../../lib/utils'

const ChatInput = ({ 
  value, 
  onChange, 
  onSend, 
  onStop,
  disabled,
  isGenerating,
  placeholder = "Type your message...",
  modelName = "Llama 3.1"
}) => {
  const textareaRef = useRef(null)
  const [isFocused, setIsFocused] = useState(false)

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`
    }
  }, [value])

  const handleSubmit = (e) => {
    e?.preventDefault()
    if (value.trim() && !disabled) {
      onSend(value)
      onChange('')
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const canSend = value.trim() && !disabled

  return (
    <div className="p-4 border-t border-border/50 bg-background/80 backdrop-blur-xl">
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
        <div className={cn(
          "flex items-end gap-2",
          "bg-accent/50 rounded-2xl",
          "border transition-all duration-200",
          isFocused 
            ? "border-violet-500/50 ring-2 ring-violet-500/20" 
            : "border-border/50 hover:border-border"
        )}>
          {/* Attachment Button */}
          <button
            type="button"
            className={cn(
              "p-3 rounded-xl flex-shrink-0",
              "text-muted-foreground hover:text-foreground",
              "hover:bg-accent transition-colors"
            )}
            title="Attach file (coming soon)"
            disabled
          >
            <HiPaperClip className="w-5 h-5" />
          </button>
          
          {/* Text Input */}
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className={cn(
              "flex-1 py-3 bg-transparent",
              "border-none outline-none resize-none",
              "text-foreground placeholder-muted-foreground",
              "text-sm md:text-base",
              "disabled:opacity-50"
            )}
            style={{ minHeight: '24px', maxHeight: '150px' }}
          />

          {/* Voice Button */}
          <button
            type="button"
            className={cn(
              "p-3 rounded-xl flex-shrink-0",
              "text-muted-foreground hover:text-foreground",
              "hover:bg-accent transition-colors"
            )}
            title="Voice input (coming soon)"
            disabled
          >
            <HiMicrophone className="w-5 h-5" />
          </button>
          
          {/* Send/Stop Button */}
          <div className="pr-2 pb-2">
            {isGenerating ? (
              <motion.button
                type="button"
                onClick={onStop}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className={cn(
                  "p-2.5 rounded-xl flex-shrink-0",
                  "bg-red-500 text-white",
                  "hover:bg-red-600 transition-colors",
                  "shadow-lg shadow-red-500/25"
                )}
              >
                <HiStop className="w-5 h-5" />
              </motion.button>
            ) : (
              <motion.button
                type="submit"
                disabled={!canSend}
                whileHover={{ scale: canSend ? 1.05 : 1 }}
                whileTap={{ scale: canSend ? 0.95 : 1 }}
                className={cn(
                  "p-2.5 rounded-xl flex-shrink-0 transition-all duration-200",
                  canSend
                    ? "bg-gradient-to-r from-violet-600 to-cyan-600 text-white shadow-lg shadow-violet-500/25 hover:shadow-xl hover:shadow-violet-500/30"
                    : "bg-muted text-muted-foreground cursor-not-allowed"
                )}
              >
                <HiPaperAirplane className="w-5 h-5 transform rotate-90" />
              </motion.button>
            )}
          </div>
        </div>
        
        {/* Helper Text */}
        <div className="flex items-center justify-between mt-2 px-2">
          <p className="text-xs text-muted-foreground">
            Press <kbd className="px-1.5 py-0.5 rounded bg-accent text-xs">Enter</kbd> to send • <kbd className="px-1.5 py-0.5 rounded bg-accent text-xs">Shift+Enter</kbd> for new line
          </p>
        </div>
        
        {/* AI Disclaimer */}
        <div className="flex items-center justify-center gap-1.5 mt-3 px-2">
          <HiExclamationCircle className="w-3.5 h-3.5 text-muted-foreground/70" />
          <p className="text-xs text-muted-foreground/70 text-center">
            AI can make mistakes. Please verify important information. Powered by <span className="font-medium text-violet-500">{modelName}</span>
          </p>
        </div>
      </form>
    </div>
  )
}

export default ChatInput
