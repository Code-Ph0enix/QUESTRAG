import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { HiPaperAirplane, HiPlus } from 'react-icons/hi'

const MessageInput = ({ onSend, disabled }) => {
  const [message, setMessage] = useState('')
  const textareaRef = useRef(null)

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`
    }
  }, [message])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (message.trim() && !disabled) {
      onSend(message)
      setMessage('')
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="p-4 md:p-6 bg-gray-900 border-t border-gray-800"
    >
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
        <div className="flex items-end gap-3 bg-gray-800 rounded-2xl px-4 py-3 border border-gray-700 focus-within:border-blue-500/50 focus-within:ring-2 focus-within:ring-blue-500/20 transition-all">
          {/* Attachment Button */}
          <button
            type="button"
            className="p-2 hover:bg-gray-700 rounded-lg flex-shrink-0 transition-colors"
            title="Add attachment (coming soon)"
          >
            <HiPlus className="w-5 h-5 text-gray-400" />
          </button>
          
          {/* Text Input */}
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask me anything about banking..."
            disabled={disabled}
            rows={1}
            className="flex-1 bg-transparent border-none outline-none resize-none text-white placeholder-gray-500 text-sm md:text-base py-1"
            style={{ minHeight: '24px', maxHeight: '120px' }}
          />
          
          {/* Send Button */}
          <motion.button
            type="submit"
            disabled={disabled || !message.trim()}
            whileHover={{ scale: disabled || !message.trim() ? 1 : 1.05 }}
            whileTap={{ scale: disabled || !message.trim() ? 1 : 0.95 }}
            className={`p-2.5 rounded-xl flex-shrink-0 transition-all duration-200 ${
              disabled || !message.trim()
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:shadow-lg hover:shadow-blue-500/30'
            }`}
          >
            <HiPaperAirplane className="w-5 h-5 transform rotate-90" />
          </motion.button>
        </div>
        
        {/* Helper Text */}
        <p className="text-xs text-gray-600 text-center mt-2">
          Press Enter to send • Shift+Enter for new line
        </p>
      </form>
    </motion.div>
  )
}

export default MessageInput
