import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HiPlus, HiChat, HiTrash, HiSearch, HiX } from 'react-icons/hi'
import { useChat } from '../../context/ChatContext'
import Button from '../UI/Button'

const Sidebar = ({ isOpen, onClose, onSelectConversation, currentConversationId }) => {
  const { conversations, loadConversations, clearMessages } = useChat()
  const [searchQuery, setSearchQuery] = useState('')
  const [filteredConversations, setFilteredConversations] = useState([])
  
  useEffect(() => {
    loadConversations()
  }, [])

  // Filter conversations based on search
  useEffect(() => {
    if (searchQuery.trim()) {
      const filtered = conversations.filter(conv => 
        conv.title?.toLowerCase().includes(searchQuery.toLowerCase())
      )
      setFilteredConversations(filtered)
    } else {
      setFilteredConversations(conversations)
    }
  }, [searchQuery, conversations])

  const handleNewChat = () => {
    clearMessages()
    onSelectConversation('new')
    if (window.innerWidth < 1024) {
      onClose()
    }
  }
  
  const sidebarVariants = {
    open: { x: 0 },
    closed: { x: '-100%' },
  }
  
  return (
    <>
      {/* Backdrop for mobile */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
          />
        )}
      </AnimatePresence>
      
      {/* Sidebar */}
      <motion.aside
        initial="closed"
        animate={isOpen ? 'open' : 'closed'}
        variants={sidebarVariants}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className="fixed lg:relative inset-y-0 left-0 z-50 w-80 bg-gray-950 border-r border-gray-800 flex flex-col shadow-2xl lg:shadow-none"
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-800">
          <Button
            variant="primary"
            className="w-full"
            icon={<HiPlus className="w-5 h-5" />}
            onClick={handleNewChat}
          >
            New Chat
          </Button>
        </div>

        {/* Search Bar */}
        <div className="px-4 py-3">
          <div className="relative">
            <HiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500 w-4 h-4" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search chats..."
              className="w-full pl-10 pr-4 py-2.5 bg-gray-800 border border-gray-700 rounded-xl text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-300"
              >
                <HiX className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
        
        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
          {filteredConversations.length === 0 ? (
            <div className="text-center text-gray-500 mt-8">
              <HiChat className="w-12 h-12 mx-auto mb-3 opacity-40" />
              <p className="text-sm font-medium">
                {searchQuery ? 'No matches found' : 'No conversations yet'}
              </p>
              <p className="text-xs text-gray-600 mt-1">
                {searchQuery ? 'Try a different search term' : 'Start a new chat to begin'}
              </p>
            </div>
          ) : (
            filteredConversations.map((conv) => (
              <motion.button
                key={conv.conversation_id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                onClick={() => {
                  onSelectConversation(conv.conversation_id)
                  if (window.innerWidth < 1024) {
                    onClose()
                  }
                }}
                className={`w-full p-3 rounded-xl text-left transition-all duration-200 group ${
                  currentConversationId === conv.conversation_id
                    ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30 text-white'
                    : 'hover:bg-gray-800/70 text-gray-300 border border-transparent'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0 pr-2">
                    <p className="text-sm font-medium truncate">
                      {conv.title || 'New Conversation'}
                    </p>
                    <p className="text-xs text-gray-500 truncate mt-0.5">
                      {new Date(conv.created_at).toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      // TODO: Implement delete conversation
                    }}
                    className="p-1.5 opacity-0 group-hover:opacity-100 hover:bg-red-500/20 rounded-lg transition-all"
                    title="Delete conversation"
                  >
                    <HiTrash className="w-4 h-4 text-red-400" />
                  </button>
                </div>
              </motion.button>
            ))
          )}
        </div>
        
        {/* Footer */}
        <div className="p-4 border-t border-gray-800">
          <div className="flex items-center justify-center gap-2 text-xs text-gray-500">
            <span className="inline-block w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            <span>Powered by RAG + RL</span>
          </div>
        </div>
      </motion.aside>
    </>
  )
}

export default Sidebar
