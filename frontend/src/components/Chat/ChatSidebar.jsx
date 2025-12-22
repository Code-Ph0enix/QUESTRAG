import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link, useNavigate } from 'react-router-dom'
import { 
  HiPlus, 
  HiChat, 
  HiTrash, 
  HiSearch, 
  HiX, 
  HiHome,
  HiCog,
  HiLogout,
  HiChevronDown
} from 'react-icons/hi'
import { useChat, AI_MODELS } from '../../context/ChatContext'
import { useAuth } from '../../context/AuthContext'
import { cn } from '../../lib/utils'
import Logo from '../shared/Logo'

const ChatSidebar = ({ isOpen, onClose, isMobile = false }) => {
  const { 
    conversations, 
    loadConversations, 
    currentConversationId,
    selectConversation,
    startNewChat,
    deleteConversation,
    isLoadingConversations,
    selectedModel,
    changeModel,
    availableModels
  } = useChat()
  
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  
  const [searchQuery, setSearchQuery] = useState('')
  const [filteredConversations, setFilteredConversations] = useState([])
  const [showModelSelector, setShowModelSelector] = useState(false)
  const [showUserMenu, setShowUserMenu] = useState(false)
  const [deletingId, setDeletingId] = useState(null)
  
  useEffect(() => {
    loadConversations()
  }, [loadConversations])

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
    startNewChat()
    if (window.innerWidth < 1024) {
      onClose()
    }
  }

  const handleSelectConversation = (conversationId) => {
    selectConversation(conversationId)
    if (window.innerWidth < 1024) {
      onClose()
    }
  }

  const handleDeleteConversation = async (e, conversationId) => {
    e.stopPropagation()
    setDeletingId(conversationId)
    await deleteConversation(conversationId)
    setDeletingId(null)
  }

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    const now = new Date()
    const diff = now - date
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    
    if (days === 0) return 'Today'
    if (days === 1) return 'Yesterday'
    if (days < 7) return `${days} days ago`
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }

  // Group conversations by date
  const groupedConversations = filteredConversations.reduce((groups, conv) => {
    const dateKey = formatDate(conv.created_at)
    if (!groups[dateKey]) groups[dateKey] = []
    groups[dateKey].push(conv)
    return groups
  }, {})
  
  return (
    <>
      {/* Backdrop for mobile */}
      <AnimatePresence>
        {isOpen && isMobile && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          />
        )}
      </AnimatePresence>
      
      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ 
          x: isOpen ? 0 : (isMobile ? '-100%' : '-100%'),
          opacity: isOpen ? 1 : 0
        }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className={cn(
          "h-full w-72 flex flex-col",
          "bg-background/95 backdrop-blur-xl",
          "border-r border-border/50",
          isMobile ? "fixed inset-y-0 left-0 z-50 shadow-2xl" : "absolute inset-y-0 left-0",
          !isOpen && "pointer-events-none"
        )}
      >
        {/* Header */}
        <div className="p-4 border-b border-border/50">
          <div className="flex items-center justify-between mb-4">
            {/* ROLLBACK: Original was just <Logo size="sm" showText={true} animated={false} /> */}
            {/* NEW: Wrapped in Link to redirect to home page */}
            <Link to="/" className="hover:opacity-80 transition-opacity">
              <Logo size="sm" showText={true} animated={false} />
            </Link>
            <button
              onClick={onClose}
              className="lg:hidden p-2 hover:bg-accent rounded-lg transition-colors"
            >
              <HiX className="w-5 h-5" />
            </button>
          </div>
          
          {/* New Chat Button */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleNewChat}
            className={cn(
              "w-full flex items-center justify-center gap-2",
              "px-4 py-2.5 rounded-xl",
              "bg-gradient-to-r from-violet-600 via-blue-600 to-cyan-600",
              "text-white font-medium text-sm",
              "shadow-lg shadow-violet-500/25",
              "hover:shadow-xl hover:shadow-violet-500/30",
              "transition-all duration-200"
            )}
          >
            <HiPlus className="w-5 h-5" />
            New Chat
          </motion.button>
        </div>

        {/* Model Selector */}
        <div className="px-4 py-3 border-b border-border/50">
          <button
            onClick={() => setShowModelSelector(!showModelSelector)}
            className={cn(
              "w-full flex items-center justify-between",
              "px-3 py-2 rounded-lg",
              "bg-accent/50 hover:bg-accent",
              "text-sm transition-colors"
            )}
          >
            <div className="flex items-center gap-2">
              <span>{selectedModel.icon}</span>
              <span className="font-medium">{selectedModel.name}</span>
            </div>
            <HiChevronDown className={cn(
              "w-4 h-4 transition-transform",
              showModelSelector && "rotate-180"
            )} />
          </button>
          
          <AnimatePresence>
            {showModelSelector && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="pt-2 space-y-1">
                  {availableModels.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => {
                        changeModel(model.id)
                        setShowModelSelector(false)
                      }}
                      className={cn(
                        "w-full flex items-start gap-2 p-2 rounded-lg text-left",
                        "hover:bg-accent transition-colors",
                        selectedModel.id === model.id && "bg-accent"
                      )}
                    >
                      <span className="text-lg">{model.icon}</span>
                      <div>
                        <p className="text-sm font-medium">{model.name}</p>
                        <p className="text-xs text-muted-foreground">{model.description}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Search Bar */}
        <div className="px-4 py-3">
          <div className="relative">
            <HiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search conversations..."
              className={cn(
                "w-full pl-10 pr-4 py-2",
                "bg-accent/50 border border-border/50 rounded-lg",
                "text-sm placeholder-muted-foreground",
                "focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50",
                "transition-all"
              )}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <HiX className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
        
        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto px-3 py-2">
          {isLoadingConversations ? (
            <div className="flex items-center justify-center py-8">
              <div className="w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : filteredConversations.length === 0 ? (
            <div className="text-center py-8">
              <HiChat className="w-12 h-12 mx-auto mb-3 text-muted-foreground/40" />
              <p className="text-sm font-medium text-muted-foreground">
                {searchQuery ? 'No matches found' : 'No conversations yet'}
              </p>
              <p className="text-xs text-muted-foreground/60 mt-1">
                {searchQuery ? 'Try a different search term' : 'Start a new chat to begin'}
              </p>
            </div>
          ) : (
            Object.entries(groupedConversations).map(([date, convs]) => (
              <div key={date} className="mb-4">
                <p className="text-xs font-medium text-muted-foreground px-2 mb-2">{date}</p>
                <div className="space-y-1">
                  {convs.map((conv) => (
                    <motion.button
                      key={conv.conversation_id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                      onClick={() => handleSelectConversation(conv.conversation_id)}
                      className={cn(
                        "w-full p-3 rounded-lg text-left",
                        "transition-all duration-200 group",
                        currentConversationId === conv.conversation_id
                          ? "bg-gradient-to-r from-violet-500/20 to-cyan-500/20 border border-violet-500/30"
                          : "hover:bg-accent border border-transparent"
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0 pr-2">
                          <p className="text-sm font-medium truncate">
                            {conv.title || 'New Conversation'}
                          </p>
                          <p className="text-xs text-muted-foreground truncate mt-0.5">
                            {conv.message_count || 0} messages
                          </p>
                        </div>
                        <button
                          onClick={(e) => handleDeleteConversation(e, conv.conversation_id)}
                          disabled={deletingId === conv.conversation_id}
                          className={cn(
                            "p-1.5 rounded-lg transition-all",
                            "opacity-0 group-hover:opacity-100",
                            "hover:bg-red-500/20 hover:text-red-500",
                            deletingId === conv.conversation_id && "opacity-100"
                          )}
                          title="Delete conversation"
                        >
                          {deletingId === conv.conversation_id ? (
                            <div className="w-4 h-4 border-2 border-red-500 border-t-transparent rounded-full animate-spin" />
                          ) : (
                            <HiTrash className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </motion.button>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer - User Profile & Actions */}
        <div className="p-4 border-t border-border/50 space-y-2">
          {/* Home Link */}
          <Link
            to="/"
            className={cn(
              "flex items-center gap-3 px-3 py-2 rounded-lg",
              "text-sm text-muted-foreground",
              "hover:bg-accent hover:text-foreground",
              "transition-colors"
            )}
          >
            <HiHome className="w-4 h-4" />
            Back to Home
          </Link>

          {/* User Menu */}
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2 rounded-lg",
                "hover:bg-accent transition-colors"
              )}
            >
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center text-white text-sm font-medium">
                {user?.full_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
              </div>
              <div className="flex-1 text-left min-w-0">
                <p className="text-sm font-medium truncate">{user?.full_name || 'User'}</p>
                <p className="text-xs text-muted-foreground truncate">{user?.email}</p>
              </div>
              <HiChevronDown className={cn(
                "w-4 h-4 text-muted-foreground transition-transform",
                showUserMenu && "rotate-180"
              )} />
            </button>

            <AnimatePresence>
              {showUserMenu && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  className={cn(
                    "absolute bottom-full left-0 right-0 mb-2",
                    "bg-popover border border-border rounded-lg shadow-xl",
                    "overflow-hidden"
                  )}
                >
                  <button
                    onClick={() => {/* TODO: Settings */}}
                    className="w-full flex items-center gap-3 px-4 py-3 hover:bg-accent transition-colors text-sm"
                  >
                    <HiCog className="w-4 h-4" />
                    Settings
                  </button>
                  <button
                    onClick={handleLogout}
                    className="w-full flex items-center gap-3 px-4 py-3 hover:bg-accent transition-colors text-sm text-red-500"
                  >
                    <HiLogout className="w-4 h-4" />
                    Sign Out
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.aside>
    </>
  )
}

export default ChatSidebar
