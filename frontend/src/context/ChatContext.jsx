import { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { chatAPI } from '../services/api'

const ChatContext = createContext()

export const useChat = () => {
  const context = useContext(ChatContext)
  if (!context) {
    throw new Error('useChat must be used within ChatProvider')
  }
  return context
}

// Available AI Models - Maps to backend Llama models via Groq
// Based on backend/app/config.py: GROQ_CHAT_MODEL and GROQ_EVAL_MODEL
// 
// OLD CODE (commented for rollback):
// export const AI_MODELS = [
//   { id: 'questrag-default', name: 'QuestRAG Default', description: 'Optimized for banking queries with RAG + RL', icon: '🏦', llmModel: 'Llama 3.1 8B', provider: 'Groq' },
//   { id: 'questrag-fast', name: 'QuestRAG Fast', description: 'Faster responses, fewer document retrievals', icon: '⚡', llmModel: 'Llama 3.1 8B Instant', provider: 'Groq' },
//   { id: 'questrag-detailed', name: 'QuestRAG Detailed', description: 'More comprehensive answers with citations', icon: '📚', llmModel: 'Llama 3.3 70B', provider: 'Groq' },
// ]
//
// NEW CODE - Using exact backend model names from config.py
// ROLLBACK: Previous version didn't have 'type' field
export const AI_MODELS = [
  { 
    id: 'llama-3.1-8b-instant', 
    name: 'QuestRAG Default', 
    description: 'Optimized for banking queries with RAG + RL',
    icon: '🏦',
    llmModel: 'llama-3.1-8b-instant',
    provider: 'Groq',
    type: 'Chat'  // NEW: Added type field
  },
  { 
    id: 'llama-3.1-8b-instant-fast', 
    name: 'QuestRAG Fast', 
    description: 'Faster responses, fewer document retrievals',
    icon: '⚡',
    llmModel: 'llama-3.1-8b-instant',
    provider: 'Groq',
    type: 'Chat'  // NEW: Added type field
  },
  { 
    id: 'llama-3.3-70b-versatile', 
    name: 'QuestRAG Detailed', 
    description: 'More comprehensive answers with citations',
    icon: '📚',
    llmModel: 'llama-3.3-70b-versatile',
    provider: 'Groq',
    type: 'Eval'  // NEW: Added type field (this is the eval model)
  },
]

export const ChatProvider = ({ children }) => {
  // Messages state
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  
  // Conversations state
  const [conversations, setConversations] = useState([])
  const [currentConversationId, setCurrentConversationId] = useState(null)
  const [isLoadingConversations, setIsLoadingConversations] = useState(false)
  
  // AI Model state
  const [selectedModel, setSelectedModel] = useState(AI_MODELS[0])

  // Load conversation history when conversation changes
  const loadConversationHistory = useCallback(async (conversationId) => {
    if (!conversationId || conversationId === 'new') {
      setMessages([])
      return
    }
    
    try {
      setIsLoading(true)
      const data = await chatAPI.getHistory(conversationId)
      
      // Transform messages to our format
      const formattedMessages = (data.messages || []).map((msg, idx) => ({
        id: msg.id || `msg-${Date.now()}-${idx}`,
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp || new Date().toISOString(),
        metadata: msg.metadata || {}
      }))
      
      setMessages(formattedMessages)
      setCurrentConversationId(conversationId)
    } catch (err) {
      console.error('Failed to load conversation history:', err)
      setError('Failed to load conversation history')
      setMessages([])
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Load all conversations
  const loadConversations = useCallback(async () => {
    try {
      setIsLoadingConversations(true)
      const data = await chatAPI.getConversations()
      
      // ROLLBACK: Original was: setConversations(data.conversations || [])
      // FIX: Map 'id' to 'conversation_id' since backend returns 'id' but frontend expects 'conversation_id'
      const mappedConversations = (data.conversations || []).map(conv => ({
        ...conv,
        conversation_id: conv.id || conv.conversation_id // Map id to conversation_id
      }))
      setConversations(mappedConversations)
    } catch (err) {
      console.log('Conversations not available:', err.message)
      setConversations([])
    } finally {
      setIsLoadingConversations(false)
    }
  }, [])

  // Send message
  const sendMessage = useCallback(async (content, conversationId = null) => {
    if (!content.trim()) return

    const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    
    try {
      setIsLoading(true)
      setIsStreaming(true)
      setError(null)

      // Add user message immediately
      const userMessage = {
        id: `user-${messageId}`,
        role: 'user',
        content: content.trim(),
        timestamp: new Date().toISOString(),
      }
      
      setMessages(prev => [...prev, userMessage])

      // Determine conversation ID
      let convId = conversationId === 'new' ? null : conversationId
      if (!convId && currentConversationId && currentConversationId !== 'new') {
        convId = currentConversationId
      }

      console.log('📤 Sending message:', { content: content.trim(), convId, model: selectedModel.id })

      // Call API
      const response = await chatAPI.sendMessage(content.trim(), convId)
      
      console.log('✅ Response received:', response)

      // Update conversation ID
      if (response.conversation_id) {
        setCurrentConversationId(response.conversation_id)
      }

      // Add assistant response
      const assistantMessage = {
        id: `assistant-${messageId}`,
        role: 'assistant',
        content: response.response,
        timestamp: response.timestamp || new Date().toISOString(),
        metadata: {
          policy_action: response.policy_action,
          policy_confidence: response.policy_confidence,
          documents_retrieved: response.documents_retrieved,
          top_doc_score: response.top_doc_score,
          total_time_ms: response.total_time_ms,
          model: selectedModel.id
        }
      }
      
      setMessages(prev => [...prev, assistantMessage])
      
      // Refresh conversations list
      loadConversations()

      return response
    } catch (err) {
      console.error('❌ Chat error:', err)
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to send message'
      setError(errorMessage)
      
      // Remove the user message on error
      setMessages(prev => prev.slice(0, -1))
      
      throw new Error(errorMessage)
    } finally {
      setIsLoading(false)
      setIsStreaming(false)
    }
  }, [currentConversationId, selectedModel, loadConversations])

  // Delete conversation
  const deleteConversation = useCallback(async (conversationId) => {
    try {
      await chatAPI.deleteConversation(conversationId)
      setConversations(prev => prev.filter(c => c.conversation_id !== conversationId))
      
      // If deleted current conversation, reset
      if (currentConversationId === conversationId) {
        setCurrentConversationId(null)
        setMessages([])
      }
      
      return true
    } catch (err) {
      console.error('Failed to delete conversation:', err)
      setError('Failed to delete conversation')
      return false
    }
  }, [currentConversationId])

  // Start new chat
  const startNewChat = useCallback(() => {
    setMessages([])
    setCurrentConversationId('new')
    setError(null)
  }, [])

  // Clear messages
  const clearMessages = useCallback(() => {
    setMessages([])
    setCurrentConversationId(null)
    setError(null)
  }, [])

  // Select conversation
  const selectConversation = useCallback((conversationId) => {
    if (conversationId === 'new') {
      startNewChat()
    } else {
      loadConversationHistory(conversationId)
    }
  }, [startNewChat, loadConversationHistory])

  // Change AI model
  const changeModel = useCallback((modelId) => {
    const model = AI_MODELS.find(m => m.id === modelId)
    if (model) {
      setSelectedModel(model)
    }
  }, [])

  const value = {
    // Messages
    messages,
    isLoading,
    isStreaming,
    error,
    sendMessage,
    clearMessages,
    setMessages,
    
    // Conversations
    conversations,
    currentConversationId,
    isLoadingConversations,
    loadConversations,
    loadConversationHistory,
    deleteConversation,
    startNewChat,
    selectConversation,
    
    // AI Model
    selectedModel,
    changeModel,
    availableModels: AI_MODELS,
  }

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  )
}