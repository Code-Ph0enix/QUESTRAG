import { createContext, useContext, useState } from 'react'
import { chatAPI } from '../services/api'

const ChatContext = createContext()

export const useChat = () => {
  const context = useContext(ChatContext)
  if (!context) {
    throw new Error('useChat must be used within ChatProvider')
  }
  return context
}

export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [conversations, setConversations] = useState([])
  const [currentConversationId, setCurrentConversationId] = useState(null)

  const sendMessage = async (message, conversationId = null) => {
    try {
      setIsLoading(true)
      setError(null)

      console.log('📤 Sending message:', { message, conversationId })
      console.log('🔑 Token in localStorage:', !!localStorage.getItem('token'))

      // Add user message immediately
      const userMessage = {
        role: 'user',
        content: message,
        timestamp: new Date().toISOString(),
      }
      setMessages(prev => [...prev, userMessage])

      // Handle conversation ID - if 'new' or null, pass null to create new conversation
      let convId = conversationId === 'new' ? null : conversationId
      // Use current conversation ID if not explicitly provided
      if (!convId && currentConversationId && currentConversationId !== 'new') {
        convId = currentConversationId
      }

      console.log('📤 Sending to API with convId:', convId)

      // Call API - pass null for new conversations
      const response = await chatAPI.sendMessage(message, convId)
      
      console.log('✅ Response received:', response)

      // Save conversation ID for next message
      if (response.conversation_id) {
        setCurrentConversationId(response.conversation_id)
      }

      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: response.response,
        timestamp: response.timestamp || new Date().toISOString(),
        metadata: {
          policy_action: response.policy_action,
          policy_confidence: response.policy_confidence,
          documents_retrieved: response.documents_retrieved,
          top_doc_score: response.top_doc_score,
          total_time_ms: response.total_time_ms
        }
      }
      setMessages(prev => [...prev, assistantMessage])

      return response
    } catch (err) {
      console.error('❌ Chat error:', err)
      console.error('Error response:', err.response?.data)
      console.error('Error status:', err.response?.status)
      
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to send message'
      setError(errorMessage)
      
      // Remove the user message if request failed
      setMessages(prev => prev.slice(0, -1))
      
      throw new Error(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }

  const loadConversations = async () => {
    try {
      const data = await chatAPI.getConversations()
      setConversations(data.conversations || [])
    } catch (err) {
      console.log('Conversation history not available yet')
      setConversations([])
    }
  }

  const clearMessages = () => {
    setMessages([])
    setCurrentConversationId(null)
  }

  const value = {
    messages,
    isLoading,
    error,
    conversations,
    currentConversationId,
    sendMessage,
    loadConversations,
    clearMessages,
  }

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  )
}