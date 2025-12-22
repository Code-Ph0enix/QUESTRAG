import axios from 'axios'

// Get base URL from environment variable
// For local development: http://localhost:8000
// For production: https://eeshanyaj-questrag-backend.hf.space
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

console.log('🔗 API Base URL:', API_BASE_URL)

// Create axios instance with CORS-friendly configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  },
  // Enable credentials for CORS (cookies, auth headers)
  withCredentials: false // Set to false for public APIs, true if you need cookies
})

// Add token to every request
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Handle 401 errors (token expired)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// // ============================================================================
// // AUTH API
// // ============================================================================

// export const authAPI = {
//   register: async (username, email, password) => {
//     const response = await api.post('/api/v1/auth/register', {
//       username,
//       email,
//       password
//     })
//     return response.data
//   },

//   login: async (username, password) => {
//     const response = await api.post('/api/v1/auth/login', {
//       username,
//       password
//     })
//     return response.data
//   },

//   me: async () => {
//     const response = await api.get('/api/v1/auth/me')
//     return response.data
//   },

//   logout: async () => {
//     const response = await api.post('/api/v1/auth/logout')
//     return response.data
//   }
// }

// ============================================================================
// AUTH API
// ============================================================================

export const authAPI = {
  register: async (email, password, fullName) => {
    const response = await api.post('/api/v1/auth/register', {
      email,
      password,
      full_name: fullName  // ✅ Changed to full_name to match backend
    })
    return response.data
  },

  login: async (email, password) => {
    const response = await api.post('/api/v1/auth/login', {
      email,
      password
    })
    return response.data
  },

  me: async () => {
    const response = await api.get('/api/v1/auth/me')
    return response.data
  },

  logout: async () => {
    const response = await api.post('/api/v1/auth/logout')
    return response.data
  }
}


// ============================================================================
// CHAT API
// ============================================================================

export const chatAPI = {
  sendMessage: async (query, conversationId = null) => {
    // Build request body - only include conversation_id if it's a valid ID
    const requestBody = { query }
    if (conversationId && conversationId !== 'new') {
      requestBody.conversation_id = conversationId
    }
    console.log('🔗 API Request body:', requestBody)
    const response = await api.post('/api/v1/chat/', requestBody)
    return response.data
  },

  getHistory: async (conversationId) => {
    // OLD CODE (commented for rollback):
    // const response = await api.get(`/api/v1/chat/conversations/${conversationId}`)
    // NEW CODE - Fixed: endpoint is /conversation/ (singular) not /conversations/
    const response = await api.get(`/api/v1/chat/conversation/${conversationId}`)
    return response.data
  },

  getConversations: async () => {
    const response = await api.get('/api/v1/chat/conversations')
    return response.data
  },

  deleteConversation: async (conversationId) => {
    const response = await api.delete(`/api/v1/chat/conversation/${conversationId}`)
    return response.data
  }
}

// Export default object with all APIs
export default {
  auth: authAPI,
  chat: chatAPI
}