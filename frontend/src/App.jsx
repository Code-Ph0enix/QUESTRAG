import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { AuthProvider } from './context/AuthContext'
import { ChatProvider } from './context/ChatContext'
import { ThemeProvider } from './context/ThemeContext'
import ProtectedRoute from './components/Auth/ProtectedRoute'
import Login from './components/Auth/Login'
import Signup from './components/Auth/Signup'
import ChatSidebar from './components/Chat/ChatSidebar'
import ChatWindow from './components/Chat/ChatWindow'
import HomePage from './pages/HomePage'

// Main Chat Page Component
function ChatPage() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isMobile, setIsMobile] = useState(false)

  // Handle responsive sidebar
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 1024
      setIsMobile(mobile)
      if (mobile) {
        setSidebarOpen(false)
      }
    }
    
    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return (
    <ChatProvider>
      <div className="flex h-screen bg-background text-foreground overflow-hidden">
        {/* Sidebar - Fixed on mobile, collapsible on desktop */}
        <div 
          className={`
            ${isMobile ? 'fixed inset-y-0 left-0 z-50' : 'relative'}
            ${!isMobile && !sidebarOpen ? 'w-0' : 'w-72'}
            transition-all duration-300 ease-in-out
            flex-shrink-0
          `}
        >
          <ChatSidebar 
            isOpen={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
            isMobile={isMobile}
          />
        </div>

        {/* Main Content - Expands to full width when sidebar is hidden */}
        <div className="flex-1 flex flex-col min-h-0 relative w-full">
          <ChatWindow 
            onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
            sidebarOpen={sidebarOpen}
          />
        </div>
      </div>
    </ChatProvider>
  )
}

// Main App with Auth Routing
function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <AuthProvider>
          <Routes>
            {/* Public Routes */}
            <Route path="/" element={<HomePage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />

            {/* Protected Route - Chat */}
            <Route
              path="/chat"
              element={
                <ProtectedRoute>
                  <ChatPage />
                </ProtectedRoute>
              }
            />

            {/* 404 - Redirect to home */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  )
}

export default App