// ============================================================================
// Frontend Integration - How to Consume SSE Streams
// ============================================================================

// ============================================================================
// VANILLA JAVASCRIPT EXAMPLE
// ============================================================================

async function sendStreamingMessage(query, conversationId = null) {
    const token = localStorage.getItem('jwt_token');
    
    // Prepare request
    const requestData = {
        query: query,
        conversation_id: conversationId
    };
    
    try {
        // Make POST request to get stream
        const response = await fetch('http://localhost:8000/api/v1/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Read stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let fullResponse = '';
        let currentConversationId = conversationId;
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            // Decode chunk
            const chunk = decoder.decode(value);
            
            // Parse SSE events
            const events = parseSSE(chunk);
            
            for (const event of events) {
                switch (event.type) {
                    case 'conversation_id':
                        currentConversationId = event.data.conversation_id;
                        console.log('Conversation ID:', currentConversationId);
                        break;
                    
                    case 'status':
                        console.log('Status:', event.data.message);
                        // Update UI with status
                        updateStatusIndicator(event.data.message);
                        break;
                    
                    case 'content':
                        fullResponse += event.data.text;
                        // Append to UI in real-time
                        appendToMessageBubble(event.data.text);
                        break;
                    
                    case 'metadata':
                        console.log('Metadata:', event.data);
                        // Save metadata if needed
                        break;
                    
                    case 'done':
                        console.log('Stream completed');
                        hideStatusIndicator();
                        break;
                    
                    case 'error':
                        console.error('Stream error:', event.data.error);
                        showError(event.data.message);
                        break;
                }
            }
        }
        
        return {
            response: fullResponse,
            conversationId: currentConversationId
        };
    
    } catch (error) {
        console.error('Streaming error:', error);
        throw error;
    }
}

// Parse SSE format
function parseSSE(text) {
    const events = [];
    const lines = text.split('\n\n');
    
    for (const line of lines) {
        if (!line.trim()) continue;
        
        const eventMatch = line.match(/event: (.+)/);
        const dataMatch = line.match(/data: (.+)/);
        
        if (eventMatch && dataMatch) {
            try {
                events.push({
                    type: eventMatch[1],
                    data: JSON.parse(dataMatch[1])
                });
            } catch (e) {
                console.warn('Failed to parse SSE event:', e);
            }
        }
    }
    
    return events;
}


// ============================================================================
// REACT EXAMPLE (with hooks)
// ============================================================================

import { useState, useRef } from 'react';

function ChatComponent() {
    const [messages, setMessages] = useState([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [statusMessage, setStatusMessage] = useState('');
    const abortControllerRef = useRef(null);
    
    const sendStreamingMessage = async (query, conversationId) => {
        setIsStreaming(true);
        setStatusMessage('');
        
        // Add user message immediately
        const userMessage = { role: 'user', content: query };
        setMessages(prev => [...prev, userMessage]);
        
        // Create assistant message placeholder
        const assistantMessageId = Date.now();
        setMessages(prev => [...prev, { 
            id: assistantMessageId, 
            role: 'assistant', 
            content: '' 
        }]);
        
        try {
            const token = localStorage.getItem('jwt_token');
            
            // Create abort controller for cancellation
            abortControllerRef.current = new AbortController();
            
            const response = await fetch('http://localhost:8000/api/v1/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ query, conversation_id: conversationId }),
                signal: abortControllerRef.current.signal
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            let newConversationId = conversationId;
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const events = parseSSE(chunk);
                
                for (const event of events) {
                    switch (event.type) {
                        case 'conversation_id':
                            newConversationId = event.data.conversation_id;
                            break;
                        
                        case 'status':
                            setStatusMessage(event.data.message);
                            break;
                        
                        case 'content':
                            // Update assistant message
                            setMessages(prev => prev.map(msg => 
                                msg.id === assistantMessageId
                                    ? { ...msg, content: msg.content + event.data.text }
                                    : msg
                            ));
                            break;
                        
                        case 'done':
                            setStatusMessage('');
                            break;
                        
                        case 'error':
                            console.error('Error:', event.data);
                            setStatusMessage('');
                            break;
                    }
                }
            }
            
            return newConversationId;
        
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Stream cancelled by user');
            } else {
                console.error('Streaming error:', error);
            }
        } finally {
            setIsStreaming(false);
            setStatusMessage('');
            abortControllerRef.current = null;
        }
    };
    
    const stopStreaming = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
    };
    
    return (
        <div className="chat-container">
            <div className="messages">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        {msg.content}
                    </div>
                ))}
                
                {statusMessage && (
                    <div className="status-indicator">
                        {statusMessage}
                    </div>
                )}
            </div>
            
            {isStreaming && (
                <button onClick={stopStreaming}>
                    Stop Generating
                </button>
            )}
        </div>
    );
}


// ============================================================================
// REGENERATE & EDIT EXAMPLES
// ============================================================================

// Regenerate last response
async function regenerateResponse(conversationId) {
    const token = localStorage.getItem('jwt_token');
    
    const response = await fetch(
        `http://localhost:8000/api/v1/chat/conversation/${conversationId}/regenerate`,
        {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        }
    );
    
    // Process stream same as sendStreamingMessage
    // ...
}

// Edit last message
async function editLastMessage(conversationId, newContent) {
    const token = localStorage.getItem('jwt_token');
    
    const response = await fetch(
        `http://localhost:8000/api/v1/chat/conversation/${conversationId}/edit`,
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ new_content: newContent })
        }
    );
    
    // Process stream
    // ...
}