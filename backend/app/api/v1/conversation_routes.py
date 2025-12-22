# ============================================================================
# backend/app/api/v1/conversation_routes.py
# ============================================================================



"""
Conversation & Chat API Endpoints (UNIFIED)

Combines:
- Chat functionality (send message, get response)
- Conversation management (list, search, rename, archive, delete)

All endpoints require JWT authentication.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from fastapi.responses import StreamingResponse
from app.services.streaming_service import streaming_service
import json

from app.services.chat_service import chat_service
from app.services.conversation_service import conversation_service
from app.db.repositories.conversation_repository import conversation_repository
from app.utils.dependencies import get_current_user
from app.models.user import TokenData
from app.models.conversation import (
    CreateConversationRequest,
    UpdateConversationRequest,
    ConversationResponse,
    ConversationListResult,
    ReactToMessageRequest  # 🆕 NEW
)


# ============================================================================
# CREATE ROUTER
# ============================================================================
router = APIRouter()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_validated_user_id(current_user: TokenData) -> str:
    """
    Extract and validate user_id from TokenData.
    
    ROLLBACK: This helper was added to fix type errors where 
    current_user.user_id (Optional[str]) was passed to functions 
    expecting str. If you need to rollback, remove this function
    and use current_user.user_id directly (will cause type warnings).
    
    Args:
        current_user: TokenData from JWT authentication
        
    Returns:
        str: Validated user_id
        
    Raises:
        HTTPException: If user_id is None
    """
    if not current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token"
        )
    return current_user.user_id


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request for chat endpoint"""
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID. If not provided, creates new conversation.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is my account balance?",
                "conversation_id": "507f1f77bcf86cd799439011"
            }
        }


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    response: str
    conversation_id: str
    policy_action: str
    policy_confidence: float
    documents_retrieved: int
    top_doc_score: Optional[float]
    total_time_ms: float
    timestamp: str


# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """
    💬 Send a message and get AI response.
    
    **Main chat endpoint** - processes user query through RAG pipeline.
    
    - If conversation_id provided: Adds to existing conversation
    - If no conversation_id: Creates new conversation with auto-generated title
    
    Requires JWT authentication.
    """
    try:
        # ROLLBACK: Original was: user_id = current_user.user_id
        user_id = get_validated_user_id(current_user)
        
        # ====================================================================
        # STEP 1: Get or Create Conversation
        # ====================================================================
        conversation_id = request.conversation_id
        
        if conversation_id:
            # Verify conversation exists and user owns it
            conversation = await conversation_repository.get_conversation(conversation_id)
            
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            
            if conversation["user_id"] != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied - you don't own this conversation"
                )
        else:
            # Create new conversation (auto-title will be generated after first response)
            from app.models.conversation import CreateConversationRequest
            create_req = CreateConversationRequest(
                title=None,  # Will be auto-generated
                first_message=request.query
            )
            
            new_conversation = await conversation_service.create_conversation(
                user_id=user_id,
                request=create_req,
                llm_manager=None  # Can pass llm_manager for smart titles
            )
            
            conversation_id = str(new_conversation.id)
        
        # ====================================================================
        # STEP 2: Get Conversation History
        # ====================================================================
        history = await conversation_repository.get_conversation_history(
            conversation_id=conversation_id,
            max_messages=10
        )
        
        # ====================================================================
        # STEP 3: Save User Message
        # ====================================================================
        await conversation_repository.add_message(
            conversation_id=conversation_id,
            message={
                'role': 'user',
                'content': request.query,
                'timestamp': datetime.utcnow(),
                'metadata': None
            }
        )
        
        # ====================================================================
        # STEP 4: Process Query (RAG Pipeline)
        # ====================================================================
        result = await chat_service.process_query(
            query=request.query,
            conversation_history=history,
            user_id=user_id
        )
        
        # ====================================================================
        # STEP 5: Save Assistant Response
        # ====================================================================
        await conversation_repository.add_message(
            conversation_id=conversation_id,
            message={
                'role': 'assistant',
                'content': result['response'],
                'timestamp': datetime.utcnow(),
                'metadata': {
                    'policy_action': result['policy_action'],
                    'policy_confidence': result['policy_confidence'],
                    'documents_retrieved': result['documents_retrieved'],
                    'top_doc_score': result['top_doc_score'],
                    'retrieval_time_ms': result['retrieval_time_ms'],
                    'generation_time_ms': result['generation_time_ms']
                }
            }
        )
        
        # ====================================================================
        # STEP 6: Log Retrieval Data (for RL training)
        # ====================================================================
        await conversation_repository.log_retrieval({
            'conversation_id': conversation_id,
            'user_id': user_id,
            'query': request.query,
            'policy_action': result['policy_action'],
            'policy_confidence': result['policy_confidence'],
            'should_retrieve': result['should_retrieve'],
            'documents_retrieved': result['documents_retrieved'],
            'top_doc_score': result['top_doc_score'],
            'response': result['response'],
            'retrieval_time_ms': result['retrieval_time_ms'],
            'generation_time_ms': result['generation_time_ms'],
            'total_time_ms': result['total_time_ms'],
            'retrieved_docs_metadata': result.get('retrieved_docs_metadata', []),
            'timestamp': datetime.utcnow()
        })
        
        # ====================================================================
        # STEP 7: Return Response
        # ====================================================================
        return ChatResponse(
            response=result['response'],
            conversation_id=conversation_id,
            policy_action=result['policy_action'],
            policy_confidence=result['policy_confidence'],
            documents_retrieved=result['documents_retrieved'],
            top_doc_score=result['top_doc_score'],
            total_time_ms=result['total_time_ms'],
            timestamp=result['timestamp']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}"
        )


# ============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/conversation", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: CreateConversationRequest = CreateConversationRequest(),
    current_user: TokenData = Depends(get_current_user)
):
    """
    🆕 Create a new conversation.
    
    Optional parameters:
    - title: Custom title (if not provided, auto-generated)
    - first_message: Optional first message to start conversation
    
    Returns full conversation object.
    """
    try:
        # ROLLBACK: Original was: user_id=current_user.user_id
        conversation = await conversation_service.create_conversation(
            user_id=get_validated_user_id(current_user),
            request=request,
            llm_manager=None
        )
        
        return ConversationResponse(
            id=str(conversation.id),
            user_id=conversation.user_id,
            title=conversation.title,
            messages=conversation.messages,
            is_archived=conversation.is_archived,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            last_message_at=conversation.last_message_at,
            message_count=conversation.message_count
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.get("/conversations", response_model=ConversationListResult)
async def list_conversations(
    page: int = 1,
    page_size: int = 20,
    include_archived: bool = False,
    current_user: TokenData = Depends(get_current_user)
):
    """
    📋 List all conversations for authenticated user.
    
    Supports:
    - Pagination (page, page_size)
    - Filter archived conversations
    - Sorted by last message (newest first)
    
    Returns lightweight list (without full message history).
    """
    try:
        # ROLLBACK: Original was: user_id=current_user.user_id
        result = await conversation_service.list_conversations(
            user_id=get_validated_user_id(current_user),
            page=page,
            page_size=page_size,
            include_archived=include_archived
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}"
        )


@router.get("/conversation/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    🔍 Get full conversation by ID.
    
    Returns complete conversation with all messages.
    User must own the conversation.
    """
    try:
        # ROLLBACK: Original was: user_id=current_user.user_id
        conversation = await conversation_service.get_conversation(
            conversation_id=conversation_id,
            user_id=get_validated_user_id(current_user)
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied"
            )
        
        return ConversationResponse(
            id=str(conversation.id),
            user_id=conversation.user_id,
            title=conversation.title,
            messages=conversation.messages,
            is_archived=conversation.is_archived,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            last_message_at=conversation.last_message_at,
            message_count=conversation.message_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}"
        )


@router.patch("/conversation/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """
    ✏️ Update conversation properties.
    
    Can update:
    - title: Rename conversation
    - is_archived: Archive/unarchive
    
    User must own the conversation.
    """
    try:
        # ROLLBACK: Original was: user_id=current_user.user_id
        conversation = await conversation_service.update_conversation(
            conversation_id=conversation_id,
            user_id=get_validated_user_id(current_user),
            request=request
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied"
            )
        
        return ConversationResponse(
            id=str(conversation.id),
            user_id=conversation.user_id,
            title=conversation.title,
            messages=conversation.messages,
            is_archived=conversation.is_archived,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            last_message_at=conversation.last_message_at,
            message_count=conversation.message_count
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update conversation: {str(e)}"
        )


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    permanent: bool = False,
    current_user: TokenData = Depends(get_current_user)
):
    """
    🗑️ Delete a conversation.
    
    - Default (permanent=False): Soft delete (can be recovered)
    - permanent=True: Hard delete (cannot be recovered)
    
    User must own the conversation.
    """
    try:
        # ROLLBACK: Original was: user_id=current_user.user_id
        success = await conversation_service.delete_conversation(
            conversation_id=conversation_id,
            user_id=get_validated_user_id(current_user),
            permanent=permanent
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied"
            )
        
        return {
            "message": "Conversation deleted successfully",
            "conversation_id": conversation_id,
            "permanent": permanent
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


@router.get("/conversations/search", response_model=ConversationListResult)
async def search_conversations(
    query: str,
    page: int = 1,
    page_size: int = 20,
    current_user: TokenData = Depends(get_current_user)
):
    """
    🔎 Search conversations by title or message content.
    
    Searches in:
    - Conversation titles
    - Message content
    
    Returns paginated results.
    """
    try:
        if not query or len(query.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query must be at least 2 characters"
            )
        
        # ROLLBACK: Original was: user_id=current_user.user_id
        result = await conversation_service.search_conversations(
            user_id=get_validated_user_id(current_user),
            query=query,
            page=page,
            page_size=page_size
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search conversations: {str(e)}"
        )


@router.get("/conversations/stats")
async def get_conversation_stats(
    current_user: TokenData = Depends(get_current_user)
):
    """
    📊 Get conversation statistics for user.
    
    Returns:
    - total: Total conversations
    - active: Non-archived conversations
    - archived: Archived conversations
    """
    try:
        # ROLLBACK: Original was: user_id=current_user.user_id
        user_id = get_validated_user_id(current_user)
        stats = await conversation_service.get_conversation_stats(
            user_id=user_id
        )
        
        return {
            "user_id": user_id,
            "stats": stats
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


# ========================================================================
# 🆕 NEW ENDPOINTS - Add at bottom before health check
# ========================================================================

@router.post("/conversation/{conversation_id}/message/{message_index}/react")
async def react_to_message(
    conversation_id: str,
    message_index: int,
    request: ReactToMessageRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """
    👍👎 React to a message.
    
    - message_index: Index of message in conversation (0-based)
    - reaction: 'like' or 'dislike'
    
    Replaces existing reaction if user reacts again.
    User must own the conversation.
    """
    try:
        # Get conversation
        # ROLLBACK: Original was: user_id=current_user.user_id
        conversation = await conversation_service.get_conversation(
            conversation_id=conversation_id,
            user_id=get_validated_user_id(current_user)
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied"
            )
        
        # Validate message index
        if message_index < 0 or message_index >= len(conversation.messages):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid message index. Conversation has {len(conversation.messages)} messages."
            )
        
        # Can only react to assistant messages
        message = conversation.messages[message_index]
        if message.role != 'assistant':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only react to assistant messages"
            )
        
        # Update reaction in MongoDB
        await conversation_repository.update_message_reaction(
            conversation_id=conversation_id,
            message_index=message_index,
            reaction=request.reaction
        )
        
        return {
            "message": "Reaction updated successfully",
            "conversation_id": conversation_id,
            "message_index": message_index,
            "reaction": request.reaction
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ React to message error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update reaction: {str(e)}"
        )


@router.delete("/conversation/{conversation_id}/message/{message_index}/react")
async def remove_reaction(
    conversation_id: str,
    message_index: int,
    current_user: TokenData = Depends(get_current_user)
):
    """
    ❌ Remove reaction from a message.
    
    User must own the conversation.
    """
    try:
        # Get conversation
        # ROLLBACK: Original was: user_id=current_user.user_id
        conversation = await conversation_service.get_conversation(
            conversation_id=conversation_id,
            user_id=get_validated_user_id(current_user)
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied"
            )
        
        # Validate message index
        if message_index < 0 or message_index >= len(conversation.messages):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid message index"
            )
        
        # Remove reaction in MongoDB
        await conversation_repository.update_message_reaction(
            conversation_id=conversation_id,
            message_index=message_index,
            reaction=None  # Remove reaction
        )
        
        return {
            "message": "Reaction removed successfully",
            "conversation_id": conversation_id,
            "message_index": message_index
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Remove reaction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove reaction: {str(e)}"
        )


# ========================================================================
# 🆕 STREAMING ENDPOINTS - Add after existing chat endpoint
# ========================================================================

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """
    💬 Send message and get AI response via Server-Sent Events (SSE).
    
    **Streaming endpoint** - returns response in real-time chunks.
    
    Events sent:
    - `status`: Progress updates (retrieval, generation)
    - `content`: Response text chunks
    - `metadata`: Final statistics (policy action, docs retrieved, timing)
    - `done`: Stream completion
    - `error`: If error occurs
    
    Frontend should use EventSource API to consume stream.
    
    Example:
    ```javascript
    const eventSource = new EventSource('/api/v1/chat/stream');
    
    eventSource.addEventListener('content', (e) => {
        const data = JSON.parse(e.data);
        console.log(data.text); // Append to UI
    });
    
    eventSource.addEventListener('done', () => {
        eventSource.close();
    });
    ```
    
    Requires JWT authentication (pass as query param: ?token=YOUR_JWT).
    """
    try:
        # ROLLBACK: Original was: user_id = current_user.user_id
        user_id = get_validated_user_id(current_user)
        
        # ====================================================================
        # STEP 1: Get or Create Conversation (same as non-streaming)
        # ====================================================================
        conversation_id = request.conversation_id
        
        if conversation_id:
            conversation = await conversation_repository.get_conversation(conversation_id)
            
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            
            if conversation["user_id"] != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        else:
            # Create new conversation
            from app.models.conversation import CreateConversationRequest
            create_req = CreateConversationRequest(
                title=None,
                first_message=request.query
            )
            
            new_conversation = await conversation_service.create_conversation(
                user_id=user_id,
                request=create_req,
                llm_manager=None
            )
            
            conversation_id = str(new_conversation.id)
        
        # ====================================================================
        # STEP 2: Get History
        # ====================================================================
        history = await conversation_repository.get_conversation_history(
            conversation_id=conversation_id,
            max_messages=10
        )
        
        # ====================================================================
        # STEP 3: Save User Message
        # ====================================================================
        await conversation_repository.add_message(
            conversation_id=conversation_id,
            message={
                'role': 'user',
                'content': request.query,
                'timestamp': datetime.utcnow(),
                'metadata': None
            }
        )
        
        # ====================================================================
        # STEP 4: Stream Response
        # ====================================================================
        async def generate_stream():
            """Generator that adds conversation_id to first event"""
            
            # Send conversation_id first (so frontend knows where to save)
            yield f"event: conversation_id\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"
            
            # Collect full response for saving
            full_response = ""
            final_metadata = {}
            
            # Stream from service
            async for sse_event in streaming_service.stream_chat_response(
                query=request.query,
                conversation_history=history,
                user_id=user_id
            ):
                yield sse_event
                
                # Parse event to collect data
                if "event: content" in sse_event:
                    # Extract text from: data: {"text": "..."}
                    import re
                    match = re.search(r'"text":\s*"([^"]*)"', sse_event)
                    if match:
                        full_response += match.group(1)
                
                elif "event: metadata" in sse_event:
                    # Extract metadata
                    import re
                    data_match = re.search(r'data: (.+)', sse_event)
                    if data_match:
                        final_metadata = json.loads(data_match.group(1))
            
            # Save assistant response after streaming completes
            await conversation_repository.add_message(
                conversation_id=conversation_id,
                message={
                    'role': 'assistant',
                    'content': full_response,
                    'timestamp': datetime.utcnow(),
                    'metadata': {
                        'policy_action': final_metadata.get('policy_action'),
                        'policy_confidence': final_metadata.get('policy_confidence'),
                        'documents_retrieved': final_metadata.get('documents_retrieved'),
                        'top_doc_score': final_metadata.get('top_doc_score'),
                        'retrieval_time_ms': final_metadata.get('retrieval_time_ms'),
                        'generation_time_ms': final_metadata.get('generation_time_ms')
                    }
                }
            )
            
            # Log retrieval data
            await conversation_repository.log_retrieval({
                'conversation_id': conversation_id,
                'user_id': user_id,
                'query': request.query,
                'policy_action': final_metadata.get('policy_action'),
                'policy_confidence': final_metadata.get('policy_confidence'),
                'should_retrieve': final_metadata.get('documents_retrieved', 0) > 0,
                'documents_retrieved': final_metadata.get('documents_retrieved', 0),
                'top_doc_score': final_metadata.get('top_doc_score'),
                'response': full_response,
                'retrieval_time_ms': final_metadata.get('retrieval_time_ms'),
                'generation_time_ms': final_metadata.get('generation_time_ms'),
                'total_time_ms': final_metadata.get('total_time_ms'),
                'retrieved_docs_metadata': final_metadata.get('retrieved_docs_metadata', []),
                'timestamp': datetime.utcnow()
            })
        
        # Return SSE stream
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Streaming endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream response: {str(e)}"
        )


# ========================================================================
# 🆕 REGENERATE RESPONSE (with streaming)
# ========================================================================

@router.post("/conversation/{conversation_id}/regenerate")
async def regenerate_last_response(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    🔄 Regenerate the last assistant response.
    
    - Removes last assistant message
    - Re-processes last user query
    - Returns streaming response
    
    User must own the conversation.
    """
    try:
        # Get conversation
        # ROLLBACK: Original was: user_id=current_user.user_id
        conversation = await conversation_service.get_conversation(
            conversation_id=conversation_id,
            user_id=get_validated_user_id(current_user)
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        if len(conversation.messages) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Need at least 2 messages to regenerate"
            )
        
        # Get last user message
        last_user_msg = None
        for msg in reversed(conversation.messages):
            if msg.role == 'user':
                last_user_msg = msg
                break
        
        if not last_user_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found to regenerate from"
            )
        
        # Remove last assistant message(s)
        await conversation_repository.remove_last_assistant_message(conversation_id)
        
        # Get updated history
        history = await conversation_repository.get_conversation_history(
            conversation_id=conversation_id,
            max_messages=10
        )
        
        # Stream regenerated response
        async def generate_stream():
            yield f"event: conversation_id\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"
            
            full_response = ""
            final_metadata = {}
            
            # ROLLBACK: Original was: user_id=current_user.user_id
            async for sse_event in streaming_service.stream_chat_response(
                query=last_user_msg.content,
                conversation_history=history,
                user_id=get_validated_user_id(current_user)
            ):
                yield sse_event
                
                if "event: content" in sse_event:
                    import re
                    match = re.search(r'"text":\s*"([^"]*)"', sse_event)
                    if match:
                        full_response += match.group(1)
                
                elif "event: metadata" in sse_event:
                    import re
                    data_match = re.search(r'data: (.+)', sse_event)
                    if data_match:
                        final_metadata = json.loads(data_match.group(1))
            
            # Save new response
            await conversation_repository.add_message(
                conversation_id=conversation_id,
                message={
                    'role': 'assistant',
                    'content': full_response,
                    'timestamp': datetime.utcnow(),
                    'metadata': {
                        'policy_action': final_metadata.get('policy_action'),
                        'policy_confidence': final_metadata.get('policy_confidence'),
                        'documents_retrieved': final_metadata.get('documents_retrieved'),
                        'regenerated': True  # Flag for analytics
                    }
                }
            )
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Regenerate error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to regenerate: {str(e)}"
        )


# ========================================================================
# 🆕 EDIT LAST MESSAGE (then regenerate)
# ========================================================================

class EditMessageRequest(BaseModel):
    """Request body for editing last user message"""
    new_content: str = Field(..., min_length=1, max_length=2000)


@router.post("/conversation/{conversation_id}/edit")
async def edit_and_regenerate(
    conversation_id: str,
    request: EditMessageRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """
    ✏️ Edit last user message and regenerate response.
    
    - Updates last user message content
    - Removes last assistant response
    - Regenerates with new message
    - Returns streaming response
    
    User must own the conversation.
    """
    try:
        # Get conversation
        # ROLLBACK: Original was: user_id=current_user.user_id
        conversation = await conversation_service.get_conversation(
            conversation_id=conversation_id,
            user_id=get_validated_user_id(current_user)
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Update last user message
        success = await conversation_repository.update_last_user_message(
            conversation_id=conversation_id,
            new_content=request.new_content.strip()
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update message"
            )
        
        # Remove last assistant message
        await conversation_repository.remove_last_assistant_message(conversation_id)
        
        # Get updated history
        history = await conversation_repository.get_conversation_history(
            conversation_id=conversation_id,
            max_messages=10
        )
        
        # Stream regenerated response with edited query
        async def generate_stream():
            yield f"event: conversation_id\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"
            
            full_response = ""
            final_metadata = {}
            
            # ROLLBACK: Original was: user_id=current_user.user_id
            async for sse_event in streaming_service.stream_chat_response(
                query=request.new_content,
                conversation_history=history,
                user_id=get_validated_user_id(current_user)
            ):
                yield sse_event
                
                if "event: content" in sse_event:
                    import re
                    match = re.search(r'"text":\s*"([^"]*)"', sse_event)
                    if match:
                        full_response += match.group(1)
                
                elif "event: metadata" in sse_event:
                    import re
                    data_match = re.search(r'data: (.+)', sse_event)
                    if data_match:
                        final_metadata = json.loads(data_match.group(1))
            
            # Save new response
            await conversation_repository.add_message(
                conversation_id=conversation_id,
                message={
                    'role': 'assistant',
                    'content': full_response,
                    'timestamp': datetime.utcnow(),
                    'metadata': {
                        'policy_action': final_metadata.get('policy_action'),
                        'policy_confidence': final_metadata.get('policy_confidence'),
                        'documents_retrieved': final_metadata.get('documents_retrieved'),
                        'edited': True  # Flag for analytics
                    }
                }
            )
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Edit error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to edit message: {str(e)}"
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def chat_health():
    """
    🏥 Health check for chat & conversation service.
    
    Public endpoint (no auth required).
    """
    try:
        health = await chat_service.health_check()
        
        return {
            "status": "healthy",
            "service": "chat & conversations",
            "components": health.get('components', {}),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "chat & conversations",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }