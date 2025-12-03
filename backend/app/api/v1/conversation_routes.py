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
        user_id = current_user.user_id
        
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
        conversation = await conversation_service.create_conversation(
            user_id=current_user.user_id,
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
        result = await conversation_service.list_conversations(
            user_id=current_user.user_id,
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
        conversation = await conversation_service.get_conversation(
            conversation_id=conversation_id,
            user_id=current_user.user_id
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
        conversation = await conversation_service.update_conversation(
            conversation_id=conversation_id,
            user_id=current_user.user_id,
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
        success = await conversation_service.delete_conversation(
            conversation_id=conversation_id,
            user_id=current_user.user_id,
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
        
        result = await conversation_service.search_conversations(
            user_id=current_user.user_id,
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
        stats = await conversation_service.get_conversation_stats(
            user_id=current_user.user_id
        )
        
        return {
            "user_id": current_user.user_id,
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
        conversation = await conversation_service.get_conversation(
            conversation_id=conversation_id,
            user_id=current_user.user_id
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
        conversation = await conversation_service.get_conversation(
            conversation_id=conversation_id,
            user_id=current_user.user_id
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