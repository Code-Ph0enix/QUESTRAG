"""
Conversation Service - Business Logic Layer (UPDATED)

OPTIMIZED:
- Better error handling
- Smart title generation with fallbacks
- Async LLM title generation (optional)
- User verification in all operations
"""

import re
from typing import Optional, Dict, Any
from datetime import datetime

from app.db.repositories.conversation_repository import conversation_repository
from app.models.conversation import (
    Conversation,
    Message,
    ConversationListResult,
    CreateConversationRequest,
    UpdateConversationRequest
)


# ============================================================================
# CONVERSATION SERVICE
# ============================================================================

class ConversationService:
    """
    Business logic for conversation management.
    
    Handles validation, auto-titles, and business rules.
    """
    
    def __init__(self):
        """Initialize service"""
        self.repository = conversation_repository
        print("✅ ConversationService initialized")
    
    # ========================================================================
    # AUTO-TITLE GENERATION
    # ========================================================================
    
    def generate_title_from_message(
        self,
        message: str,
        max_length: int = 50
    ) -> str:
        """
        Generate conversation title from first user message.
        
        Optimized with better truncation logic.
        
        Args:
            message: First user message
            max_length: Maximum title length
        
        Returns:
            str: Generated title
        """
        message = message.strip()
        
        if not message:
            return "New Conversation"
        
        # Remove extra whitespace
        message = re.sub(r'\s+', ' ', message)
        
        # Try first sentence
        sentences = re.split(r'[.!?]+', message)
        first_sentence = sentences[0].strip()
        
        # Use more if first sentence too short
        if len(first_sentence) < 15 and len(sentences) > 1:
            first_sentence = f"{first_sentence}. {sentences[1].strip()}"
        
        # Truncate smartly
        if len(first_sentence) > max_length:
            title = first_sentence[:max_length].strip()
            # Break at word boundary
            last_space = title.rfind(' ')
            if last_space > max_length * 0.6:
                title = title[:last_space]
            title += "..."
        else:
            title = first_sentence
        
        # Capitalize
        if title:
            title = title[0].upper() + title[1:]
        
        # Remove trailing punctuation before ellipsis
        title = re.sub(r'[,;:]\.\.\.$', '...', title)
        
        return title if title else "New Conversation"
    
    async def generate_smart_title(
        self,
        first_message: str,
        llm_manager = None
    ) -> str:
        """
        Generate smart title using LLM (optional).
        
        Falls back gracefully if LLM unavailable.
        
        Args:
            first_message: First user message
            llm_manager: Optional LLM manager
        
        Returns:
            str: Generated title
        """
        # Try LLM generation if available
        if llm_manager:
            try:
                prompt = f"""Generate a concise title (max 50 chars) for this banking query:

"{first_message}"

Requirements:
- Clear and descriptive
- Banking/finance focused
- No quotes or formatting
- Maximum 50 characters

Title:"""
                
                # Use simple generation (not full chat)
                title = await llm_manager.generate_simple_response(
                    prompt=prompt,
                    max_tokens=15,
                    temperature=0.3
                )
                
                # Clean and validate
                title = title.strip().strip('"\'`')
                title = re.sub(r'\s+', ' ', title)
                
                if 5 < len(title) <= 60:
                    return title
            
            except Exception as e:
                print(f"⚠️ Smart title generation failed: {e}")
        
        # Fallback to simple generation
        return self.generate_title_from_message(first_message)
    
    # ========================================================================
    # CREATE
    # ========================================================================
    
    async def create_conversation(
        self,
        user_id: str,
        request: CreateConversationRequest = None,
        llm_manager = None
    ) -> Conversation:
        """
        Create a new conversation with optional first message.
        
        OPTIMIZED: Better title generation + error handling
        
        Args:
            user_id: User ID
            request: Optional create request
            llm_manager: Optional LLM manager
        
        Returns:
            Conversation: Created conversation (full object)
        """
        if request is None:
            request = CreateConversationRequest()
        
        # Determine title
        if request.title:
            title = request.title
        elif request.first_message:
            # Auto-generate from message
            title = await self.generate_smart_title(
                request.first_message,
                llm_manager
            )
        else:
            title = f"New Chat - {datetime.now().strftime('%b %d, %H:%M')}"
        
        # Create conversation (returns ID string)
        conversation_id = await self.repository.create_conversation(
            user_id=user_id,
            title=title,
            first_message=request.first_message
        )
        
        # Fetch full conversation
        conversation = await self.repository.get_conversation_by_id(
            conversation_id,
            user_id
        )
        
        if not conversation:
            raise ValueError("Failed to create conversation")
        
        return conversation
    
    # ========================================================================
    # READ
    # ========================================================================
    
    async def get_conversation(
        self,
        conversation_id: str,
        user_id: str
    ) -> Optional[Conversation]:
        """
        Get conversation with user verification.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID (must match owner)
        
        Returns:
            Conversation or None
        """
        return await self.repository.get_conversation_by_id(
            conversation_id,
            user_id
        )
    
    async def list_conversations(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        include_archived: bool = False
    ) -> ConversationListResult:
        """
        List conversations for user with pagination.
        
        Args:
            user_id: User ID
            page: Page number (1-indexed)
            page_size: Items per page
            include_archived: Include archived?
        
        Returns:
            ConversationListResult: Paginated list
        """
        # Validate pagination
        page = max(1, page)
        page_size = min(max(1, page_size), 100)  # Cap at 100
        
        return await self.repository.list_conversations(
            user_id=user_id,
            page=page,
            page_size=page_size,
            include_archived=include_archived
        )
    
    async def search_conversations(
        self,
        user_id: str,
        query: str,
        page: int = 1,
        page_size: int = 20
    ) -> ConversationListResult:
        """
        Search conversations by title/content.
        
        Args:
            user_id: User ID
            query: Search query
            page: Page number
            page_size: Items per page
        
        Returns:
            ConversationListResult: Search results
        """
        # Validate query
        if not query or len(query.strip()) < 2:
            # Return empty results for invalid query
            return ConversationListResult(
                conversations=[],
                total=0,
                page=page,
                page_size=page_size,
                has_more=False
            )
        
        return await self.repository.search_conversations(
            user_id=user_id,
            query=query.strip(),
            page=page,
            page_size=page_size
        )
    
    # ========================================================================
    # UPDATE
    # ========================================================================
    
    async def add_message_to_conversation(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Conversation]:
        """
        Add message to conversation with validation.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID (must match owner)
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata
        
        Returns:
            Updated Conversation or None
        """
        # Validate role
        if role not in ['user', 'assistant']:
            raise ValueError(f"Invalid role: {role}")
        
        # Validate content
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty")
        
        # Verify ownership
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            return None
        
        # Create message
        message = Message(
            role=role,
            content=content.strip(),
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        # Add to repository
        success = await self.repository.add_message(
            conversation_id,
            message.dict()
        )
        
        if success:
            return await self.get_conversation(conversation_id, user_id)
        return None
    
    async def update_conversation(
        self,
        conversation_id: str,
        user_id: str,
        request: UpdateConversationRequest
    ) -> Optional[Conversation]:
        """
        Update conversation properties.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID (must match owner)
            request: Update request
        
        Returns:
            Updated Conversation or None
        """
        update_data = {}
        
        if request.title is not None:
            # Validate title
            title = request.title.strip()
            if not title:
                raise ValueError("Title cannot be empty")
            if len(title) > 100:
                raise ValueError("Title too long (max 100 chars)")
            update_data["title"] = title
        
        if request.is_archived is not None:
            update_data["is_archived"] = request.is_archived
        
        if not update_data:
            # Nothing to update
            return await self.get_conversation(conversation_id, user_id)
        
        return await self.repository.update_conversation(
            conversation_id,
            user_id,
            update_data
        )
    
    async def rename_conversation(
        self,
        conversation_id: str,
        user_id: str,
        new_title: str
    ) -> Optional[Conversation]:
        """
        Rename conversation with validation.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            new_title: New title
        
        Returns:
            Updated Conversation or None
        """
        # Validate title
        new_title = new_title.strip()
        if not new_title:
            raise ValueError("Title cannot be empty")
        if len(new_title) > 100:
            raise ValueError("Title too long (max 100 chars)")
        
        return await self.repository.rename_conversation(
            conversation_id,
            user_id,
            new_title
        )
    
    async def archive_conversation(
        self,
        conversation_id: str,
        user_id: str
    ) -> Optional[Conversation]:
        """Archive a conversation."""
        return await self.repository.archive_conversation(
            conversation_id,
            user_id,
            archived=True
        )
    
    async def unarchive_conversation(
        self,
        conversation_id: str,
        user_id: str
    ) -> Optional[Conversation]:
        """Unarchive a conversation."""
        return await self.repository.archive_conversation(
            conversation_id,
            user_id,
            archived=False
        )
    
    # ========================================================================
    # DELETE
    # ========================================================================
    
    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: str,
        permanent: bool = False
    ) -> bool:
        """
        Delete conversation (with ownership verification).
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID (must match owner)
            permanent: Hard delete if True
        
        Returns:
            bool: True if deleted
        """
        # Verify ownership first
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            return False
        
        return await self.repository.delete_conversation(
            conversation_id,
            soft_delete=not permanent
        )
    
    # ========================================================================
    # UTILITY
    # ========================================================================
    
    async def get_conversation_stats(
        self,
        user_id: str
    ) -> Dict[str, int]:
        """
        Get conversation statistics.
        
        Args:
            user_id: User ID
        
        Returns:
            dict: Stats with total, active, archived
        """
        return await self.repository.get_conversation_count(user_id)


# ============================================================================
# GLOBAL SERVICE INSTANCE
# ============================================================================

conversation_service = ConversationService()