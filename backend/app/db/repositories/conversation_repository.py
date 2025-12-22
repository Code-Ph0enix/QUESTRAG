# ============================================================================
# backend/app/db/repositories/conversation_repository.py
# ============================================================================


"""
Conversation Repository - MongoDB Operations (UPDATED)

NOW COMPATIBLE with existing chat.py!

Added methods:
- get_conversation_history() - For chat.py compatibility
- log_retrieval() - For RL training data
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from bson import ObjectId
from pymongo import DESCENDING, ASCENDING

from app.db.mongodb import get_database
from app.models.conversation import (
    Conversation,
    ConversationListResponse,
    ConversationListResult
)


# ============================================================================
# CONVERSATION REPOSITORY
# ============================================================================

class ConversationRepository:
    """
    Repository for conversation operations.
    
    All MongoDB queries for conversations go through here.
    """
    
    def __init__(self):
        """Initialize repository - database connection is fetched dynamically"""
        self.collection_name = "conversations"
        self.retrieval_logs_collection = "retrieval_logs"
        print("✅ ConversationRepository initialized")
    
    @property
    def db(self):
        """Get database connection dynamically"""
        return get_database()
    
    @property
    def collection(self):
        """Get conversations collection"""
        if self.db is None:
            raise RuntimeError("MongoDB database not available")
        return self.db[self.collection_name]
    
    @property
    def retrieval_logs(self):
        """Get retrieval logs collection"""
        if self.db is None:
            raise RuntimeError("MongoDB database not available")
        return self.db[self.retrieval_logs_collection]
    
    # ========================================================================
    # CREATE (UPDATED - Compatible with chat.py)
    # ========================================================================
    
    async def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
        first_message: Optional[str] = None
    ) -> str:
        """
        Create a new conversation.
        
        UPDATED: Returns conversation_id string (not full object)
        This matches chat.py's expectation!
        
        Args:
            user_id: User ID who owns the conversation
            title: Optional conversation title
            first_message: Optional first user message
        
        Returns:
            str: conversation_id (ObjectId as string)
        """
        now = datetime.utcnow()
        
        # Auto-generate title if not provided
        if not title:
            if first_message:
                # Simple title from first 50 chars
                title = first_message[:50] + ("..." if len(first_message) > 50 else "")
            else:
                title = f"Conversation {now.strftime('%Y-%m-%d %H:%M')}"
        
        # Create conversation document
        conversation_data = {
            "user_id": user_id,
            "title": title,
            "messages": [],
            "is_archived": False,
            "is_deleted": False,
            "created_at": now,
            "updated_at": now,
            "last_message_at": None,
            "message_count": 0
        }
        
        # Add first message if provided
        if first_message:
            message = {
                "role": "user",
                "content": first_message,
                "timestamp": now,
                "metadata": None
            }
            conversation_data["messages"].append(message)
            conversation_data["last_message_at"] = now
            conversation_data["message_count"] = 1
        
        # Insert into database
        result = await self.collection.insert_one(conversation_data)
        
        # Return conversation_id as string
        return str(result.inserted_id)
    
    # ========================================================================
    # READ
    # ========================================================================
    
    async def get_conversation(
        self,
        conversation_id: str
    ) -> Optional[Dict]:
        """
        Get conversation by ID (returns raw dict).
        
        UPDATED: No user_id verification here (done in service layer)
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            dict or None: Raw conversation document
        """
        try:
            result = await self.collection.find_one({
                "_id": ObjectId(conversation_id),
                "is_deleted": False
            })
            
            if result:
                # Add conversation_id field for compatibility
                result['conversation_id'] = str(result['_id'])
            
            return result
        
        except Exception as e:
            print(f"❌ Error getting conversation: {e}")
            return None
    
    async def get_conversation_by_id(
        self,
        conversation_id: str,
        user_id: str
    ) -> Optional[Conversation]:
        """
        Get conversation by ID (with user verification, returns Pydantic model).
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID (for ownership verification)
        
        Returns:
            Conversation or None
        """
        try:
            result = await self.collection.find_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id,
                "is_deleted": False
            })
            
            if result:
                return Conversation(**result)
            return None
        
        except Exception as e:
            print(f"❌ Error getting conversation: {e}")
            return None
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for chat.py compatibility.
        
        Returns format expected by chat_service.process_query():
        [
            {'role': 'user', 'content': '...', 'metadata': {...}},
            {'role': 'assistant', 'content': '...', 'metadata': {...}}
        ]
        
        Args:
            conversation_id: Conversation ID
            max_messages: Maximum messages to return (recent first)
        
        Returns:
            List of message dicts
        """
        try:
            conversation = await self.get_conversation(conversation_id)
            
            if not conversation:
                return []
            
            messages = conversation.get('messages', [])
            
            # Return last N messages
            recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
            
            # Convert to expected format
            history = []
            for msg in recent_messages:
                history.append({
                    'role': msg['role'],
                    'content': msg['content'],
                    'metadata': msg.get('metadata')
                })
            
            return history
        
        except Exception as e:
            print(f"❌ Error getting history: {e}")
            return []
    
    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 10,
        skip: int = 0
    ) -> List[Dict]:
        """
        Get conversations for a user (for chat.py compatibility).
        
        Args:
            user_id: User ID
            limit: Max conversations to return
            skip: Number to skip (pagination)
        
        Returns:
            List of conversation dicts
        """
        try:
            cursor = self.collection.find({
                "user_id": user_id,
                "is_deleted": False
            }).sort("updated_at", DESCENDING).skip(skip).limit(limit)
            
            conversations = []
            async for doc in cursor:
                doc['conversation_id'] = str(doc['_id'])
                conversations.append(doc)
            
            return conversations
        
        except Exception as e:
            print(f"❌ Error listing conversations: {e}")
            return []
    
    async def list_conversations(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        include_archived: bool = False,
        search_query: Optional[str] = None
    ) -> ConversationListResult:
        """
        List conversations with pagination and filtering (for new API).
        
        Args:
            user_id: User ID
            page: Page number (1-indexed)
            page_size: Items per page
            include_archived: Include archived conversations?
            search_query: Optional search query
        
        Returns:
            ConversationListResult: Paginated list
        """
        # Build query filter
        query_filter = {
            "user_id": user_id,
            "is_deleted": False
        }
        
        if not include_archived:
            query_filter["is_archived"] = False
        
        if search_query:
            query_filter["$or"] = [
                {"title": {"$regex": search_query, "$options": "i"}},
                {"messages.content": {"$regex": search_query, "$options": "i"}}
            ]
        
        # Get total count
        total = await self.collection.count_documents(query_filter)
        
        # Calculate pagination
        skip = (page - 1) * page_size
        has_more = (skip + page_size) < total
        
        # Get conversations
        cursor = self.collection.find(query_filter).sort(
            "last_message_at", DESCENDING
        ).skip(skip).limit(page_size)
        
        conversations = []
        async for doc in cursor:
            preview = ""
            if doc.get("messages"):
                last_msg = doc["messages"][-1]
                preview = last_msg.get("content", "")[:100]
                if len(last_msg.get("content", "")) > 100:
                    preview += "..."
            
            conversations.append(ConversationListResponse(
                id=str(doc["_id"]),
                user_id=doc["user_id"],
                title=doc["title"],
                preview=preview,
                is_archived=doc.get("is_archived", False),
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
                last_message_at=doc.get("last_message_at"),
                message_count=doc.get("message_count", 0)
            ))
        
        return ConversationListResult(
            conversations=conversations,
            total=total,
            page=page,
            page_size=page_size,
            has_more=has_more
        )
    
    # ========================================================================
    # UPDATE
    # ========================================================================
    
    async def add_message(
        self,
        conversation_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Add a message to conversation (chat.py compatible).
        
        Args:
            conversation_id: Conversation ID
            message: Message dict with role, content, timestamp, metadata
        
        Returns:
            bool: True if added successfully
        """
        try:
            now = datetime.utcnow()
            
            # Ensure timestamp is datetime
            if 'timestamp' not in message or not isinstance(message['timestamp'], datetime):
                message['timestamp'] = now
            
            result = await self.collection.update_one(
                {
                    "_id": ObjectId(conversation_id),
                    "is_deleted": False
                },
                {
                    "$push": {"messages": message},
                    "$set": {
                        "updated_at": now,
                        "last_message_at": message['timestamp']
                    },
                    "$inc": {"message_count": 1}
                }
            )
            
            return result.modified_count > 0
        
        except Exception as e:
            print(f"❌ Error adding message: {e}")
            return False
    
    async def update_conversation(
        self,
        conversation_id: str,
        user_id: str,
        update_data: Dict[str, Any]
    ) -> Optional[Conversation]:
        """Update conversation properties."""
        try:
            update_data["updated_at"] = datetime.utcnow()
            
            result = await self.collection.update_one(
                {
                    "_id": ObjectId(conversation_id),
                    "user_id": user_id,
                    "is_deleted": False
                },
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                return await self.get_conversation_by_id(conversation_id, user_id)
            return None
        
        except Exception as e:
            print(f"❌ Error updating conversation: {e}")
            return None
    
    async def rename_conversation(
        self,
        conversation_id: str,
        user_id: str,
        new_title: str
    ) -> Optional[Conversation]:
        """Rename a conversation."""
        return await self.update_conversation(
            conversation_id,
            user_id,
            {"title": new_title}
        )
    
    async def archive_conversation(
        self,
        conversation_id: str,
        user_id: str,
        archived: bool = True
    ) -> Optional[Conversation]:
        """Archive or unarchive a conversation."""
        return await self.update_conversation(
            conversation_id,
            user_id,
            {"is_archived": archived}
        )
    
    # ========================================================================
    # DELETE
    # ========================================================================
    
    async def delete_conversation(
        self,
        conversation_id: str,
        soft_delete: bool = True
    ) -> bool:
        """
        Delete a conversation (chat.py compatible - no user_id check).
        
        Args:
            conversation_id: Conversation ID
            soft_delete: If True, mark as deleted. If False, remove from DB.
        
        Returns:
            bool: True if deleted
        """
        try:
            if soft_delete:
                result = await self.collection.update_one(
                    {"_id": ObjectId(conversation_id)},
                    {
                        "$set": {
                            "is_deleted": True,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                return result.modified_count > 0
            else:
                result = await self.collection.delete_one({
                    "_id": ObjectId(conversation_id)
                })
                return result.deleted_count > 0
        
        except Exception as e:
            print(f"❌ Error deleting conversation: {e}")
            return False
    
    # ========================================================================
    # RETRIEVAL LOGGING (For RL Training)
    # ========================================================================
    
    async def log_retrieval(
        self,
        log_data: Dict[str, Any]
    ) -> bool:
        """
        Log retrieval data for RL training.
        
        Stores query, policy decision, retrieval results for model improvement.
        
        Args:
            log_data: Dict with retrieval metadata
        
        Returns:
            bool: True if logged successfully
        """
        try:
            # Ensure timestamp
            if 'timestamp' not in log_data:
                log_data['timestamp'] = datetime.utcnow()
            
            await self.retrieval_logs.insert_one(log_data)
            return True
        
        except Exception as e:
            print(f"❌ Error logging retrieval: {e}")
            return False
    
    # ========================================================================
    # SEARCH
    # ========================================================================
    
    async def search_conversations(
        self,
        user_id: str,
        query: str,
        page: int = 1,
        page_size: int = 20
    ) -> ConversationListResult:
        """Search conversations by title or content."""
        return await self.list_conversations(
            user_id=user_id,
            page=page,
            page_size=page_size,
            include_archived=True,
            search_query=query
        )
    
    # ========================================================================
    # UTILITY
    # ========================================================================
    
    async def get_conversation_count(self, user_id: str) -> Dict[str, int]:
        """Get conversation counts for a user."""
        total = await self.collection.count_documents({
            "user_id": user_id,
            "is_deleted": False
        })
        
        archived = await self.collection.count_documents({
            "user_id": user_id,
            "is_deleted": False,
            "is_archived": True
        })
        
        return {
            "total": total,
            "active": total - archived,
            "archived": archived
        }
    
    async def create_indexes(self):
        """Create database indexes for better performance."""
        try:
            await self.collection.create_index([
                ("user_id", ASCENDING),
                ("is_deleted", ASCENDING),
                ("last_message_at", DESCENDING)
            ])
            
            await self.collection.create_index([
                ("user_id", ASCENDING),
                ("title", "text"),
                ("messages.content", "text")
            ])
            
            print("✅ Conversation indexes created")
        
        except Exception as e:
            print(f"⚠️ Failed to create indexes: {e}")
    # ============================================================================
    # ADD TO: backend/app/db/repositories/conversation_repository.py
    # Add this method to ConversationRepository class
    # ============================================================================

    async def update_message_reaction(
        self,
        conversation_id: str,
        message_index: int,
        reaction: Optional[str]
    ) -> bool:
        """
        Update reaction for a specific message.
    
        Args:
            conversation_id: Conversation ID
            message_index: Index of message in messages array (0-based)
            reaction: 'like', 'dislike', or None (to remove)
    
        Returns:
            bool: True if updated
        """
        try:
            from bson import ObjectId
        
            # Build update query
            result = await self.collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$set": {
                        f"messages.{message_index}.reaction": reaction,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
            if result.modified_count > 0:
                print(f"✅ Reaction updated for message {message_index}: {reaction}")
                return True
        
            print(f"⚠️ No message updated (conversation or index not found)")
            return False
    
        except Exception as e:
            print(f"❌ Update reaction error: {e}")
            return False
    
    
    
    # ============================================================================
    # ADD TO: backend/app/db/repositories/conversation_repository.py
    # Add these methods to ConversationRepository class
    # ============================================================================

    async def remove_last_assistant_message(
        self,
        conversation_id: str
    ) -> bool:
        """
        Remove the last assistant message from conversation.

        Used for regenerate functionality.

        Args:
            conversation_id: Conversation ID

        Returns:
            bool: True if removed
        """
        try:
            from bson import ObjectId

            # Get conversation
            conversation = await self.collection.find_one(
                {"_id": ObjectId(conversation_id)}
            )

            if not conversation:
                return False

            messages = conversation.get('messages', [])

            # Find last assistant message index
            last_assistant_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('role') == 'assistant':
                    last_assistant_idx = i
                    break
                    
            if last_assistant_idx is None:
                print("⚠️ No assistant message to remove")
                return False

            # Remove message
            messages.pop(last_assistant_idx)

            # Update conversation
            result = await self.collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$set": {
                        "messages": messages,
                        "message_count": len(messages),
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            if result.modified_count > 0:
                print(f"✅ Removed last assistant message from conversation {conversation_id}")
                return True

            return False

        except Exception as e:
            print(f"❌ Remove last assistant message error: {e}")
            return False

    # ROLLBACK: Below function was previously defined OUTSIDE the class (no indentation)
    # causing "update_last_user_message is not a known attribute of ConversationRepository"
    # If you need to rollback, remove the indentation from this function
    async def update_last_user_message(
        self,
        conversation_id: str,
        new_content: str
    ) -> bool:
        """
        Update the content of last user message.
        
        Used for edit functionality.
        
        Args:
            conversation_id: Conversation ID
            new_content: New message content
        
        Returns:
            bool: True if updated
        """
        try:
            from bson import ObjectId
            
            # Get conversation
            conversation = await self.collection.find_one(
                {"_id": ObjectId(conversation_id)}
            )
            
            if not conversation:
                return False
            
            messages = conversation.get('messages', [])
            
            # Find last user message index
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('role') == 'user':
                    last_user_idx = i
                    break
            
            if last_user_idx is None:
                print("⚠️ No user message to update")
                return False
            
            # Update message content
            messages[last_user_idx]['content'] = new_content
            messages[last_user_idx]['timestamp'] = datetime.utcnow()
            messages[last_user_idx]['edited'] = True  # Flag as edited
            
            # Update conversation
            result = await self.collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$set": {
                        "messages": messages,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                print(f"✅ Updated last user message in conversation {conversation_id}")
                return True
            
            return False
        
        except Exception as e:
            print(f"❌ Update last user message error: {e}")
            return False


# ============================================================================
# GLOBAL REPOSITORY INSTANCE
# ============================================================================

conversation_repository = ConversationRepository()