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
    Message,
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
        """Initialize repository with database connection"""
        self.db = get_database()
        self.collection_name = "conversations"
        self.retrieval_logs_collection = "retrieval_logs"
        print("✅ ConversationRepository initialized with MongoDB")
    
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
# GLOBAL REPOSITORY INSTANCE
# ============================================================================

conversation_repository = ConversationRepository()


# """
# Conversation Repository - MongoDB CRUD operations
# Handles storing and retrieving conversations from MongoDB Atlas

# Repository Pattern: Separates database logic from business logic
# This makes code cleaner and easier to test

# Collections:
# - conversations: Stores complete conversations with messages
# - retrieval_logs: Logs each retrieval operation (for RL training data)
# """

# import uuid
# from datetime import datetime
# from typing import List, Dict, Optional
# from bson import ObjectId

# from app.db.mongodb import get_database


# # ============================================================================
# # CONVERSATION REPOSITORY
# # ============================================================================

# class ConversationRepository:
#     """
#     Repository for conversation data in MongoDB.
    
#     Provides CRUD operations for:
#     1. Conversations (user chat sessions)
#     2. Retrieval logs (for RL training and analytics)
#     """
    
#     def __init__(self):
#         """
#         Initialize repository with database connection.
        
#         Gracefully handles case where MongoDB is not connected.
#         """
#         self.db = get_database()
        
#         # Graceful handling if MongoDB not connected
#         if self.db is None:
#             print("⚠️ ConversationRepository: MongoDB not connected")
#             print("   Repository will not function until database is connected")
#             self.conversations = None
#             self.retrieval_logs = None
#         else:
#             self.conversations = self.db["conversations"]
#             self.retrieval_logs = self.db["retrieval_logs"]
#             print("✅ ConversationRepository initialized with MongoDB")
    
#     def _check_connection(self):
#         """
#         Check if MongoDB is connected.
        
#         Raises:
#             RuntimeError: If MongoDB is not connected
#         """
#         if self.db is None or self.conversations is None:
#             raise RuntimeError(
#                 "MongoDB not connected. Cannot perform database operations. "
#                 "Check MONGODB_URI in .env file."
#             )
    
#     # ========================================================================
#     # CONVERSATION CRUD OPERATIONS
#     # ========================================================================
    
#     async def create_conversation(
#         self,
#         user_id: str,
#         conversation_id: Optional[str] = None
#     ) -> str:
#         """
#         Create a new conversation.
        
#         Args:
#             user_id: User ID who owns this conversation
#             conversation_id: Optional custom conversation ID (auto-generated if None)
        
#         Returns:
#             str: Conversation ID
        
#         Raises:
#             RuntimeError: If MongoDB not connected
#         """
#         self._check_connection()
        
#         if conversation_id is None:
#             conversation_id = str(uuid.uuid4())
        
#         conversation = {
#             "conversation_id": conversation_id,
#             "user_id": user_id,
#             "messages": [],  # Will store all messages
#             "created_at": datetime.now(),
#             "updated_at": datetime.now(),
#             "status": "active"  # active, archived, deleted
#         }
        
#         await self.conversations.insert_one(conversation)
        
#         return conversation_id
    
#     async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
#         """
#         Get a conversation by ID.
        
#         Args:
#             conversation_id: Conversation ID
        
#         Returns:
#             dict or None: Conversation document
        
#         Raises:
#             RuntimeError: If MongoDB not connected
#         """
#         self._check_connection()
        
#         conversation = await self.conversations.find_one(
#             {"conversation_id": conversation_id}
#         )
        
#         # Convert MongoDB ObjectId to string for JSON serialization
#         if conversation and "_id" in conversation:
#             conversation["_id"] = str(conversation["_id"])
        
#         return conversation
    
#     # async def get_user_conversations(
#     #     self,
#     #     user_id: str,
#     #     limit: int = 10,
#     #     skip: int = 0
#     # ) -> List[Dict]:
#     #     """
#     #     Get all conversations for a user.
        
#     #     Args:
#     #         user_id: User ID
#     #         limit: Maximum number of conversations to return
#     #         skip: Number of conversations to skip (for pagination)
        
#     #     Returns:
#     #         list: List of conversation documents
        
#     #     Raises:
#     #         RuntimeError: If MongoDB not connected
#     #     """
#     #     self._check_connection()
        
#     #     cursor = self.conversations.find(
#     #         {"user_id": user_id, "status": "active"}
#     #     ).sort("updated_at", -1).skip(skip).limit(limit)
        
#     #     conversations = await cursor.to_list(length=limit)
        
#     #     # Convert ObjectIds to strings
#     #     for conv in conversations:
#     #         if "_id" in conv:
#     #             conv["_id"] = str(conv["_id"])
        
#     #     return conversations
#     async def get_user_conversations(
#         self,
#         user_id: str,
#         limit: int = 10,
#         skip: int = 0
#     ) -> List[Dict]:
#         """Get all conversations for a user."""
#         # Gracefully return empty list if not connected
#         if self.db is None or self.conversations is None:
#             print("⚠️  MongoDB not connected - returning empty conversations list")
#             return []
    
#         cursor = self.conversations.find(
#             {"user_id": user_id, "status": "active"}
#         ).sort("updated_at", -1).skip(skip).limit(limit)
    
#         conversations = await cursor.to_list(length=limit)
    
#         # Convert ObjectIds to strings
#         for conv in conversations:
#             if "_id" in conv:
#                 conv["_id"] = str(conv["_id"])
    
#         return conversations

    
#     async def add_message(
#         self,
#         conversation_id: str,
#         message: Dict
#     ) -> bool:
#         """
#         Add a message to a conversation.
        
#         Args:
#             conversation_id: Conversation ID
#             message: Message dict
#                 {
#                     'role': 'user' or 'assistant',
#                     'content': str,
#                     'timestamp': datetime,
#                     'metadata': dict (optional - policy_action, docs_retrieved, etc.)
#                 }
        
#         Returns:
#             bool: Success status
        
#         Raises:
#             RuntimeError: If MongoDB not connected
#         """
#         self._check_connection()
        
#         # Ensure timestamp exists
#         if "timestamp" not in message:
#             message["timestamp"] = datetime.now()
        
#         # Add message to conversation
#         result = await self.conversations.update_one(
#             {"conversation_id": conversation_id},
#             {
#                 "$push": {"messages": message},
#                 "$set": {"updated_at": datetime.now()}
#             }
#         )
        
#         return result.modified_count > 0
    
#     async def get_conversation_history(
#         self,
#         conversation_id: str,
#         max_messages: int = None
#     ) -> List[Dict]:
#         """
#         Get conversation history (messages only).
        
#         Args:
#             conversation_id: Conversation ID
#             max_messages: Optional limit on number of messages
        
#         Returns:
#             list: List of messages
        
#         Raises:
#             RuntimeError: If MongoDB not connected
#         """
#         self._check_connection()
        
#         conversation = await self.get_conversation(conversation_id)
        
#         if not conversation:
#             return []
        
#         messages = conversation.get("messages", [])
        
#         if max_messages:
#             messages = messages[-max_messages:]
        
#         return messages
    
#     async def delete_conversation(self, conversation_id: str) -> bool:
#         """
#         Soft delete a conversation (mark as deleted, don't actually delete).
        
#         Args:
#             conversation_id: Conversation ID
        
#         Returns:
#             bool: Success status
        
#         Raises:
#             RuntimeError: If MongoDB not connected
#         """
#         self._check_connection()
        
#         result = await self.conversations.update_one(
#             {"conversation_id": conversation_id},
#             {
#                 "$set": {
#                     "status": "deleted",
#                     "deleted_at": datetime.now()
#                 }
#             }
#         )
        
#         return result.modified_count > 0
    
#     # ========================================================================
#     # RETRIEVAL LOGS (for RL training)
#     # ========================================================================
    
#     async def log_retrieval(
#         self,
#         log_data: Dict
#     ) -> str:
#         """
#         Log a retrieval operation (for RL training and analysis).
        
#         Args:
#             log_data: Log data dict
#                 {
#                     'conversation_id': str,
#                     'user_id': str,
#                     'query': str,
#                     'policy_action': 'FETCH' or 'NO_FETCH',
#                     'policy_confidence': float,
#                     'documents_retrieved': int,
#                     'top_doc_score': float or None,
#                     'retrieved_docs_metadata': list,
#                     'response': str,
#                     'retrieval_time_ms': float,
#                     'generation_time_ms': float,
#                     'total_time_ms': float,
#                     'timestamp': datetime
#                 }
        
#         Returns:
#             str: Log ID
        
#         Raises:
#             RuntimeError: If MongoDB not connected
#         """
#         self._check_connection()
        
#         # Add timestamp if not present
#         if "timestamp" not in log_data:
#             log_data["timestamp"] = datetime.now()
        
#         # Generate log ID
#         log_id = str(uuid.uuid4())
#         log_data["log_id"] = log_id
        
#         # Insert log
#         await self.retrieval_logs.insert_one(log_data)
        
#         return log_id
    
#     async def get_retrieval_logs(
#         self,
#         conversation_id: Optional[str] = None,
#         user_id: Optional[str] = None,
#         limit: int = 100,
#         skip: int = 0
#     ) -> List[Dict]:
#         """
#         Get retrieval logs (for analysis and RL training).
        
#         Args:
#             conversation_id: Optional filter by conversation
#             user_id: Optional filter by user
#             limit: Maximum number of logs
#             skip: Number of logs to skip
        
#         Returns:
#             list: List of log documents
        
#         Raises:
#             RuntimeError: If MongoDB not connected
#         """
#         self._check_connection()
        
#         # Build query
#         query = {}
#         if conversation_id:
#             query["conversation_id"] = conversation_id
#         if user_id:
#             query["user_id"] = user_id
        
#         # Fetch logs
#         cursor = self.retrieval_logs.find(query).sort("timestamp", -1).skip(skip).limit(limit)
#         logs = await cursor.to_list(length=limit)
        
#         # Convert ObjectIds to strings
#         for log in logs:
#             if "_id" in log:
#                 log["_id"] = str(log["_id"])
        
#         return logs
    
#     async def get_logs_for_rl_training(
#         self,
#         min_date: Optional[datetime] = None,
#         limit: int = 1000
#     ) -> List[Dict]:
#         """
#         Get logs specifically for RL training.
#         Filters for logs with both policy decision and retrieval results.
        
#         Args:
#             min_date: Optional minimum date for logs
#             limit: Maximum number of logs
        
#         Returns:
#             list: List of log documents suitable for RL training
        
#         Raises:
#             RuntimeError: If MongoDB not connected
#         """
#         self._check_connection()
        
#         # Build query
#         query = {
#             "policy_action": {"$exists": True},
#             "response": {"$exists": True}
#         }
        
#         if min_date:
#             query["timestamp"] = {"$gte": min_date}
        
#         # Fetch logs
#         cursor = self.retrieval_logs.find(query).sort("timestamp", -1).limit(limit)
#         logs = await cursor.to_list(length=limit)
        
#         # Convert ObjectIds
#         for log in logs:
#             if "_id" in log:
#                 log["_id"] = str(log["_id"])
        
#         return logs
    
#     # ========================================================================
#     # ANALYTICS QUERIES
#     # ========================================================================
    
#     async def get_conversation_stats(self, user_id: str) -> Dict:
#         """
#         Get conversation statistics for a user.
        
#         Args:
#             user_id: User ID
        
#         Returns:
#             dict: Statistics
        
#         Raises:
#             RuntimeError: If MongoDB not connected
#         """
#         self._check_connection()
        
#         # Count total conversations
#         total_conversations = await self.conversations.count_documents({
#             "user_id": user_id,
#             "status": "active"
#         })
        
#         # Count total messages
#         pipeline = [
#             {"$match": {"user_id": user_id, "status": "active"}},
#             {"$project": {"message_count": {"$size": "$messages"}}}
#         ]
        
#         result = await self.conversations.aggregate(pipeline).to_list(length=None)
#         total_messages = sum(doc.get("message_count", 0) for doc in result)
        
#         return {
#             "total_conversations": total_conversations,
#             "total_messages": total_messages,
#             "avg_messages_per_conversation": total_messages / total_conversations if total_conversations > 0 else 0
#         }
    
#     async def get_policy_stats(self, user_id: Optional[str] = None) -> Dict:
#         """
#         Get policy decision statistics.
        
#         Args:
#             user_id: Optional user ID filter
        
#         Returns:
#             dict: Policy statistics
        
#         Raises:
#             RuntimeError: If MongoDB not connected
#         """
#         self._check_connection()
        
#         # Build query
#         query = {}
#         if user_id:
#             query["user_id"] = user_id
        
#         # Count FETCH vs NO_FETCH
#         fetch_count = await self.retrieval_logs.count_documents({
#             **query,
#             "policy_action": "FETCH"
#         })
        
#         no_fetch_count = await self.retrieval_logs.count_documents({
#             **query,
#             "policy_action": "NO_FETCH"
#         })
        
#         total = fetch_count + no_fetch_count
        
#         return {
#             "fetch_count": fetch_count,
#             "no_fetch_count": no_fetch_count,
#             "total": total,
#             "fetch_rate": fetch_count / total if total > 0 else 0,
#             "no_fetch_rate": no_fetch_count / total if total > 0 else 0
#         }


# # ============================================================================
# # USAGE EXAMPLE (for reference)
# # ============================================================================
# """
# # In your service or API endpoint:

# from app.db.repositories.conversation_repository import ConversationRepository

# repo = ConversationRepository()

# # Create conversation
# conv_id = await repo.create_conversation(user_id="user_123")

# # Add user message
# await repo.add_message(conv_id, {
#     'role': 'user',
#     'content': 'What is my balance?',
#     'timestamp': datetime.now()
# })

# # Add assistant message
# await repo.add_message(conv_id, {
#     'role': 'assistant',
#     'content': 'Your balance is $1000',
#     'timestamp': datetime.now(),
#     'metadata': {
#         'policy_action': 'FETCH',
#         'documents_retrieved': 3
#     }
# })

# # Get conversation history
# history = await repo.get_conversation_history(conv_id)

# # Log retrieval for RL training
# await repo.log_retrieval({
#     'conversation_id': conv_id,
#     'user_id': 'user_123',
#     'query': 'What is my balance?',
#     'policy_action': 'FETCH',
#     'documents_retrieved': 3,
#     'response': 'Your balance is $1000'
# })
# """







# """
# Conversation Repository - MongoDB CRUD operations
# Handles storing and retrieving conversations from MongoDB Atlas

# Repository Pattern: Separates database logic from business logic
# This makes code cleaner and easier to test
# """

# import uuid
# from datetime import datetime
# from typing import List, Dict, Optional
# from bson import ObjectId

# from app.db.mongodb import get_database


# # ============================================================================
# # CONVERSATION REPOSITORY
# # ============================================================================

# class ConversationRepository:
#     """
#     Repository for conversation data in MongoDB.
    
#     Collections used:
#     - conversations: Stores complete conversations with messages
#     - retrieval_logs: Logs each retrieval operation (for RL training)
#     """
    
#     def __init__(self):
#         """Initialize repository with database connection"""
#         self.db = get_database()
#         self.conversations = self.db["conversations"]
#         self.retrieval_logs = self.db["retrieval_logs"]
    
#     # ========================================================================
#     # CONVERSATION CRUD OPERATIONS
#     # ========================================================================
    
#     async def create_conversation(
#         self,
#         user_id: str,
#         conversation_id: Optional[str] = None
#     ) -> str:
#         """
#         Create a new conversation.
        
#         Args:
#             user_id: User ID who owns this conversation
#             conversation_id: Optional custom conversation ID (auto-generated if None)
        
#         Returns:
#             str: Conversation ID
#         """
#         if conversation_id is None:
#             conversation_id = str(uuid.uuid4())
        
#         conversation = {
#             "conversation_id": conversation_id,
#             "user_id": user_id,
#             "messages": [],  # Will store all messages
#             "created_at": datetime.now(),
#             "updated_at": datetime.now(),
#             "status": "active"  # active, archived, deleted
#         }
        
#         await self.conversations.insert_one(conversation)
        
#         return conversation_id
    
#     async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
#         """
#         Get a conversation by ID.
        
#         Args:
#             conversation_id: Conversation ID
        
#         Returns:
#             dict or None: Conversation document
#         """
#         conversation = await self.conversations.find_one(
#             {"conversation_id": conversation_id}
#         )
        
#         # Convert MongoDB ObjectId to string for JSON serialization
#         if conversation and "_id" in conversation:
#             conversation["_id"] = str(conversation["_id"])
        
#         return conversation
    
#     async def get_user_conversations(
#         self,
#         user_id: str,
#         limit: int = 10,
#         skip: int = 0
#     ) -> List[Dict]:
#         """
#         Get all conversations for a user.
        
#         Args:
#             user_id: User ID
#             limit: Maximum number of conversations to return
#             skip: Number of conversations to skip (for pagination)
        
#         Returns:
#             list: List of conversation documents
#         """
#         cursor = self.conversations.find(
#             {"user_id": user_id, "status": "active"}
#         ).sort("updated_at", -1).skip(skip).limit(limit)
        
#         conversations = await cursor.to_list(length=limit)
        
#         # Convert ObjectIds to strings
#         for conv in conversations:
#             if "_id" in conv:
#                 conv["_id"] = str(conv["_id"])
        
#         return conversations
    
#     async def add_message(
#         self,
#         conversation_id: str,
#         message: Dict
#     ) -> bool:
#         """
#         Add a message to a conversation.
        
#         Args:
#             conversation_id: Conversation ID
#             message: Message dict
#                 {
#                     'role': 'user' or 'assistant',
#                     'content': str,
#                     'timestamp': datetime,
#                     'metadata': dict (optional - policy_action, docs_retrieved, etc.)
#                 }
        
#         Returns:
#             bool: Success status
#         """
#         # Ensure timestamp exists
#         if "timestamp" not in message:
#             message["timestamp"] = datetime.now()
        
#         # Add message to conversation
#         result = await self.conversations.update_one(
#             {"conversation_id": conversation_id},
#             {
#                 "$push": {"messages": message},
#                 "$set": {"updated_at": datetime.now()}
#             }
#         )
        
#         return result.modified_count > 0
    
#     async def get_conversation_history(
#         self,
#         conversation_id: str,
#         max_messages: int = None
#     ) -> List[Dict]:
#         """
#         Get conversation history (messages only).
        
#         Args:
#             conversation_id: Conversation ID
#             max_messages: Optional limit on number of messages
        
#         Returns:
#             list: List of messages
#         """
#         conversation = await self.get_conversation(conversation_id)
        
#         if not conversation:
#             return []
        
#         messages = conversation.get("messages", [])
        
#         if max_messages:
#             messages = messages[-max_messages:]
        
#         return messages
    
#     async def delete_conversation(self, conversation_id: str) -> bool:
#         """
#         Soft delete a conversation (mark as deleted, don't actually delete).
        
#         Args:
#             conversation_id: Conversation ID
        
#         Returns:
#             bool: Success status
#         """
#         result = await self.conversations.update_one(
#             {"conversation_id": conversation_id},
#             {
#                 "$set": {
#                     "status": "deleted",
#                     "deleted_at": datetime.now()
#                 }
#             }
#         )
        
#         return result.modified_count > 0
    
#     # ========================================================================
#     # RETRIEVAL LOGS (for RL training)
#     # ========================================================================
    
#     async def log_retrieval(
#         self,
#         log_data: Dict
#     ) -> str:
#         """
#         Log a retrieval operation (for RL training and analysis).
        
#         Args:
#             log_data: Log data dict
#                 {
#                     'conversation_id': str,
#                     'user_id': str,
#                     'query': str,
#                     'policy_action': 'FETCH' or 'NO_FETCH',
#                     'policy_confidence': float,
#                     'documents_retrieved': int,
#                     'top_doc_score': float or None,
#                     'retrieved_docs_metadata': list,
#                     'response': str,
#                     'retrieval_time_ms': float,
#                     'generation_time_ms': float,
#                     'total_time_ms': float,
#                     'timestamp': datetime
#                 }
        
#         Returns:
#             str: Log ID
#         """
#         # Add timestamp if not present
#         if "timestamp" not in log_data:
#             log_data["timestamp"] = datetime.now()
        
#         # Generate log ID
#         log_id = str(uuid.uuid4())
#         log_data["log_id"] = log_id
        
#         # Insert log
#         await self.retrieval_logs.insert_one(log_data)
        
#         return log_id
    
#     async def get_retrieval_logs(
#         self,
#         conversation_id: Optional[str] = None,
#         user_id: Optional[str] = None,
#         limit: int = 100,
#         skip: int = 0
#     ) -> List[Dict]:
#         """
#         Get retrieval logs (for analysis and RL training).
        
#         Args:
#             conversation_id: Optional filter by conversation
#             user_id: Optional filter by user
#             limit: Maximum number of logs
#             skip: Number of logs to skip
        
#         Returns:
#             list: List of log documents
#         """
#         # Build query
#         query = {}
#         if conversation_id:
#             query["conversation_id"] = conversation_id
#         if user_id:
#             query["user_id"] = user_id
        
#         # Fetch logs
#         cursor = self.retrieval_logs.find(query).sort("timestamp", -1).skip(skip).limit(limit)
#         logs = await cursor.to_list(length=limit)
        
#         # Convert ObjectIds to strings
#         for log in logs:
#             if "_id" in log:
#                 log["_id"] = str(log["_id"])
        
#         return logs
    
#     async def get_logs_for_rl_training(
#         self,
#         min_date: Optional[datetime] = None,
#         limit: int = 1000
#     ) -> List[Dict]:
#         """
#         Get logs specifically for RL training.
#         Filters for logs with both policy decision and retrieval results.
        
#         Args:
#             min_date: Optional minimum date for logs
#             limit: Maximum number of logs
        
#         Returns:
#             list: List of log documents suitable for RL training
#         """
#         # Build query
#         query = {
#             "policy_action": {"$exists": True},
#             "response": {"$exists": True}
#         }
        
#         if min_date:
#             query["timestamp"] = {"$gte": min_date}
        
#         # Fetch logs
#         cursor = self.retrieval_logs.find(query).sort("timestamp", -1).limit(limit)
#         logs = await cursor.to_list(length=limit)
        
#         # Convert ObjectIds
#         for log in logs:
#             if "_id" in log:
#                 log["_id"] = str(log["_id"])
        
#         return logs
    
#     # ========================================================================
#     # ANALYTICS QUERIES
#     # ========================================================================
    
#     async def get_conversation_stats(self, user_id: str) -> Dict:
#         """
#         Get conversation statistics for a user.
        
#         Args:
#             user_id: User ID
        
#         Returns:
#             dict: Statistics
#         """
#         # Count total conversations
#         total_conversations = await self.conversations.count_documents({
#             "user_id": user_id,
#             "status": "active"
#         })
        
#         # Count total messages
#         pipeline = [
#             {"$match": {"user_id": user_id, "status": "active"}},
#             {"$project": {"message_count": {"$size": "$messages"}}}
#         ]
        
#         result = await self.conversations.aggregate(pipeline).to_list(length=None)
#         total_messages = sum(doc.get("message_count", 0) for doc in result)
        
#         return {
#             "total_conversations": total_conversations,
#             "total_messages": total_messages,
#             "avg_messages_per_conversation": total_messages / total_conversations if total_conversations > 0 else 0
#         }
    
#     async def get_policy_stats(self, user_id: Optional[str] = None) -> Dict:
#         """
#         Get policy decision statistics.
        
#         Args:
#             user_id: Optional user ID filter
        
#         Returns:
#             dict: Policy statistics
#         """
#         # Build query
#         query = {}
#         if user_id:
#             query["user_id"] = user_id
        
#         # Count FETCH vs NO_FETCH
#         fetch_count = await self.retrieval_logs.count_documents({
#             **query,
#             "policy_action": "FETCH"
#         })
        
#         no_fetch_count = await self.retrieval_logs.count_documents({
#             **query,
#             "policy_action": "NO_FETCH"
#         })
        
#         total = fetch_count + no_fetch_count
        
#         return {
#             "fetch_count": fetch_count,
#             "no_fetch_count": no_fetch_count,
#             "total": total,
#             "fetch_rate": fetch_count / total if total > 0 else 0,
#             "no_fetch_rate": no_fetch_count / total if total > 0 else 0
#         }


# # ============================================================================
# # USAGE EXAMPLE (for reference)
# # ============================================================================
# """
# # In your service or API endpoint:

# from app.db.repositories.conversation_repository import ConversationRepository

# repo = ConversationRepository()

# # Create conversation
# conv_id = await repo.create_conversation(user_id="user_123")

# # Add user message
# await repo.add_message(conv_id, {
#     'role': 'user',
#     'content': 'What is my balance?',
#     'timestamp': datetime.now()
# })

# # Add assistant message
# await repo.add_message(conv_id, {
#     'role': 'assistant',
#     'content': 'Your balance is $1000',
#     'timestamp': datetime.now(),
#     'metadata': {
#         'policy_action': 'FETCH',
#         'documents_retrieved': 3
#     }
# })

# # Get conversation history
# history = await repo.get_conversation_history(conv_id)

# # Log retrieval for RL training
# await repo.log_retrieval({
#     'conversation_id': conv_id,
#     'user_id': 'user_123',
#     'query': 'What is my balance?',
#     'policy_action': 'FETCH',
#     'documents_retrieved': 3,
#     'response': 'Your balance is $1000'
# })
# """