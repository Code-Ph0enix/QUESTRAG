# """
# Conversation Models for MongoDB

# Handles conversation persistence with:
# - Auto-generated titles from first message
# - Message metadata (policy actions, retrieval stats)
# - Archive/unarchive support
# - Search indexing ready
# """

# from datetime import datetime
# from typing import List, Optional, Dict, Any
# from pydantic import BaseModel, Field
# from bson import ObjectId


# # ============================================================================
# # CUSTOM TYPES
# # ============================================================================

# class PyObjectId(ObjectId):
#     """Custom ObjectId type compatible with Pydantic v2"""

#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate

#     @classmethod
#     def validate(cls, v):
#         if not ObjectId.is_valid(v):
#             raise ValueError("Invalid ObjectId")
#         return ObjectId(v)

#     @classmethod
#     def __get_pydantic_json_schema__(cls, core_schema, handler):
#         schema = handler(core_schema)
#         schema.update(type="string")
#         return schema



# # ============================================================================
# # MESSAGE MODEL
# # ============================================================================

# class Message(BaseModel):
#     """
#     Single message in a conversation.
    
#     Contains:
#     - User/assistant content
#     - Metadata from RAG pipeline (policy action, retrieval stats)
#     - Timestamp
#     """
    
#     role: str = Field(..., description="Role: 'user' or 'assistant'")
#     content: str = Field(..., description="Message content")
#     timestamp: datetime = Field(default_factory=datetime.utcnow)
    
#     # Metadata from RAG pipeline (only for assistant messages)
#     metadata: Optional[Dict[str, Any]] = Field(
#         default=None,
#         description="RAG metadata: policy_action, confidence, docs_retrieved, etc."
#     )
    
#     class Config:
#         json_encoders = {
#             datetime: lambda v: v.isoformat()
#         }
#         schema_extra = {
#             "example": {
#                 "role": "user",
#                 "content": "What is my account balance?",
#                 "timestamp": "2024-01-15T10:30:00",
#                 "metadata": None
#             }
#         }


# # ============================================================================
# # CONVERSATION MODEL (MongoDB Document)
# # ============================================================================

# class Conversation(BaseModel):
#     """
#     Full conversation document stored in MongoDB.
    
#     Features:
#     - Auto-generated title from first user message
#     - Message history with metadata
#     - Archive/active status
#     - User association
#     - Search-ready structure
#     """
    
#     id: Optional[PyObjectId] = Field(alias="_id", default=None)
#     user_id: str = Field(..., description="User ID who owns this conversation")
#     title: str = Field(..., description="Conversation title (auto-generated or custom)")
    
#     messages: List[Message] = Field(
#         default_factory=list,
#         description="List of messages in chronological order"
#     )
    
#     # Status flags
#     is_archived: bool = Field(default=False, description="Is conversation archived?")
#     is_deleted: bool = Field(default=False, description="Soft delete flag")
    
#     # Timestamps
#     created_at: datetime = Field(default_factory=datetime.utcnow)
#     updated_at: datetime = Field(default_factory=datetime.utcnow)
#     last_message_at: Optional[datetime] = Field(default=None)
    
#     # Metadata
#     message_count: int = Field(default=0, description="Total messages (excluding deleted)")
    
#     class Config:
#         model_config = {
#             "populate_by_name": True,
#             "arbitrary_types_allowed": True,
#             "json_encoders": {
#                 ObjectId: str,
#                 datetime: lambda v: v.isoformat(),
#             },
#         }
#         schema_extra = {
#             "example": {
#                 "user_id": "user_123",
#                 "title": "Account Balance Inquiry",
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": "What is my account balance?",
#                         "timestamp": "2024-01-15T10:30:00"
#                     },
#                     {
#                         "role": "assistant",
#                         "content": "Your current account balance is...",
#                         "timestamp": "2024-01-15T10:30:05",
#                         "metadata": {
#                             "policy_action": "FETCH",
#                             "confidence": 0.95,
#                             "documents_retrieved": 3
#                         }
#                     }
#                 ],
#                 "is_archived": False,
#                 "created_at": "2024-01-15T10:30:00",
#                 "updated_at": "2024-01-15T10:30:05",
#                 "message_count": 2
#             }
#         }


# # ============================================================================
# # REQUEST/RESPONSE MODELS (for API)
# # ============================================================================

# class CreateConversationRequest(BaseModel):
#     """Request body for creating a new conversation"""
    
#     title: Optional[str] = Field(
#         default=None,
#         description="Optional custom title. If not provided, will be auto-generated from first message",
#         max_length=100
#     )
#     first_message: Optional[str] = Field(
#         default=None,
#         description="Optional first user message to start the conversation",
#         max_length=1000
#     )
    
#     class Config:
#         schema_extra = {
#             "example": {
#                 "title": "Savings Account Help",
#                 "first_message": "How do I open a savings account?"
#             }
#         }


# class AddMessageRequest(BaseModel):
#     """Request body for adding a message to conversation"""
    
#     message: str = Field(..., description="User message to add")
    
#     class Config:
#         schema_extra = {
#             "example": {
#                 "message": "What are the interest rates?"
#             }
#         }


# class UpdateConversationRequest(BaseModel):
#     """Request body for updating conversation properties"""
    
#     title: Optional[str] = Field(default=None, description="New title")
#     is_archived: Optional[bool] = Field(default=None, description="Archive status")
    
#     class Config:
#         schema_extra = {
#             "example": {
#                 "title": "Fixed Deposit Rates Discussion"
#             }
#         }


# class ConversationResponse(BaseModel):
#     """Response model for single conversation"""
    
#     id: str = Field(..., description="Conversation ID")
#     user_id: str
#     title: str
#     messages: List[Message]
#     is_archived: bool
#     created_at: datetime
#     updated_at: datetime
#     last_message_at: Optional[datetime]
#     message_count: int
    
#     class Config:
#         json_encoders = {
#             datetime: lambda v: v.isoformat()
#         }


# class ConversationListResponse(BaseModel):
#     """Response model for list of conversations (without full messages)"""
    
#     id: str
#     user_id: str
#     title: str
#     preview: str = Field(..., description="Last message preview (first 100 chars)")
#     is_archived: bool
#     created_at: datetime
#     updated_at: datetime
#     last_message_at: Optional[datetime]
#     message_count: int
    
#     class Config:
#         json_encoders = {
#             datetime: lambda v: v.isoformat()
#         }
#         schema_extra = {
#             "example": {
#                 "id": "507f1f77bcf86cd799439011",
#                 "user_id": "user_123",
#                 "title": "Account Balance Inquiry",
#                 "preview": "What is my current account balance?",
#                 "is_archived": False,
#                 "created_at": "2024-01-15T10:30:00",
#                 "updated_at": "2024-01-15T10:35:00",
#                 "last_message_at": "2024-01-15T10:35:00",
#                 "message_count": 6
#             }
#         }


# class ConversationListResult(BaseModel):
#     """Paginated list of conversations"""
    
#     conversations: List[ConversationListResponse]
#     total: int = Field(..., description="Total conversations matching filter")
#     page: int = Field(default=1, description="Current page number")
#     page_size: int = Field(default=20, description="Items per page")
#     has_more: bool = Field(..., description="Are there more pages?")
    
#     class Config:
#         schema_extra = {
#             "example": {
#                 "conversations": [],
#                 "total": 42,
#                 "page": 1,
#                 "page_size": 20,
#                 "has_more": True
#             }
#         }




# # class PyObjectId(ObjectId):
# #     """Custom ObjectId type for Pydantic validation"""
    
# #     @classmethod
# #     def __get_validators__(cls):
# #         yield cls.validate
    
# #     @classmethod
# #     def validate(cls, v):
# #         if not ObjectId.is_valid(v):
# #             raise ValueError("Invalid ObjectId")
# #         return ObjectId(v)
    
# #     @classmethod
# #     def __modify_schema__(cls, field_schema):
# #         field_schema.update(type="string")




#         # allow_population_by_field_name = True
#         # arbitrary_types_allowed = True
#         # model_config = {
#         #     "populate_by_name": True,
#         #     "arbitrary_types_allowed": True,
#         # }

#         # json_encoders = {
#         #     ObjectId: str,
#         #     datetime: lambda v: v.isoformat()
#         # }