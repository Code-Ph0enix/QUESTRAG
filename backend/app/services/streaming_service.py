# ============================================================================
# backend/app/services/streaming_service.py - NEW FILE
# ============================================================================

"""
Streaming Service - Server-Sent Events (SSE)

Handles real-time streaming of AI responses.
Integrates with chat_service.py RAG pipeline.
"""

import asyncio
import json
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime

from app.config import settings
from app.ml.policy_network import predict_policy_action
from app.ml.retriever import retrieve_documents, format_context
from app.core.llm_manager import llm_manager


# ============================================================================
# STREAMING SERVICE
# ============================================================================

class StreamingService:
    """
    Handles SSE streaming for real-time chat responses.
    
    Events sent:
    - status: Progress updates (retrieval, generation stages)
    - content: Response chunks (word by word)
    - metadata: Final stats (policy action, docs retrieved, etc.)
    - done: Stream completion signal
    - error: Error occurred
    """
    
    def __init__(self):
        print("🌊 StreamingService initialized")
    
    async def stream_chat_response(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response with progress updates.
        
        Yields SSE-formatted events:
        - event: status, content, metadata, done, error
        - data: JSON payload
        
        Args:
            query: User query
            conversation_history: Previous messages
            user_id: User ID
        
        Yields:
            str: SSE formatted events
        """
        import time
        start_time = time.time()
        
        if conversation_history is None:
            conversation_history = []
        
        try:
            # ================================================================
            # STAGE 1: Policy Decision
            # ================================================================
            yield self._format_sse_event(
                event="status",
                data={"stage": "policy", "message": "Analyzing query..."}
            )
            
            await asyncio.sleep(0.1)  # Small delay for UX
            
            policy_result = predict_policy_action(
                query=query,
                history=conversation_history,
                return_probs=True
            )
            
            # ================================================================
            # STAGE 2: Retrieval (if needed)
            # ================================================================
            retrieved_docs = []
            context = ""
            retrieval_time = 0
            
            if policy_result['should_retrieve']:
                yield self._format_sse_event(
                    event="status",
                    data={"stage": "retrieval", "message": "Searching knowledge base..."}
                )
                
                retrieval_start = time.time()
                
                try:
                    retrieved_docs = retrieve_documents(
                        query=query,
                        top_k=settings.TOP_K,
                        min_similarity=settings.SIMILARITY_THRESHOLD
                    )
                    
                    retrieval_time = (time.time() - retrieval_start) * 1000
                    
                    if retrieved_docs:
                        context = format_context(
                            retrieved_docs,
                            max_context_length=settings.MAX_CONTEXT_LENGTH
                        )
                        
                        yield self._format_sse_event(
                            event="status",
                            data={
                                "stage": "retrieval",
                                "message": f"Found {len(retrieved_docs)} relevant documents"
                            }
                        )
                
                except Exception as e:
                    print(f"⚠️ Retrieval error during streaming: {e}")
                    # Continue without retrieval
            
            # ================================================================
            # STAGE 3: Stream Generation
            # ================================================================
            yield self._format_sse_event(
                event="status",
                data={"stage": "generation", "message": "Generating response..."}
            )
            
            generation_start = time.time()
            full_response = ""
            
            # Stream from LLM
            async for chunk in llm_manager.stream_chat_response(
                query=query,
                context=context,
                history=conversation_history
            ):
                full_response += chunk
                
                yield self._format_sse_event(
                    event="content",
                    data={"text": chunk}
                )
            
            generation_time = (time.time() - generation_start) * 1000
            total_time = (time.time() - start_time) * 1000
            
            # ================================================================
            # STAGE 4: Send Metadata
            # ================================================================
            metadata = {
                "policy_action": policy_result['action'],
                "policy_confidence": policy_result['confidence'],
                "documents_retrieved": len(retrieved_docs),
                "top_doc_score": retrieved_docs[0]['score'] if retrieved_docs else None,
                "retrieval_time_ms": round(retrieval_time, 2),
                "generation_time_ms": round(generation_time, 2),
                "total_time_ms": round(total_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add retrieved docs metadata
            if retrieved_docs:
                metadata['retrieved_docs_metadata'] = [
                    {
                        'faq_id': doc['faq_id'],
                        'score': doc['score'],
                        'category': doc['category'],
                        'rank': doc['rank']
                    }
                    for doc in retrieved_docs
                ]
            
            yield self._format_sse_event(
                event="metadata",
                data=metadata
            )
            
            # ================================================================
            # STAGE 5: Done
            # ================================================================
            yield self._format_sse_event(
                event="done",
                data={"message": "Stream completed"}
            )
        
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            import traceback
            traceback.print_exc()
            
            yield self._format_sse_event(
                event="error",
                data={"error": str(e), "message": "An error occurred during streaming"}
            )
    
    def _format_sse_event(self, event: str, data: Dict[str, Any]) -> str:
        """
        Format data as SSE event.
        
        SSE format:
        event: <event_name>
        data: <json_data>
        
        (blank line to separate events)
        """
        json_data = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {json_data}\n\n"


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

streaming_service = StreamingService()