"""
Multi-LLM Manager with Groq (ChatGroq) and HuggingFace Fallback Logic

Architecture:
- Primary: Groq API with 3 keys (sequential fallback)
- Fallback: HuggingFace Inference API with 3 tokens (sequential fallback)
- Llama 3 8B for chat interface
- Llama 3 70B for evaluation

Fallback Logic:
1. Try GROQ_API_KEY_1
2. If fails, try GROQ_API_KEY_2
3. If fails, try GROQ_API_KEY_3
4. If all Groq keys fail, try HF_TOKEN_1
5. If fails, try HF_TOKEN_2
6. If fails, try HF_TOKEN_3
"""

import time
from typing import AsyncGenerator, List, Dict, Optional, Literal
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from huggingface_hub import InferenceClient
from app.config import settings
import asyncio

# ============================================================================
# GROQ MANAGER WITH FALLBACK
# ============================================================================
class GroqManager:
    """
    Groq API Manager with multiple API key fallback support
    Uses ChatGroq from langchain_groq
    """
    
    def __init__(self):
        """Initialize Groq manager with all available API keys"""
        self.api_keys = settings.get_groq_api_keys()
        self.chat_model_name = settings.GROQ_CHAT_MODEL  # llama-3.1-8b-instant
        self.eval_model_name = settings.GROQ_EVAL_MODEL  # llama-3.3-70b-versatile
        
        # Track current key index
        self.current_key_index = 0
        
        # Rate limiting tracking
        self.requests_this_minute = 0
        self.last_reset = time.time()
        
        if not self.api_keys:
            raise ValueError("No Groq API keys configured. Set GROQ_API_KEY_1 in .env")
        
        print(f"✅ Groq Manager initialized with {len(self.api_keys)} API key(s)")
        print(f"   Chat Model: {self.chat_model_name}")
        print(f"   Eval Model: {self.eval_model_name}")
    
    def _check_rate_limits(self):
        """
        Check and reset rate limit counters.
        Groq Free: 30 requests/min
        """
        current_time = time.time()
        
        # Reset counters every minute
        if current_time - self.last_reset > 60:
            self.requests_this_minute = 0
            self.last_reset = current_time
        
        # Check if limits exceeded
        # =================================================================
        # Uncomment below if rate limiting enforcement is needed
        # =================================================================

        # if self.requests_this_minute >= settings.GROQ_REQUESTS_PER_MINUTE:
        #     wait_time = 60 - (current_time - self.last_reset)
        #     print(f"⚠️ Groq rate limit hit. Waiting {wait_time:.1f}s...")
        #     time.sleep(wait_time)
        #     self._check_rate_limits()
    
    def _create_llm(self, api_key: str, model_name: str) -> ChatGroq:
        """Create ChatGroq instance with given API key and model"""
        return ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            max_retries=0  # Disable automatic retries, we handle fallback manually
        )
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        task: Literal["chat", "evaluation"] = "chat"
    ) -> str:
        """
        Generate response using Groq with fallback logic.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            task: Task type to determine model (chat uses 8B, evaluation uses 70B)
        
        Returns:
            str: Generated response text
        
        Raises:
            Exception: If all Groq API keys fail
        """
        self._check_rate_limits()
        
        # Select model based on task
        model_name = self.eval_model_name if task == "evaluation" else self.chat_model_name
        
        # Format messages for LangChain
        formatted_messages = []
        
        # Add system message if provided
        if system_prompt:
            formatted_messages.append(SystemMessage(content=system_prompt))
        
        # Convert conversation messages
        for msg in messages:
            if msg['role'] == 'user':
                formatted_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                formatted_messages.append(AIMessage(content=msg['content']))
        
        # Try each Groq API key sequentially
        for key_index, api_key in enumerate(self.api_keys, 1):
            try:
                print(f"🔑 Trying Groq API Key {key_index}/{len(self.api_keys)} with {model_name}...")
                
                # Create LLM instance with current key
                llm = self._create_llm(api_key, model_name)
                
                # Generate response
                response = await llm.ainvoke(formatted_messages)
                
                # Track rate limits
                self.requests_this_minute += 1
                
                print(f"✅ Groq API Key {key_index} succeeded")
                return response.content
                
            except Exception as e:
                print(f"❌ Groq API Key {key_index} failed: {e}")
                
                # If this was the last key, raise exception
                if key_index == len(self.api_keys):
                    print(f"❌ All {len(self.api_keys)} Groq API keys exhausted")
                    raise Exception(f"All Groq API keys failed. Last error: {e}")
                
                # Otherwise, continue to next key
                print(f"⏭️ Falling back to next Groq API key...")
                continue

# ============================================================================
# HUGGINGFACE MANAGER WITH FALLBACK
# ============================================================================
class HuggingFaceManager:
    """
    HuggingFace Inference API Manager with multiple token fallback support
    Uses InferenceClient from huggingface_hub
    """
    
    def __init__(self):
        """Initialize HuggingFace manager with all available tokens"""
        self.tokens = settings.get_hf_tokens()
        self.chat_model_name = settings.HF_CHAT_MODEL
        self.eval_model_name = settings.HF_EVAL_MODEL
        
        if not self.tokens:
            raise ValueError("No HuggingFace tokens configured. Set HF_TOKEN_1 in .env")
        
        print(f"✅ HuggingFace Manager initialized with {len(self.tokens)} token(s)")
        print(f"   Chat Model: {self.chat_model_name}")
        print(f"   Eval Model: {self.eval_model_name}")
    
    def _create_client(self, token: str, model_name: str) -> InferenceClient:
        """Create InferenceClient instance with given token and model"""
        return InferenceClient(
            model=model_name,
            token=token
        )
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        task: Literal["chat", "evaluation"] = "chat"
    ) -> str:
        """
        Generate response using HuggingFace Inference API with fallback logic.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            task: Task type to determine model
        
        Returns:
            str: Generated response text
        
        Raises:
            Exception: If all HuggingFace tokens fail
        """
        # Select model based on task
        model_name = self.eval_model_name if task == "evaluation" else self.chat_model_name
        
        # Format messages for HuggingFace chat API
        formatted_messages = []
        
        # Add system message if provided
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Convert conversation messages
        for msg in messages:
            formatted_messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Try each HuggingFace token sequentially
        for token_index, token in enumerate(self.tokens, 1):
            try:
                print(f"🔑 Trying HuggingFace Token {token_index}/{len(self.tokens)} with {model_name}...")
                
                # Create client with current token
                client = self._create_client(token, model_name)
                
                # Generate response using chat completion
                response = client.chat_completion(
                    messages=formatted_messages,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE
                )
                
                # Extract content from response
                content = response.choices[0].message.content
                
                print(f"✅ HuggingFace Token {token_index} succeeded")
                return content
                
            except Exception as e:
                print(f"❌ HuggingFace Token {token_index} failed: {e}")
                
                # If this was the last token, raise exception
                if token_index == len(self.tokens):
                    print(f"❌ All {len(self.tokens)} HuggingFace tokens exhausted")
                    raise Exception(f"All HuggingFace tokens failed. Last error: {e}")
                
                # Otherwise, continue to next token
                print(f"⏭️ Falling back to next HuggingFace token...")
                continue

# ============================================================================
# UNIFIED LLM MANAGER (Groq Primary, HuggingFace Fallback)
# ============================================================================
class LLMManager:
    """
    Unified LLM Manager with cascading fallback logic:
    1. Try all Groq API keys (primary)
    2. If all fail, try all HuggingFace tokens (fallback)
    
    Models:
    - Chat: Llama 3 8B (for user-facing chat responses)
    - Evaluation: Llama 3 70B (for response evaluation)
    """
    
    def __init__(self):
        """Initialize all LLM managers"""
        self.groq = None
        self.huggingface = None
        
        # Initialize Groq if configured
        if settings.is_groq_enabled():
            try:
                self.groq = GroqManager()
            except Exception as e:
                print(f"⚠️ Failed to initialize Groq: {e}")
        
        # Initialize HuggingFace if configured
        if settings.is_hf_enabled():
            try:
                self.huggingface = HuggingFaceManager()
            except Exception as e:
                print(f"⚠️ Failed to initialize HuggingFace: {e}")
        
        # Check if at least one is available
        if not self.groq and not self.huggingface:
            raise ValueError("No LLM provider configured. Set either Groq or HuggingFace credentials in .env")
        
        print("✅ LLM Manager initialized with fallback logic")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        task: Literal["chat", "evaluation"] = "chat"
        ) -> str:
        """
        Generate response with cascading fallback logic.
        
        Fallback order:
        1. Try all Groq API keys (3 keys)
        2. If all Groq keys fail, try all HuggingFace tokens (3 tokens)
        
        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            task: Task type - "chat" (8B) or "evaluation" (70B)
        
        Returns:
            str: Generated response
        
        Raises:
            ValueError: If all providers fail
        """
        # Try Groq first (if available)
        if self.groq:
            try:
                print("🚀 Attempting Groq API (Primary)...")
                response = await self.groq.generate(messages, system_prompt, task)
                return response
            except Exception as groq_error:
                print(f"❌ All Groq API keys failed: {groq_error}")
                
                # Fall back to HuggingFace if available
                if self.huggingface:
                    print("🔄 Falling back to HuggingFace Inference API...")
                else:
                    raise ValueError(f"Groq failed and no HuggingFace fallback configured: {groq_error}")
        
        # Try HuggingFace (if Groq failed or not available)
        if self.huggingface:
            try:
                print("🚀 Attempting HuggingFace API (Fallback)...")
                response = await self.huggingface.generate(messages, system_prompt, task)
                return response
            except Exception as hf_error:
                raise ValueError(f"All LLM providers exhausted. HuggingFace error: {hf_error}")
        
        raise ValueError("No LLM provider available")
    
    # ============================================================================
    # ADD TO: backend/app/core/llm_manager.py
    # Add this method to LLMManager class
    # ============================================================================

    async def stream_chat_response(
        self,
        query: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response (yields chunks as they're generated).
    
        Tries Groq first (streaming), falls back to HuggingFace (non-streaming).
    
        Args:
            query: User query
            context: Retrieved context
            history: Conversation history
            max_tokens: Max response length
            temperature: Sampling temperature
    
        Yields:
            str: Response chunks
        """
        if history is None:
            history = []
    
        # Build system prompt
        system_prompt = """You are an expert banking assistant specialized in Indian financial regulations and banking practices.

    Instructions:
    - Answer accurately using provided context when available
    - If context is insufficient, still respond helpfully
    - Keep responses clear and concise
    - Never fabricate specific policies or rates
    - Maintain a professional tone"""
    
        # Build user message
        user_message = query
        if context:
            user_message = f"""Context from knowledge base:
    {context}

    User Query: {query}

    Please answer the query using the context above when relevant."""
    
        # ====================================================================
        # TRY GROQ (STREAMING SUPPORTED)
        # ====================================================================
        if self.groq:
            try:
                # Build messages for Groq
                messages = [{"role": "system", "content": system_prompt}]
            
                # Add history
                for msg in history[-10:]:  # Last 10 messages
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
                # Add current query
                messages.append({"role": "user", "content": user_message})
            
                # Stream from Groq
                stream = self.groq.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True  # Enable streaming
                )
            
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
                return  # Success, exit
        
            except Exception as e:
                print(f"⚠️ Groq streaming failed: {e}")
                # Fall through to HuggingFace
    
        # ====================================================================
        # FALLBACK: HUGGINGFACE (NO STREAMING - SIMULATE)
        # ====================================================================
        if self.huggingface:
            try:
                print("⚠️ Using HuggingFace (simulated streaming)")
            
                # Build prompt for HuggingFace
                prompt = f"{system_prompt}\n\n"
            
                # Add history
                for msg in history[-5:]:
                    role = "Human" if msg['role'] == 'user' else "Assistant"
                    prompt += f"{role}: {msg['content']}\n"
            
                prompt += f"Human: {user_message}\nAssistant:"
            
                # Generate full response
                response = self.huggingface(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    return_full_text=False
                )[0]['generated_text']
            
                # Simulate streaming by splitting into words
                words = response.split()
                for i, word in enumerate(words):
                    # Add space except for first word
                    chunk = word if i == 0 else f" {word}"
                    yield chunk
                
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.05)  # 50ms per word
            
                return
        
            except Exception as e:
                print(f"❌ HuggingFace streaming failed: {e}")
    
        # ====================================================================
        # BOTH FAILED - RETURN ERROR
        # ====================================================================
        yield "I apologize, but I'm unable to generate a response at the moment. Please try again."
    
    async def generate_chat_response(
        self,
        query: str,
        context: str,
        history: List[Dict[str, str]]
    ) -> str:
        """
        Generate chat response (uses Llama 3 8B).
        
        Args:
            query: User query
            context: Retrieved context (from FAISS)
            history: Conversation history
        
        Returns:
            str: Chat response
        """
        # Import the detailed prompt
        from app.services.chat_service import BANKING_SYSTEM_PROMPT
        
        # Build enhanced system prompt with context
        system_prompt = BANKING_SYSTEM_PROMPT
        if context:
            system_prompt += f"\n\nRelevant Knowledge Base Context:\n{context}"
        else:
            system_prompt += "\n\nNo specific banking documents were retrieved for this query. Provide a helpful general response while acknowledging your banking specialization."
        
        # Build messages
        messages = history + [{'role': 'user', 'content': query}]
        
        # Generate using chat task (Llama 3 8B)
        return await self.generate(messages, system_prompt, task="chat")
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        context: str = ""
    ) -> Dict:
        """
        Evaluate response quality (uses Llama 3 70B for better evaluation).
        Used during RL training.
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context (if any)
        
        Returns:
            dict: Evaluation results
            {'quality': 'Good'/'Bad', 'explanation': '...'}
        """
        eval_prompt = f"""Evaluate this response:

Query: {query}
Response: {response}
Context used: {context if context else 'None'}

Is this response Good or Bad? Respond with just "Good" or "Bad" and brief explanation."""
        
        messages = [{'role': 'user', 'content': eval_prompt}]
        
        # Generate using evaluation task (Llama 3 70B)
        result = await self.generate(messages, task="evaluation")
        
        # Parse result
        quality = "Good" if "Good" in result else "Bad"
        
        return {
            'quality': quality,
            'explanation': result
        }

# ============================================================================
# GLOBAL LLM MANAGER INSTANCE
# ============================================================================
llm_manager = LLMManager()

# ============================================================================
# USAGE EXAMPLE (for reference)
# ============================================================================
"""
# In your service file:
from app.core.llm_manager import llm_manager

# Generate chat response (uses Llama 3 8B with Groq → HF fallback)
response = await llm_manager.generate_chat_response(
    query="What is my account balance?",
    context="Your balance is $1000",
    history=[]
)

# Evaluate response (uses Llama 3 70B with Groq → HF fallback)
evaluation = await llm_manager.evaluate_response(
    query="What is my balance?",
    response="Your balance is $1000",
    context="Balance: $1000"
)
"""