"""
Application Configuration
Settings for Banking RAG Chatbot with JWT Authentication
Updated to support multiple Groq API keys and HuggingFace tokens with fallback logic
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings loaded from environment variables"""
    
    # ========================================================================
    # ENVIRONMENT
    # ========================================================================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # ========================================================================
    # MONGODB
    # ========================================================================
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "aml_ia_db")
    
    # ========================================================================
    # JWT AUTHENTICATION
    # ========================================================================
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
    # ========================================================================
    # CORS (for frontend)
    # ========================================================================
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    
    # ========================================================================
    # GROQ API KEYS (Multiple for fallback)
    # ========================================================================
    GROQ_API_KEY_1: str = os.getenv("GROQ_API_KEY_1", "")  # Primary
    GROQ_API_KEY_2: str = os.getenv("GROQ_API_KEY_2", "")  # Fallback 1
    GROQ_API_KEY_3: str = os.getenv("GROQ_API_KEY_3", "")  # Fallback 2
    
    # Model names for Groq (using correct GroqCloud naming)
    GROQ_CHAT_MODEL: str = os.getenv("GROQ_CHAT_MODEL", "llama3-8b-8192")  # For chat interface
    GROQ_EVAL_MODEL: str = os.getenv("GROQ_EVAL_MODEL", "llama3-70b-8192")  # For evaluation
    
    # ========================================================================
    # Commented as of now, can be re-enabled if rate limiting is needed
    # ========================================================================
    
    # GROQ_REQUESTS_PER_MINUTE: int = int(os.getenv("GROQ_REQUESTS_PER_MINUTE", "30"))
    
    # ========================================================================
    # HUGGING FACE TOKENS (Multiple for fallback)
    # ========================================================================
    HF_TOKEN_1: str = os.getenv("HF_TOKEN_1", "")  # Primary
    HF_TOKEN_2: str = os.getenv("HF_TOKEN_2", "")  # Fallback 1
    HF_TOKEN_3: str = os.getenv("HF_TOKEN_3", "")  # Fallback 2
    
    # HuggingFace model for inference (fallback from Groq)
    HF_CHAT_MODEL: str = os.getenv("HF_CHAT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    HF_EVAL_MODEL: str = os.getenv("HF_EVAL_MODEL", "meta-llama/Meta-Llama-3-70B-Instruct")
    
    # ========================================================================
    # MODEL PATHS (for RL Policy Network and RAG models)
    # ========================================================================
    POLICY_MODEL_PATH: str = os.getenv("POLICY_MODEL_PATH", "app/models/best_policy_model.pth")
    RETRIEVER_MODEL_PATH: str = os.getenv("RETRIEVER_MODEL_PATH", "app/models/best_retriever_model.pth")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "app/models/faiss_index.pkl")
    KB_PATH: str = os.getenv("KB_PATH", "app/data/final_knowledge_base.jsonl")
    
    # ========================================================================
    # DEVICE SETTINGS (for PyTorch/TensorFlow models)
    # ========================================================================
    DEVICE: str = os.getenv("DEVICE", "cpu")
    
    # ========================================================================
    # LLM PARAMETERS
    # ========================================================================
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    
    # ========================================================================
    # RAG PARAMETERS
    # ========================================================================
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "2000"))
    
    # ========================================================================
    # POLICY NETWORK PARAMETERS
    # ========================================================================
    POLICY_MAX_LEN: int = int(os.getenv("POLICY_MAX_LEN", "256"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    def get_groq_api_keys(self) -> List[str]:
        """Get all configured Groq API keys in priority order"""
        keys = []
        if self.GROQ_API_KEY_1:
            keys.append(self.GROQ_API_KEY_1)
        if self.GROQ_API_KEY_2:
            keys.append(self.GROQ_API_KEY_2)
        if self.GROQ_API_KEY_3:
            keys.append(self.GROQ_API_KEY_3)
        return keys
    
    def get_hf_tokens(self) -> List[str]:
        """Get all configured HuggingFace tokens in priority order"""
        tokens = []
        if self.HF_TOKEN_1:
            tokens.append(self.HF_TOKEN_1)
        if self.HF_TOKEN_2:
            tokens.append(self.HF_TOKEN_2)
        if self.HF_TOKEN_3:
            tokens.append(self.HF_TOKEN_3)
        return tokens
    
    def is_groq_enabled(self) -> bool:
        """Check if at least one Groq API key is configured"""
        return bool(self.get_groq_api_keys())
    
    def is_hf_enabled(self) -> bool:
        """Check if at least one HuggingFace token is configured"""
        return bool(self.get_hf_tokens())
    
    def get_allowed_origins(self) -> List[str]:
        """Parse allowed origins from comma-separated string"""
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    def get_llm_for_task(self, task: str = "chat") -> str:
        """
        Get LLM model name for a specific task.
        
        Args:
            task: Task type ('chat' or 'evaluation')
        
        Returns:
            str: Model name for the task
        """
        if task == "evaluation":
            return self.GROQ_EVAL_MODEL  # llama3-70b-8192
        else:
            return self.GROQ_CHAT_MODEL  # llama3-8b-8192

# ============================================================================
# CREATE GLOBAL SETTINGS INSTANCE
# ============================================================================
settings = Settings()

# ============================================================================
# PRINT CONFIGURATION ON LOAD
# ============================================================================
print("=" * 80)
print("✅ Configuration Loaded")
print("=" * 80)
print(f"Environment: {settings.ENVIRONMENT}")
print(f"Debug Mode: {settings.DEBUG}")
print(f"Database: {settings.DATABASE_NAME}")
print(f"Device: {settings.DEVICE}")
print(f"CORS Origins: {settings.ALLOWED_ORIGINS}")
print()
print("🔑 API Keys:")
groq_keys = settings.get_groq_api_keys()
print(f"   Groq Keys: {len(groq_keys)} configured")
for i, key in enumerate(groq_keys, 1):
    print(f"     - Key {i}: {'✅ Set' if key else '❌ Missing'}")
hf_tokens = settings.get_hf_tokens()
print(f"   HuggingFace Tokens: {len(hf_tokens)} configured")
for i, token in enumerate(hf_tokens, 1):
    print(f"     - Token {i}: {'✅ Set' if token else '❌ Missing'}")
print(f"   MongoDB: {'✅ Configured' if settings.MONGODB_URI else '❌ Missing'}")
print(f"   JWT Secret: {'✅ Configured' if settings.SECRET_KEY != 'your-secret-key-change-in-production' else '⚠️ Using default (CHANGE THIS!)'}")
print()
print("🤖 LLM Models:")
print(f"   Chat Model: {settings.GROQ_CHAT_MODEL} (Llama 3 8B)")
print(f"   Eval Model: {settings.GROQ_EVAL_MODEL} (Llama 3 70B)")
print()
print("🤖 Model Paths:")
print(f"   Policy Model: {settings.POLICY_MODEL_PATH}")
print(f"   Retriever Model: {settings.RETRIEVER_MODEL_PATH}")
print(f"   FAISS Index: {settings.FAISS_INDEX_PATH}")
print(f"   Knowledge Base: {settings.KB_PATH}")
print("=" * 80)
