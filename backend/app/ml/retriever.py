"""
Custom Retriever with E5-Base-V2 and FAISS
Trained with InfoNCE + Triplet Loss for banking domain

This is adapted from your RAG.py with:
- CustomSentenceTransformer (e5-base-v2)
- Mean pooling + L2 normalization
- FAISS vector search
- Module-level caching (load once on startup)
- ✅ Compatible with v2 FAISS format
"""

import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel

from app.config import settings


# ============================================================================
# CUSTOM SENTENCE TRANSFORMER (From RAG.py)
# ============================================================================

class CustomSentenceTransformer(nn.Module):
    """
    Custom SentenceTransformer matching your training code.
    Uses e5-base-v2 with mean pooling and L2 normalization.
    
    Training Details:
    - Base model: intfloat/e5-base-v2
    - Loss: InfoNCE + Triplet Loss
    - Pooling: Mean pooling on last hidden state
    - Normalization: L2 normalization
    """
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        super().__init__()
        # Load pre-trained e5-base-v2 encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = self.encoder.config
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT encoder.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for padding
        
        Returns:
            torch.Tensor: L2-normalized embeddings (shape: [batch_size, 768])
        """
        # Get BERT outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling - same as training
        # Take hidden states from last layer
        token_embeddings = outputs.last_hidden_state
        
        # Expand attention mask to match token embeddings shape
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings (weighted by attention mask) and divide by sum of mask
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        
        # L2 normalize embeddings - same as training
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode(
        self, 
        sentences: List[str], 
        batch_size: int = 32, 
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode sentences using the same method as training.
        Adds 'query: ' prefix for e5-base-v2 compatibility.
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            convert_to_numpy: Whether to convert to numpy array
            show_progress_bar: Whether to show progress bar
        
        Returns:
            np.ndarray: Encoded embeddings (shape: [num_sentences, 768])
        """
        self.eval()  # Set model to evaluation mode
        
        # Handle single string input
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Add 'query: ' prefix for e5-base-v2 (required by model)
        # Handle None values and empty strings
        processed_sentences = []
        for sentence in sentences:
            if sentence is None:
                processed_sentences.append("query: ")  # Default empty query
            elif isinstance(sentence, str):
                processed_sentences.append(f"query: {sentence.strip()}")
            else:
                processed_sentences.append(f"query: {str(sentence)}")
        
        all_embeddings = []
        
        # Encode in batches
        with torch.no_grad():  # No gradient computation
            for i in range(0, len(processed_sentences), batch_size):
                batch_sentences = processed_sentences[i:i + batch_size]
                
                # Tokenize batch
                tokens = self.tokenizer(
                    batch_sentences,
                    truncation=True,
                    padding=True,
                    max_length=128,  # Same as training
                    return_tensors='pt'
                ).to(next(self.parameters()).device)
                
                # Get embeddings
                embeddings = self.forward(tokens['input_ids'], tokens['attention_mask'])
                
                # Convert to numpy if requested
                if convert_to_numpy:
                    embeddings = embeddings.cpu().numpy()
                
                all_embeddings.append(embeddings)
        
        # Combine all batches
        if convert_to_numpy:
            all_embeddings = np.vstack(all_embeddings)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)
        
        return all_embeddings


# ============================================================================
# CUSTOM RETRIEVER MODEL (Wrapper)
# ============================================================================

class CustomRetrieverModel:
    """
    Wrapper for your custom trained retriever model.
    Handles both knowledge base documents and query encoding.
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize retriever model.
        
        Args:
            model_path: Path to trained model weights (.pth file)
            device: Device to load model on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Create model instance
        self.model = CustomSentenceTransformer("intfloat/e5-base-v2").to(device)
        
        # Load your trained weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Custom retriever model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load custom model: {e}")
            print("🔄 Using base e5-base-v2 model (not trained)...")
        
        # Set to evaluation mode
        self.model.eval()
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode knowledge base documents.
        These are the responses/instructions we're retrieving.
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
        
        Returns:
            np.ndarray: Document embeddings (shape: [num_docs, 768])
        """
        return self.model.encode(documents, batch_size=batch_size, convert_to_numpy=True)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode user query for retrieval.
        
        Args:
            query: User query text
        
        Returns:
            np.ndarray: Query embedding (shape: [1, 768])
        """
        return self.model.encode([query], convert_to_numpy=True)


# ============================================================================
# MODULE-LEVEL CACHING (Load once on import)
# ============================================================================

# Global variables for caching
RETRIEVER_MODEL: Optional[CustomRetrieverModel] = None
FAISS_INDEX: Optional[faiss.Index] = None
KB_DATA: Optional[List[Dict]] = None


# ============================================================================
# ✅ UPDATED: COMPATIBLE FAISS LOADING (SUPPORTS V2 FORMAT)
# ============================================================================

def load_retriever() -> CustomRetrieverModel:
    """
    Load custom retriever model (called once on startup).
    Downloads from HuggingFace Hub if not present locally.
    Uses module-level caching - model stays in RAM.
    
    Returns:
        CustomRetrieverModel: Loaded retriever model
    """
    global RETRIEVER_MODEL
    
    if RETRIEVER_MODEL is None:
        # Download model from HF Hub if needed (for deployment)
        settings.download_model_if_needed(
            hf_filename="models/best_retriever_model.pth",
            local_path=settings.RETRIEVER_MODEL_PATH
        )
        
        print(f"Loading custom retriever from {settings.RETRIEVER_MODEL_PATH}...")
        
        RETRIEVER_MODEL = CustomRetrieverModel(
            model_path=settings.RETRIEVER_MODEL_PATH,
            device=settings.DEVICE
        )
        
        print("✅ Retriever model loaded and cached")
    
    return RETRIEVER_MODEL


def load_faiss_index():
    """
    ✅ UPDATED: Load FAISS index with v2 format compatibility.
    
    Supports multiple formats:
    - v2 format (dict with version info) - RECOMMENDED
    - Old tuple format (index_bytes, kb_data)
    - Legacy format (direct FAISS object) - will show warning
    
    Downloads from HuggingFace Hub if not present locally.
    Uses module-level caching - loaded once on startup.
    
    Returns:
        tuple: (faiss.Index, List[Dict]) - FAISS index and KB data
    """
    global FAISS_INDEX, KB_DATA
    
    if FAISS_INDEX is None or KB_DATA is None:
        # Download FAISS index from HF Hub if needed (for deployment)
        settings.download_model_if_needed(
            hf_filename="models/faiss_index.pkl",
            local_path=settings.FAISS_INDEX_PATH
        )
        
        # Download knowledge base from HF Hub if needed (for deployment)
        settings.download_model_if_needed(
            hf_filename="data/final_knowledge_base.jsonl",
            local_path=settings.KB_PATH
        )
        
        print(f"Loading FAISS index from {settings.FAISS_INDEX_PATH}...")
        
        try:
            # Load pickled data
            with open(settings.FAISS_INDEX_PATH, 'rb') as f:
                data = pickle.load(f)
            
            print(f"📦 Pickle loaded successfully")
            
            # ========================================================================
            # FORMAT 1: v2 Dictionary Format (RECOMMENDED)
            # ========================================================================
            if isinstance(data, dict) and data.get('format_version') == 'v2':
                print("📦 Detected v2 format (compatible)")
                try:
                    FAISS_INDEX = faiss.deserialize_index(data['index_bytes'])
                    KB_DATA = data['kb_data']
                    print(f"✅ FAISS index loaded successfully")
                    print(f"   Vectors: {FAISS_INDEX.ntotal}")
                    print(f"   KB docs: {len(KB_DATA)}")
                    print(f"   Dimension: {data.get('dimension', 'unknown')}")
                    print(f"   Built with FAISS: {data.get('faiss_version', 'unknown')}")
                    print(f"   Current FAISS: {faiss.__version__}")
                except Exception as e:
                    print(f"❌ Failed to deserialize v2 format: {e}")
                    raise RuntimeError(
                        f"Failed to load v2 FAISS index: {e}\n"
                        f"Please rebuild using: python build_faiss_index.py"
                    )
            
            # ========================================================================
            # FORMAT 2: Old Tuple Format (index_bytes, kb_data)
            # ========================================================================
            elif isinstance(data, tuple) and len(data) == 2:
                first_item, KB_DATA = data
                
                # Check if first item is bytes (serialized index)
                if isinstance(first_item, bytes):
                    print("📦 Detected old tuple format with bytes (attempting conversion)")
                    try:
                        FAISS_INDEX = faiss.deserialize_index(first_item)
                        print(f"✅ FAISS index deserialized from bytes")
                        print(f"   Vectors: {FAISS_INDEX.ntotal}")
                        print(f"   KB docs: {len(KB_DATA)}")
                    except Exception as e:
                        print(f"❌ Failed to deserialize index bytes: {e}")
                        raise RuntimeError(
                            f"Failed to deserialize FAISS index: {e}\n"
                            f"Please rebuild using: python build_faiss_index.py"
                        )
                
                # Otherwise it's a direct FAISS object (LEGACY - DANGEROUS!)
                else:
                    print(f"📦 Detected old tuple format with direct object")
                    print(f"⚠️ WARNING: Direct FAISS objects are not compatible across versions")
                    
                    # Try to use it, but expect it might fail
                    try:
                        FAISS_INDEX = first_item
                        # Test if it works
                        num_vectors = FAISS_INDEX.ntotal
                        print(f"✅ FAISS index appears valid ({num_vectors} vectors)")
                        print(f"   KB docs: {len(KB_DATA)}")
                        print(f"⚠️ However, this format may break across FAISS versions")
                        print(f"🔧 Recommended: Rebuild using: python build_faiss_index.py")
                    except Exception as e:
                        print(f"❌ FAISS index object is corrupted: {e}")
                        print(f"   This usually means FAISS version mismatch")
                        raise RuntimeError(
                            f"FAISS index is corrupted or incompatible (version mismatch).\n"
                            f"Error: {e}\n\n"
                            f"🔧 SOLUTION: Rebuild FAISS index using:\n"
                            f"   python build_faiss_index.py\n"
                        )
            
            # ========================================================================
            # FORMAT 3: Unknown Format
            # ========================================================================
            else:
                print(f"❌ Unknown pickle format: {type(data)}")
                if isinstance(data, dict):
                    print(f"   Dict keys: {list(data.keys())}")
                raise ValueError(
                    f"Unrecognized pickle format: {type(data)}.\n"
                    f"Please rebuild using: python build_faiss_index.py"
                )
            
            # Final validation
            if FAISS_INDEX is None or KB_DATA is None:
                raise RuntimeError("Failed to load FAISS index or KB data")
            
            print(f"✅ FAISS index ready: {FAISS_INDEX.ntotal} vectors")
            print(f"✅ Knowledge base ready: {len(KB_DATA)} documents")
            
        except FileNotFoundError:
            print(f"❌ FAISS index file not found: {settings.FAISS_INDEX_PATH}")
            print(f"⚠️ Make sure models are uploaded to HuggingFace Hub: {settings.HF_MODEL_REPO}")
            raise
        except RuntimeError:
            raise  # Re-raise our custom error with instructions
        except Exception as e:
            print(f"❌ Unexpected error loading FAISS index: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to load FAISS index: {e}\n"
                f"Please rebuild using: python build_faiss_index.py"
            )
    
    return FAISS_INDEX, KB_DATA


def retrieve_documents(
    query: str, 
    top_k: int = None, 
    min_similarity: float = None
) -> List[Dict]:
    """
    Retrieve top-k documents for a query using custom retriever + FAISS.
    
    Args:
        query: User query text
        top_k: Number of documents to retrieve (default from config)
        min_similarity: Minimum similarity threshold (default from config)
    
    Returns:
        List[Dict]: Retrieved documents with scores
            Each dict contains:
            - instruction: FAQ question
            - response: FAQ answer
            - category: Document category
            - intent: Document intent
            - score: Similarity score (0-1)
            - rank: Rank in results (1-indexed)
            - faq_id: Document ID
    """
    # Use config defaults if not provided
    if top_k is None:
        top_k = settings.TOP_K
    if min_similarity is None:
        min_similarity = settings.SIMILARITY_THRESHOLD
    
    # Validate query
    if not query or query.strip() == "":
        print("⚠️ Empty query provided")
        return []
    
    try:
        # Load models (cached, no overhead after first call)
        retriever = load_retriever()
        index, kb = load_faiss_index()
        
        # Step 1: Encode query
        query_embedding = retriever.encode_query(query)
        
        # Step 2: Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Step 3: Search in FAISS index
        similarities, indices = index.search(query_embedding, top_k)
        
        # Step 4: Check similarity threshold for top result
        if similarities[0][0] < min_similarity:
            print(f"🚫 NO_FETCH (similarity: {similarities[0][0]:.3f} < {min_similarity})")
            return []
        
        print(f"✅ FETCH (similarity: {similarities[0][0]:.3f} >= {min_similarity})")
        
        # Step 5: Format results
        results = []
        for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(kb):
                doc = kb[idx]
                results.append({
                    'instruction': doc.get('instruction', ''),
                    'response': doc.get('response', ''),
                    'category': doc.get('category', 'Unknown'),
                    'intent': doc.get('intent', 'Unknown'),
                    'score': float(similarity),
                    'rank': rank + 1,
                    'faq_id': doc.get('faq_id', f'doc_{idx}')
                })
        
        return results
    
    except RuntimeError as e:
        # Handle our custom errors with clear messages
        print(f"❌ Retrieval error: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return []


def format_context(retrieved_docs: List[Dict], max_context_length: int = None) -> str:
    """
    Format retrieved documents into context string for LLM.
    Prioritizes by score and limits total length.
    
    Args:
        retrieved_docs: List of retrieved documents
        max_context_length: Maximum context length in characters
    
    Returns:
        str: Formatted context string
    """
    if max_context_length is None:
        max_context_length = settings.MAX_CONTEXT_LENGTH
    
    if not retrieved_docs:
        return ""
    
    context_parts = []
    current_length = 0
    
    for doc in retrieved_docs:
        # Create context entry with None checks
        instruction = doc.get('instruction', '') or ''
        response = doc.get('response', '') or ''
        category = doc.get('category', 'N/A') or 'N/A'
        
        context_entry = f"[Rank {doc['rank']}, Score: {doc['score']:.3f}]\n"
        context_entry += f"Q: {instruction}\n"
        context_entry += f"A: {response}\n"
        context_entry += f"Category: {category}\n\n"
        
        # Check length limit
        if current_length + len(context_entry) > max_context_length:
            break
        
        context_parts.append(context_entry)
        current_length += len(context_entry)
    
    return "".join(context_parts)


# ============================================================================
# USAGE EXAMPLE (for reference)
# ============================================================================
"""
# In your service file:

from app.ml.retriever import retrieve_documents, format_context

# Retrieve documents
docs = retrieve_documents("What is my account balance?", top_k=5)

# Format context for LLM
context = format_context(docs)

# Use context in LLM prompt
prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
"""





























# """
# Custom Retriever with E5-Base-V2 and FAISS
# Trained with InfoNCE + Triplet Loss for banking domain

# This is adapted from your RAG.py with:
# - CustomSentenceTransformer (e5-base-v2)
# - Mean pooling + L2 normalization
# - FAISS vector search
# - Module-level caching (load once on startup)
# """

# import os
# import json
# import pickle
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import faiss
# import numpy as np
# from typing import List, Dict, Optional
# from transformers import AutoTokenizer, AutoModel

# from app.config import settings


# # ============================================================================
# # CUSTOM SENTENCE TRANSFORMER (From RAG.py)
# # ============================================================================

# class CustomSentenceTransformer(nn.Module):
#     """
#     Custom SentenceTransformer matching your training code.
#     Uses e5-base-v2 with mean pooling and L2 normalization.
    
#     Training Details:
#     - Base model: intfloat/e5-base-v2
#     - Loss: InfoNCE + Triplet Loss
#     - Pooling: Mean pooling on last hidden state
#     - Normalization: L2 normalization
#     """
    
#     def __init__(self, model_name: str = "intfloat/e5-base-v2"):
#         super().__init__()
#         # Load pre-trained e5-base-v2 encoder
#         self.encoder = AutoModel.from_pretrained(model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.config = self.encoder.config
    
#     def forward(self, input_ids, attention_mask):
#         """
#         Forward pass through BERT encoder.
        
#         Args:
#             input_ids: Tokenized input IDs
#             attention_mask: Attention mask for padding
        
#         Returns:
#             torch.Tensor: L2-normalized embeddings (shape: [batch_size, 768])
#         """
#         # Get BERT outputs
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
#         # Mean pooling - same as training
#         # Take hidden states from last layer
#         token_embeddings = outputs.last_hidden_state
        
#         # Expand attention mask to match token embeddings shape
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
#         # Sum embeddings (weighted by attention mask) and divide by sum of mask
#         embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#             input_mask_expanded.sum(1), min=1e-9
#         )
        
#         # L2 normalize embeddings - same as training
#         embeddings = F.normalize(embeddings, p=2, dim=1)
        
#         return embeddings
    
#     def encode(
#         self, 
#         sentences: List[str], 
#         batch_size: int = 32, 
#         convert_to_numpy: bool = True,
#         show_progress_bar: bool = False
#     ) -> np.ndarray:
#         """
#         Encode sentences using the same method as training.
#         Adds 'query: ' prefix for e5-base-v2 compatibility.
        
#         Args:
#             sentences: List of sentences to encode
#             batch_size: Batch size for encoding
#             convert_to_numpy: Whether to convert to numpy array
#             show_progress_bar: Whether to show progress bar
        
#         Returns:
#             np.ndarray: Encoded embeddings (shape: [num_sentences, 768])
#         """
#         self.eval()  # Set model to evaluation mode
        
#         # Handle single string input
#         if isinstance(sentences, str):
#             sentences = [sentences]
        
#         # Add 'query: ' prefix for e5-base-v2 (required by model)
#         # Handle None values and empty strings
#         processed_sentences = []
#         for sentence in sentences:
#             if sentence is None:
#                 processed_sentences.append("query: ")  # Default empty query
#             elif isinstance(sentence, str):
#                 processed_sentences.append(f"query: {sentence.strip()}")
#             else:
#                 processed_sentences.append(f"query: {str(sentence)}")
        
#         all_embeddings = []
        
#         # Encode in batches
#         with torch.no_grad():  # No gradient computation
#             for i in range(0, len(processed_sentences), batch_size):
#                 batch_sentences = processed_sentences[i:i + batch_size]
                
#                 # Tokenize batch
#                 tokens = self.tokenizer(
#                     batch_sentences,
#                     truncation=True,
#                     padding=True,
#                     max_length=128,  # Same as training
#                     return_tensors='pt'
#                 ).to(next(self.parameters()).device)
                
#                 # Get embeddings
#                 embeddings = self.forward(tokens['input_ids'], tokens['attention_mask'])
                
#                 # Convert to numpy if requested
#                 if convert_to_numpy:
#                     embeddings = embeddings.cpu().numpy()
                
#                 all_embeddings.append(embeddings)
        
#         # Combine all batches
#         if convert_to_numpy:
#             all_embeddings = np.vstack(all_embeddings)
#         else:
#             all_embeddings = torch.cat(all_embeddings, dim=0)
        
#         return all_embeddings


# # ============================================================================
# # CUSTOM RETRIEVER MODEL (Wrapper)
# # ============================================================================

# class CustomRetrieverModel:
#     """
#     Wrapper for your custom trained retriever model.
#     Handles both knowledge base documents and query encoding.
#     """
    
#     def __init__(self, model_path: str, device: str = "cpu"):
#         """
#         Initialize retriever model.
        
#         Args:
#             model_path: Path to trained model weights (.pth file)
#             device: Device to load model on ('cpu' or 'cuda')
#         """
#         self.device = device
        
#         # Create model instance
#         self.model = CustomSentenceTransformer("intfloat/e5-base-v2").to(device)
        
#         # Load your trained weights
#         try:
#             state_dict = torch.load(model_path, map_location=device)
#             self.model.load_state_dict(state_dict)
#             print(f"✅ Custom retriever model loaded from {model_path}")
#         except Exception as e:
#             print(f"❌ Failed to load custom model: {e}")
#             print("🔄 Using base e5-base-v2 model (not trained)...")
        
#         # Set to evaluation mode
#         self.model.eval()
    
#     def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
#         """
#         Encode knowledge base documents.
#         These are the responses/instructions we're retrieving.
        
#         Args:
#             documents: List of document texts
#             batch_size: Batch size for encoding
        
#         Returns:
#             np.ndarray: Document embeddings (shape: [num_docs, 768])
#         """
#         return self.model.encode(documents, batch_size=batch_size, convert_to_numpy=True)
    
#     def encode_query(self, query: str) -> np.ndarray:
#         """
#         Encode user query for retrieval.
        
#         Args:
#             query: User query text
        
#         Returns:
#             np.ndarray: Query embedding (shape: [1, 768])
#         """
#         return self.model.encode([query], convert_to_numpy=True)


# # ============================================================================
# # MODULE-LEVEL CACHING (Load once on import)
# # ============================================================================

# # Global variables for caching
# RETRIEVER_MODEL: Optional[CustomRetrieverModel] = None
# FAISS_INDEX: Optional[faiss.Index] = None
# KB_DATA: Optional[List[Dict]] = None





# # =============================================================================================
# # Latest version given by perplexity, should work, if not then use one of the other versions.
# # =============================================================================================

# def load_retriever() -> CustomRetrieverModel:
#     """
#     Load custom retriever model (called once on startup).
#     Downloads from HuggingFace Hub if not present locally.
#     Uses module-level caching - model stays in RAM.
    
#     Returns:
#         CustomRetrieverModel: Loaded retriever model
#     """
#     global RETRIEVER_MODEL
    
#     if RETRIEVER_MODEL is None:
#         # Download model from HF Hub if needed (for deployment)
#         settings.download_model_if_needed(
#             hf_filename="models/best_retriever_model.pth",
#             local_path=settings.RETRIEVER_MODEL_PATH
#         )
        
#         print(f"Loading custom retriever from {settings.RETRIEVER_MODEL_PATH}...")
        
#         RETRIEVER_MODEL = CustomRetrieverModel(
#             model_path=settings.RETRIEVER_MODEL_PATH,
#             device=settings.DEVICE
#         )
        
#         print("✅ Retriever model loaded and cached")
    
#     return RETRIEVER_MODEL









# # ===========================================================================
# # This version is used in the code, atleast for localhost testing
# # ===========================================================================

# # def load_retriever() -> CustomRetrieverModel:
# #     """
# #     Load custom retriever model (called once on startup).
# #     Uses module-level caching - model stays in RAM.
    
# #     Returns:
# #         CustomRetrieverModel: Loaded retriever model
# #     """
# #     global RETRIEVER_MODEL
    
# #     if RETRIEVER_MODEL is None:
# #         print(f"Loading custom retriever from {settings.RETRIEVER_MODEL_PATH}...")
# #         RETRIEVER_MODEL = CustomRetrieverModel(
# #             model_path=settings.RETRIEVER_MODEL_PATH,
# #             device=settings.DEVICE
# #         )
# #         print("✅ Retriever model loaded and cached")
    
# #     return RETRIEVER_MODEL

# # ==================================================================================================
# # Latest version given by perplexity, should work, if not then use one of the other versions.
# # ==================================================================================================
# def load_faiss_index():
#     """
#     Load FAISS index + knowledge base from pickle file.
#     Downloads from HuggingFace Hub if not present locally.
#     Uses module-level caching - loaded once on startup.
    
#     Returns:
#         tuple: (faiss.Index, List[Dict]) - FAISS index and KB data
#     """
#     global FAISS_INDEX, KB_DATA
    
#     if FAISS_INDEX is None or KB_DATA is None:
#         # Download FAISS index from HF Hub if needed (for deployment)
#         settings.download_model_if_needed(
#             hf_filename="models/faiss_index.pkl",
#             local_path=settings.FAISS_INDEX_PATH
#         )
        
#         # Download knowledge base from HF Hub if needed (for deployment)
#         settings.download_model_if_needed(
#             hf_filename="data/final_knowledge_base.jsonl",
#             local_path=settings.KB_PATH
#         )
        
#         print(f"Loading FAISS index from {settings.FAISS_INDEX_PATH}...")
        
#         try:
#             # Load pickled data
#             with open(settings.FAISS_INDEX_PATH, 'rb') as f:
#                 loaded_data = pickle.load(f)
            
#             print(f"📦 Pickle loaded successfully")
            
#             # ✅ Handle both formats: (index, kb_data) OR (index_bytes, kb_data)
#             if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
#                 first_item, KB_DATA = loaded_data
                
#                 # Check if first item is bytes (new format) - SAFE to check
#                 if isinstance(first_item, bytes):
#                     print("📦 Detected new format (serialized bytes)")
#                     FAISS_INDEX = faiss.deserialize_index(first_item)
#                     print(f"✅ FAISS index deserialized successfully")
                
#                 # Otherwise assume it's old format and try to use it
#                 else:
#                     print(f"📦 Detected old format (attempting to use directly)")
                    
#                     # ❌ DON'T use hasattr() - it crashes on corrupted FAISS!
#                     # Instead, try to use it and catch errors
#                     try:
#                         FAISS_INDEX = first_item
#                         # Test if it works by accessing ntotal
#                         num_vectors = FAISS_INDEX.ntotal
#                         print(f"✅ FAISS index is valid ({num_vectors} vectors)")
#                     except Exception as e:
#                         print(f"❌ FAISS index object is corrupted: {e}")
#                         print(f"⚠️ This pickle was created with incompatible FAISS version")
#                         print(f"")
#                         print(f"🔧 SOLUTION: Rebuild FAISS index using:")
#                         print(f"   python build_faiss_index.py")
#                         print(f"")
#                         raise RuntimeError(
#                             f"FAISS index is corrupted or incompatible (FAISS version mismatch). "
#                             f"Please rebuild using: python build_faiss_index.py"
#                         )
#             else:
#                 raise ValueError(f"Invalid pickle format: expected tuple, got {type(loaded_data)}")
            
#             print(f"✅ FAISS index loaded: {FAISS_INDEX.ntotal} vectors")
#             print(f"✅ Knowledge base loaded: {len(KB_DATA)} documents")
            
#         except FileNotFoundError:
#             print(f"❌ FAISS index file not found: {settings.FAISS_INDEX_PATH}")
#             print(f"⚠️ Make sure models are uploaded to HuggingFace Hub: {settings.HF_MODEL_REPO}")
#             raise
#         except RuntimeError:
#             raise  # Re-raise our custom error
#         except Exception as e:
#             print(f"❌ Failed to load FAISS index: {e}")
#             import traceback
#             traceback.print_exc()
#             raise
    
#     return FAISS_INDEX, KB_DATA



# # ==================================================================================================
# # Second Latest version given by perplexity, should work, if not then use one of the other versions.
# # ==================================================================================================

# # def load_faiss_index():
# #     """
# #     Load FAISS index + knowledge base from pickle file.
# #     Downloads from HuggingFace Hub if not present locally.
# #     Uses module-level caching - loaded once on startup.
    
# #     Returns:
# #         tuple: (faiss.Index, List[Dict]) - FAISS index and KB data
# #     """
# #     global FAISS_INDEX, KB_DATA
    
# #     if FAISS_INDEX is None or KB_DATA is None:
# #         # Download FAISS index from HF Hub if needed (for deployment)
# #         settings.download_model_if_needed(
# #             hf_filename="models/faiss_index.pkl",
# #             local_path=settings.FAISS_INDEX_PATH
# #         )
        
# #         # Download knowledge base from HF Hub if needed (for deployment)
# #         settings.download_model_if_needed(
# #             hf_filename="data/final_knowledge_base.jsonl",
# #             local_path=settings.KB_PATH
# #         )
        
# #         print(f"Loading FAISS index from {settings.FAISS_INDEX_PATH}...")
        
# #         try:
# #             # Load pickled FAISS index + KB data
# #             with open(settings.FAISS_INDEX_PATH, 'rb') as f:
# #                 FAISS_INDEX, KB_DATA = pickle.load(f)
            
# #             print(f"✅ FAISS index loaded: {FAISS_INDEX.ntotal} vectors")
# #             print(f"✅ Knowledge base loaded: {len(KB_DATA)} documents")
            
# #         except FileNotFoundError:
# #             print(f"❌ FAISS index file not found: {settings.FAISS_INDEX_PATH}")
# #             print(f"⚠️ Make sure models are uploaded to HuggingFace Hub: {settings.HF_MODEL_REPO}")
# #             raise
# #         except Exception as e:
# #             print(f"❌ Failed to load FAISS index: {e}")
# #             raise
    
# #     return FAISS_INDEX, KB_DATA



# # ===========================================================================
# # This version is used in the code, atleast for localhost testing
# # ===========================================================================

# # def load_faiss_index():
# #     """
# #     Load FAISS index + knowledge base from pickle file.
# #     Uses module-level caching - loaded once on startup.
    
# #     Returns:
# #         tuple: (faiss.Index, List[Dict]) - FAISS index and KB data
# #     """
# #     global FAISS_INDEX, KB_DATA
    
# #     if FAISS_INDEX is None or KB_DATA is None:
# #         print(f"Loading FAISS index from {settings.FAISS_INDEX_PATH}...")
        
# #         try:
# #             # Load pickled FAISS index + KB data
# #             with open(settings.FAISS_INDEX_PATH, 'rb') as f:
# #                 FAISS_INDEX, KB_DATA = pickle.load(f)
            
# #             print(f"✅ FAISS index loaded: {FAISS_INDEX.ntotal} vectors")
# #             print(f"✅ Knowledge base loaded: {len(KB_DATA)} documents")
        
# #         except FileNotFoundError:
# #             print(f"❌ FAISS index file not found: {settings.FAISS_INDEX_PATH}")
# #             print("⚠️  You need to create the FAISS index first!")
# #             raise
        
# #         except Exception as e:
# #             print(f"❌ Failed to load FAISS index: {e}")
# #             raise
    
# #     return FAISS_INDEX, KB_DATA


# # ============================================================================
# # RETRIEVAL FUNCTIONS
# # ============================================================================

# def retrieve_documents(
#     query: str, 
#     top_k: int = None, 
#     min_similarity: float = None
# ) -> List[Dict]:
#     """
#     Retrieve top-k documents for a query using custom retriever + FAISS.
    
#     Args:
#         query: User query text
#         top_k: Number of documents to retrieve (default from config)
#         min_similarity: Minimum similarity threshold (default from config)
    
#     Returns:
#         List[Dict]: Retrieved documents with scores
#             Each dict contains:
#             - instruction: FAQ question
#             - response: FAQ answer
#             - category: Document category
#             - intent: Document intent
#             - score: Similarity score (0-1)
#             - rank: Rank in results (1-indexed)
#             - faq_id: Document ID
#     """
#     # Use config defaults if not provided
#     if top_k is None:
#         top_k = settings.TOP_K
#     if min_similarity is None:
#         min_similarity = settings.SIMILARITY_THRESHOLD
    
#     # Validate query
#     if not query or query.strip() == "":
#         print("⚠️ Empty query provided")
#         return []
    
#     # Load models (cached, no overhead after first call)
#     retriever = load_retriever()
#     index, kb = load_faiss_index()
    
#     try:
#         # Step 1: Encode query
#         query_embedding = retriever.encode_query(query)
        
#         # Step 2: Normalize for cosine similarity
#         faiss.normalize_L2(query_embedding)
        
#         # Step 3: Search in FAISS index
#         similarities, indices = index.search(query_embedding, top_k)
        
#         # Step 4: Check similarity threshold for top result
#         if similarities[0][0] < min_similarity:
#             print(f"🚫 NO_FETCH (similarity: {similarities[0][0]:.3f} < {min_similarity})")
#             return []
        
#         print(f"✅ FETCH (similarity: {similarities[0][0]:.3f} >= {min_similarity})")
        
#         # Step 5: Format results
#         results = []
#         for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
#             if idx < len(kb):
#                 doc = kb[idx]
#                 results.append({
#                     'instruction': doc.get('instruction', ''),
#                     'response': doc.get('response', ''),
#                     'category': doc.get('category', 'Unknown'),
#                     'intent': doc.get('intent', 'Unknown'),
#                     'score': float(similarity),
#                     'rank': rank + 1,
#                     'faq_id': doc.get('faq_id', f'doc_{idx}')
#                 })
        
#         return results
    
#     except Exception as e:
#         print(f"❌ Retrieval error: {e}")
#         import traceback
#         traceback.print_exc()
#         return []


# def format_context(retrieved_docs: List[Dict], max_context_length: int = None) -> str:
#     """
#     Format retrieved documents into context string for LLM.
#     Prioritizes by score and limits total length.
    
#     Args:
#         retrieved_docs: List of retrieved documents
#         max_context_length: Maximum context length in characters
    
#     Returns:
#         str: Formatted context string
#     """
#     if max_context_length is None:
#         max_context_length = settings.MAX_CONTEXT_LENGTH
    
#     if not retrieved_docs:
#         return ""
    
#     context_parts = []
#     current_length = 0
    
#     for doc in retrieved_docs:
#         # Create context entry with None checks
#         instruction = doc.get('instruction', '') or ''
#         response = doc.get('response', '') or ''
#         category = doc.get('category', 'N/A') or 'N/A'
        
#         context_entry = f"[Rank {doc['rank']}, Score: {doc['score']:.3f}]\n"
#         context_entry += f"Q: {instruction}\n"
#         context_entry += f"A: {response}\n"
#         context_entry += f"Category: {category}\n\n"
        
#         # Check length limit
#         if current_length + len(context_entry) > max_context_length:
#             break
        
#         context_parts.append(context_entry)
#         current_length += len(context_entry)
    
#     return "".join(context_parts)


# # ============================================================================
# # USAGE EXAMPLE (for reference)
# # ============================================================================
# """
# # In your service file:

# from app.ml.retriever import retrieve_documents, format_context

# # Retrieve documents
# docs = retrieve_documents("What is my account balance?", top_k=5)

# # Format context for LLM
# context = format_context(docs)

# # Use context in LLM prompt
# prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
# """