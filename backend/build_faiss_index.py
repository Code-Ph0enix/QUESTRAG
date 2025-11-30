# # ======================================================================================================
# # CLAUDE ORIGINAL VERSION 
# # ======================================================================================================

# """
# Build FAISS Index from Scratch - FIXED VERSION
# Creates faiss_index.pkl with proper serialization

# Run this ONCE before starting the backend:
#     python build_faiss_index.py

# Author: Banking RAG Chatbot
# Date: November 2025
# """

# # Suppress warnings
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import warnings
# warnings.filterwarnings('ignore')

# import pickle
# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import faiss
# import numpy as np
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModel
# from typing import List

# # ============================================================================
# # CONFIGURATION - UPDATE THESE PATHS!
# # ============================================================================

# # Where is your knowledge base JSONL file?
# KB_JSONL_FILE = "data/final_knowledge_base.jsonl"

# # Where is your trained retriever model?
# RETRIEVER_MODEL_PATH = "app/models/best_retriever_model.pth"

# # Where to save the output FAISS pickle?
# OUTPUT_PKL_FILE = "app/models/faiss_index.pkl"

# # Device (auto-detect GPU/CPU)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Batch size for encoding (reduce if you get OOM errors)
# BATCH_SIZE = 32

# # ============================================================================
# # CUSTOM SENTENCE TRANSFORMER (Same as retriever.py)
# # ============================================================================

# class CustomSentenceTransformer(nn.Module):
#     """
#     Custom SentenceTransformer - exact copy from retriever.py
#     Uses e5-base-v2 with mean pooling and L2 normalization
#     """
    
#     def __init__(self, model_name: str = "intfloat/e5-base-v2"):
#         super().__init__()
#         print(f"   Loading base model: {model_name}...")
#         self.encoder = AutoModel.from_pretrained(model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.config = self.encoder.config
#         print(f"   ✅ Base model loaded")
    
#     def forward(self, input_ids, attention_mask):
#         """Forward pass through BERT encoder"""
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
#         # Mean pooling
#         token_embeddings = outputs.last_hidden_state
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#             input_mask_expanded.sum(1), min=1e-9
#         )
        
#         # L2 normalize
#         embeddings = F.normalize(embeddings, p=2, dim=1)
#         return embeddings
    
#     def encode(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
#         """Encode sentences - same as training"""
#         self.eval()
#         if isinstance(sentences, str):
#             sentences = [sentences]
        
#         # Add 'query: ' prefix for e5-base-v2
#         processed_sentences = [f"query: {s.strip()}" for s in sentences]
        
#         all_embeddings = []
#         with torch.no_grad():
#             for i in range(0, len(processed_sentences), batch_size):
#                 batch_sentences = processed_sentences[i:i + batch_size]
                
#                 # Tokenize
#                 tokens = self.tokenizer(
#                     batch_sentences,
#                     truncation=True,
#                     padding=True,
#                     max_length=128,
#                     return_tensors='pt'
#                 ).to(self.encoder.device)
                
#                 # Get embeddings
#                 embeddings = self.forward(tokens['input_ids'], tokens['attention_mask'])
#                 all_embeddings.append(embeddings.cpu().numpy())
        
#         return np.vstack(all_embeddings)

# # ============================================================================
# # RETRIEVER MODEL (Wrapper)
# # ============================================================================

# class RetrieverModel:
#     """Wrapper for trained retriever model"""
    
#     def __init__(self, model_path: str, device: str = "cpu"):
#         print(f"\n🤖 Loading retriever model...")
#         print(f"   Device: {device}")
#         self.device = device
#         self.model = CustomSentenceTransformer("intfloat/e5-base-v2").to(device)
        
#         # Load trained weights
#         print(f"   Loading weights from: {model_path}")
#         try:
#             state_dict = torch.load(model_path, map_location=device)
#             self.model.load_state_dict(state_dict)
#             print(f"   ✅ Trained weights loaded")
#         except Exception as e:
#             print(f"   ⚠️ Warning: Could not load trained weights: {e}")
#             print(f"   Using base e5-base-v2 model instead")
        
#         self.model.eval()
    
#     def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
#         """Encode documents"""
#         return self.model.encode(documents, batch_size=batch_size)

# # ============================================================================
# # MAIN: BUILD FAISS INDEX
# # ============================================================================

# def build_faiss_index():
#     """Main function to build FAISS index from scratch"""
    
#     print("=" * 80)
#     print("🏗️  BUILDING FAISS INDEX FROM SCRATCH")
#     print("=" * 80)
    
#     # ========================================================================
#     # STEP 1: LOAD KNOWLEDGE BASE
#     # ========================================================================
#     print(f"\n📖 STEP 1: Loading knowledge base...")
#     print(f"   File: {KB_JSONL_FILE}")
    
#     if not os.path.exists(KB_JSONL_FILE):
#         print(f"   ❌ ERROR: File not found!")
#         print(f"   Please copy your knowledge base to: {KB_JSONL_FILE}")
#         return False
    
#     kb_data = []
#     with open(KB_JSONL_FILE, 'r', encoding='utf-8') as f:
#         for line_num, line in enumerate(f, 1):
#             try:
#                 kb_data.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 print(f"   ⚠️ Warning: Skipping invalid JSON on line {line_num}: {e}")
    
#     print(f"   ✅ Loaded {len(kb_data)} documents")
    
#     if len(kb_data) == 0:
#         print(f"   ❌ ERROR: Knowledge base is empty!")
#         return False
    
#     # ========================================================================
#     # STEP 2: PREPARE DOCUMENTS FOR ENCODING
#     # ========================================================================
#     print(f"\n📝 STEP 2: Preparing documents for encoding...")
    
#     documents = []
#     for i, item in enumerate(kb_data):
#         # Combine instruction + response for embedding (same as training)
#         instruction = item.get('instruction', '')
#         response = item.get('response', '')
        
#         # Create combined text
#         if instruction and response:
#             text = f"{instruction} {response}"
#         elif instruction:
#             text = instruction
#         elif response:
#             text = response
#         else:
#             print(f"   ⚠️ Warning: Document {i} has no content, using placeholder")
#             text = "empty document"
        
#         documents.append(text)
    
#     print(f"   ✅ Prepared {len(documents)} documents for encoding")
#     print(f"   Average length: {sum(len(d) for d in documents) / len(documents):.1f} chars")
    
#     # ========================================================================
#     # STEP 3: LOAD RETRIEVER AND ENCODE DOCUMENTS
#     # ========================================================================
#     print(f"\n🔮 STEP 3: Encoding documents with trained retriever...")
    
#     if not os.path.exists(RETRIEVER_MODEL_PATH):
#         print(f"   ❌ ERROR: Retriever model not found!")
#         print(f"   Please copy your trained model to: {RETRIEVER_MODEL_PATH}")
#         return False
    
#     # Load retriever
#     retriever = RetrieverModel(RETRIEVER_MODEL_PATH, device=DEVICE)
    
#     # Encode all documents
#     print(f"   Encoding {len(documents)} documents...")
#     print(f"   Batch size: {BATCH_SIZE}")
#     print(f"   This may take a few minutes... ☕")
    
#     try:
#         embeddings = retriever.encode_documents(documents, batch_size=BATCH_SIZE)
#         print(f"   ✅ Encoded {embeddings.shape[0]} documents")
#         print(f"   Embedding dimension: {embeddings.shape[1]}")
#     except Exception as e:
#         print(f"   ❌ ERROR during encoding: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
    
#     # ========================================================================
#     # STEP 4: BUILD FAISS INDEX
#     # ========================================================================
#     print(f"\n🔍 STEP 4: Building FAISS index...")
    
#     dimension = embeddings.shape[1]
#     print(f"   Dimension: {dimension}")
    
#     # Create FAISS index (Inner Product = Cosine similarity after normalization)
#     print(f"   Creating IndexFlatIP...")
#     index = faiss.IndexFlatIP(dimension)
    
#     # Normalize embeddings for cosine similarity
#     print(f"   Normalizing embeddings...")
#     faiss.normalize_L2(embeddings)
    
#     # Add to index
#     print(f"   Adding {embeddings.shape[0]} vectors to FAISS index...")
#     index.add(embeddings.astype('float32'))
    
#     print(f"   ✅ FAISS index built successfully")
#     print(f"   Total vectors: {index.ntotal}")
    
#     # ========================================================================
#     # STEP 5: SAVE WITH COMPATIBLE FORMAT
#     # ========================================================================
#     print(f"\n💾 STEP 5: Saving with compatible format...")
    
#     # Create models directory if it doesn't exist
#     os.makedirs(os.path.dirname(OUTPUT_PKL_FILE), exist_ok=True)
    
#     try:
#         # ✅ NEW FORMAT: Save as dictionary with version info
#         print(f"   Creating compatible save format...")
#         save_data = {
#             'format_version': 'v2',  # Mark as new format
#             'index_bytes': faiss.serialize_index(index),  # FAISS bytes
#             'kb_data': kb_data,  # Knowledge base
#             'dimension': dimension,
#             'num_vectors': index.ntotal,
#             'faiss_version': faiss.__version__
#         }
        
#         print(f"   Pickling dictionary...")
#         with open(OUTPUT_PKL_FILE, 'wb') as f:
#             pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
#         file_size_mb = Path(OUTPUT_PKL_FILE).stat().st_size / (1024 * 1024)
#         print(f"   ✅ Saved: {OUTPUT_PKL_FILE}")
#         print(f"   File size: {file_size_mb:.2f} MB")
#         print(f"   Format: v2 (compatible)")
#         print(f"   FAISS version used: {faiss.__version__}")
#     except Exception as e:
#         print(f"   ❌ ERROR saving pickle: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
    
#     # ========================================================================
#     # STEP 6: VERIFY SAVED FILE
#     # ========================================================================
#     print(f"\n✅ STEP 6: Verifying saved file...")
    
#     try:
#         with open(OUTPUT_PKL_FILE, 'rb') as f:
#             loaded_data = pickle.load(f)
        
#         # Check format
#         if isinstance(loaded_data, dict) and loaded_data.get('format_version') == 'v2':
#             print(f"   ✅ Format: v2 (compatible)")
            
#             # Deserialize FAISS index
#             loaded_index = faiss.deserialize_index(loaded_data['index_bytes'])
#             loaded_kb = loaded_data['kb_data']
            
#             print(f"   ✅ Verification successful")
#             print(f"   Index vectors: {loaded_index.ntotal}")
#             print(f"   KB documents: {len(loaded_kb)}")
#             print(f"   Dimension: {loaded_data['dimension']}")
            
#             if loaded_index.ntotal != len(loaded_kb):
#                 print(f"   ⚠️ WARNING: Size mismatch detected!")
#         else:
#             print(f"   ⚠️ WARNING: Unexpected format detected")
            
#     except Exception as e:
#         print(f"   ❌ ERROR verifying file: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
    
#     # ========================================================================
#     # SUCCESS!
#     # ========================================================================
#     print("\n" + "=" * 80)
#     print("🎉 SUCCESS! FAISS INDEX BUILT AND SAVED")
#     print("=" * 80)
#     print(f"\n📊 Summary:")
#     print(f"   Documents: {len(kb_data)}")
#     print(f"   Vectors: {index.ntotal}")
#     print(f"   Dimension: {dimension}")
#     print(f"   File: {OUTPUT_PKL_FILE} ({file_size_mb:.2f} MB)")
#     print(f"   Format: v2 (compatible with all FAISS versions)")
#     print(f"   FAISS version: {faiss.__version__}")
#     print(f"\n🚀 Next steps:")
#     print(f"   1. Upload {OUTPUT_PKL_FILE} to HuggingFace Hub")
#     print(f"   2. Update your retriever loading code (see below)")
#     print(f"   3. Restart your backend")
#     print(f"   4. Test retrieval - should work now!")
#     print("\n" + "=" * 80)
#     print("📝 IMPORTANT: Update your loading code to handle v2 format:")
#     print("=" * 80)
#     print("""
# # In your retriever.py or wherever you load the index:

# with open('faiss_index.pkl', 'rb') as f:
#     data = pickle.load(f)

# # Check format
# if isinstance(data, dict) and data.get('format_version') == 'v2':
#     # NEW FORMAT (v2)
#     index = faiss.deserialize_index(data['index_bytes'])
#     kb_data = data['kb_data']
#     print(f"✅ Loaded v2 format: {index.ntotal} vectors")
# else:
#     # OLD FORMAT (try to handle gracefully)
#     print("⚠️ Old format detected, please rebuild")
#     raise ValueError("Please rebuild FAISS index with new script")
#     """)
#     print("=" * 80 + "\n")
    
#     return True

# # ============================================================================
# # RUN SCRIPT
# # ============================================================================

# if __name__ == "__main__":
#     success = build_faiss_index()
    
#     if not success:
#         print("\n" + "=" * 80)
#         print("❌ FAILED TO BUILD FAISS INDEX")
#         print("=" * 80)
#         print("\nPlease check:")
#         print("1. Knowledge base file exists: data/final_knowledge_base.jsonl")
#         print("2. Retriever model exists: app/models/best_retriever_model.pth")
#         print("3. You have enough RAM (embeddings need ~1GB for 10k docs)")
#         print("=" * 80 + "\n")
#         exit(1)




























# ======================================================================================================
# CLAUDE OPTIMIZED VERSION FOR FASTER ENCODING PROCESS
# ======================================================================================================
"""
Build FAISS Index from Scratch - OPTIMIZED VERSION
⚡ FASTER with GPU auto-detection and larger batches

Run this ONCE before starting the backend:
    python build_faiss_index.py

Optimizations:
- Auto-detects GPU (10-50x faster than CPU)
- Larger batch sizes (256 instead of 32)
- Progress bars
- Memory efficient processing

Author: Banking RAG Chatbot
Date: November 2025
"""

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import List
from tqdm import tqdm  # Progress bars

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS!
# ============================================================================

KB_JSONL_FILE = "data/final_knowledge_base.jsonl"
RETRIEVER_MODEL_PATH = "app/models/best_retriever_model.pth"
OUTPUT_PKL_FILE = "app/models/faiss_index.pkl"

# ⚡ AUTO-DETECT GPU (will use GPU if available, CPU otherwise)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ⚡ OPTIMIZED BATCH SIZE
# - GPU: Use 256 (or even 512 if you have 16GB+ VRAM)
# - CPU: Use 32 (to avoid RAM issues)
BATCH_SIZE = 256 if DEVICE == "cuda" else 32

print(f"🔧 Configuration:")
print(f"   Device: {DEVICE}")
print(f"   Batch size: {BATCH_SIZE}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# CUSTOM SENTENCE TRANSFORMER
# ============================================================================

class CustomSentenceTransformer(nn.Module):
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        super().__init__()
        print(f"   Loading base model: {model_name}...")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = self.encoder.config
        print(f"   ✅ Base model loaded")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def encode(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """⚡ Optimized encoding with progress bar"""
        self.eval()
        if isinstance(sentences, str):
            sentences = [sentences]
        
        processed_sentences = [f"query: {s.strip()}" for s in sentences]
        all_embeddings = []
        
        # ⚡ Progress bar
        num_batches = (len(processed_sentences) + batch_size - 1) // batch_size
        pbar = tqdm(total=len(processed_sentences), desc="   Encoding", unit="docs")
        
        with torch.no_grad():
            for i in range(0, len(processed_sentences), batch_size):
                batch_sentences = processed_sentences[i:i + batch_size]
                
                tokens = self.tokenizer(
                    batch_sentences,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.encoder.device)
                
                embeddings = self.forward(tokens['input_ids'], tokens['attention_mask'])
                all_embeddings.append(embeddings.cpu().numpy())
                
                pbar.update(len(batch_sentences))
        
        pbar.close()
        return np.vstack(all_embeddings)

# ============================================================================
# RETRIEVER MODEL
# ============================================================================

class RetrieverModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        print(f"\n🤖 Loading retriever model...")
        print(f"   Device: {device}")
        self.device = device
        self.model = CustomSentenceTransformer("intfloat/e5-base-v2").to(device)
        
        print(f"   Loading weights from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"   ✅ Trained weights loaded")
        except Exception as e:
            print(f"   ⚠️ Warning: Could not load trained weights: {e}")
            print(f"   Using base e5-base-v2 model instead")
        
        self.model.eval()
        
        # ⚡ Mixed precision for faster inference on GPU
        if device == "cuda":
            self.model = self.model.half()  # FP16
            print(f"   ⚡ Using FP16 (mixed precision) for faster encoding")
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(documents, batch_size=batch_size)

# ============================================================================
# MAIN: BUILD FAISS INDEX
# ============================================================================

def build_faiss_index():
    print("=" * 80)
    print("🏗️  BUILDING FAISS INDEX FROM SCRATCH (OPTIMIZED)")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: LOAD KNOWLEDGE BASE
    # ========================================================================
    print(f"\n📖 STEP 1: Loading knowledge base...")
    print(f"   File: {KB_JSONL_FILE}")
    
    if not os.path.exists(KB_JSONL_FILE):
        print(f"   ❌ ERROR: File not found!")
        return False
    
    kb_data = []
    with open(KB_JSONL_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                kb_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"   ⚠️ Warning: Skipping invalid JSON on line {line_num}")
    
    print(f"   ✅ Loaded {len(kb_data)} documents")
    
    if len(kb_data) == 0:
        print(f"   ❌ ERROR: Knowledge base is empty!")
        return False
    
    # ========================================================================
    # STEP 2: PREPARE DOCUMENTS
    # ========================================================================
    print(f"\n📝 STEP 2: Preparing documents for encoding...")
    
    documents = []
    for i, item in enumerate(kb_data):
        instruction = item.get('instruction', '')
        response = item.get('response', '')
        
        if instruction and response:
            text = f"{instruction} {response}"
        elif instruction:
            text = instruction
        elif response:
            text = response
        else:
            text = "empty document"
        
        documents.append(text)
    
    print(f"   ✅ Prepared {len(documents)} documents")
    avg_len = sum(len(d) for d in documents) / len(documents)
    print(f"   Average length: {avg_len:.1f} chars")
    
    # ⚡ Estimate encoding time
    if DEVICE == "cuda":
        est_time = len(documents) / (BATCH_SIZE * 100)  # ~100 docs/sec on GPU
        print(f"   ⚡ Estimated time on GPU: {est_time:.1f} seconds")
    else:
        est_time = len(documents) / (BATCH_SIZE * 2)  # ~2 docs/sec on CPU
        print(f"   ⏳ Estimated time on CPU: {est_time/60:.1f} minutes")
    
    # ========================================================================
    # STEP 3: ENCODE DOCUMENTS
    # ========================================================================
    print(f"\n🔮 STEP 3: Encoding documents with trained retriever...")
    
    if not os.path.exists(RETRIEVER_MODEL_PATH):
        print(f"   ❌ ERROR: Retriever model not found!")
        return False
    
    import time
    start_time = time.time()
    
    retriever = RetrieverModel(RETRIEVER_MODEL_PATH, device=DEVICE)
    
    print(f"   Encoding {len(documents)} documents...")
    print(f"   Batch size: {BATCH_SIZE}")
    
    try:
        embeddings = retriever.encode_documents(documents, batch_size=BATCH_SIZE)
        
        elapsed = time.time() - start_time
        print(f"   ✅ Encoded {embeddings.shape[0]} documents in {elapsed:.1f}s")
        print(f"   Speed: {len(documents)/elapsed:.1f} docs/sec")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
    except Exception as e:
        print(f"   ❌ ERROR during encoding: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # STEP 4: BUILD FAISS INDEX
    # ========================================================================
    print(f"\n🔍 STEP 4: Building FAISS index...")
    
    dimension = embeddings.shape[1]
    print(f"   Dimension: {dimension}")
    
    print(f"   Creating IndexFlatIP...")
    index = faiss.IndexFlatIP(dimension)
    
    print(f"   Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    print(f"   Adding {embeddings.shape[0]} vectors to FAISS index...")
    index.add(embeddings.astype('float32'))
    
    print(f"   ✅ FAISS index built successfully")
    print(f"   Total vectors: {index.ntotal}")
    
    # ========================================================================
    # STEP 5: SAVE WITH V2 FORMAT
    # ========================================================================
    print(f"\n💾 STEP 5: Saving with v2 format (compatible)...")
    
    os.makedirs(os.path.dirname(OUTPUT_PKL_FILE), exist_ok=True)
    
    try:
        print(f"   Creating v2 save format...")
        save_data = {
            'format_version': 'v2',
            'index_bytes': faiss.serialize_index(index),
            'kb_data': kb_data,
            'dimension': dimension,
            'num_vectors': index.ntotal,
            'faiss_version': faiss.__version__,
            'device_used': DEVICE,
            'build_time': elapsed
        }
        
        print(f"   Pickling...")
        with open(OUTPUT_PKL_FILE, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = Path(OUTPUT_PKL_FILE).stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved: {OUTPUT_PKL_FILE}")
        print(f"   File size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"   ❌ ERROR saving: {e}")
        return False
    
    # ========================================================================
    # STEP 6: VERIFY
    # ========================================================================
    print(f"\n✅ STEP 6: Verifying...")
    
    try:
        with open(OUTPUT_PKL_FILE, 'rb') as f:
            loaded = pickle.load(f)
        
        loaded_index = faiss.deserialize_index(loaded['index_bytes'])
        loaded_kb = loaded['kb_data']
        
        print(f"   ✅ Verification successful")
        print(f"   Index vectors: {loaded_index.ntotal}")
        print(f"   KB documents: {len(loaded_kb)}")
    except Exception as e:
        print(f"   ❌ ERROR verifying: {e}")
        return False
    
    # ========================================================================
    # SUCCESS!
    # ========================================================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("🎉 SUCCESS! FAISS INDEX BUILT AND SAVED")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Documents: {len(kb_data)}")
    print(f"   Vectors: {index.ntotal}")
    print(f"   Dimension: {dimension}")
    print(f"   File: {OUTPUT_PKL_FILE} ({file_size_mb:.2f} MB)")
    print(f"   Format: v2 (compatible)")
    print(f"   Device: {DEVICE}")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Encoding speed: {len(documents)/elapsed:.1f} docs/sec")
    print(f"\n🚀 Next steps:")
    print(f"   1. Upload {OUTPUT_PKL_FILE} to HuggingFace Hub")
    print(f"   2. Restart your backend")
    print(f"   3. Test - should work now!")
    print("=" * 80 + "\n")
    
    return True

if __name__ == "__main__":
    success = build_faiss_index()
    
    if not success:
        print("\n❌ FAILED TO BUILD FAISS INDEX")
        print("\nPlease check:")
        print("1. data/final_knowledge_base.jsonl exists")
        print("2. app/models/best_retriever_model.pth exists")
        print("3. You have enough RAM/VRAM")
        exit(1)


































# """
# Build FAISS Index from Scratch - COMPATIBLE VERSION
# Creates faiss_index.pkl with proper serialization for version compatibility

# Run this ONCE before starting the backend:
#     python build_faiss_index.py

# Author: Banking RAG Chatbot
# Date: November 2025
# """

# # Suppress warnings
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import warnings
# warnings.filterwarnings('ignore')

# import pickle
# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import faiss
# import numpy as np
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModel
# from typing import List

# # ============================================================================
# # CONFIGURATION - UPDATE THESE PATHS!
# # ============================================================================

# # Where is your knowledge base JSONL file?
# KB_JSONL_FILE = "data/final_knowledge_base.jsonl"

# # Where is your trained retriever model?
# RETRIEVER_MODEL_PATH = "app/models/best_retriever_model.pth"

# # Where to save the output FAISS pickle?
# OUTPUT_PKL_FILE = "app/models/faiss_index.pkl"

# # Device (auto-detect GPU/CPU)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Batch size for encoding (reduce if you get OOM errors)
# BATCH_SIZE = 32

# # ============================================================================
# # CUSTOM SENTENCE TRANSFORMER (Same as retriever.py)
# # ============================================================================

# class CustomSentenceTransformer(nn.Module):
#     """
#     Custom SentenceTransformer - exact copy from retriever.py
#     Uses e5-base-v2 with mean pooling and L2 normalization
#     """
    
#     def __init__(self, model_name: str = "intfloat/e5-base-v2"):
#         super().__init__()
#         print(f"   Loading base model: {model_name}...")
#         self.encoder = AutoModel.from_pretrained(model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.config = self.encoder.config
#         print(f"   ✅ Base model loaded")
    
#     def forward(self, input_ids, attention_mask):
#         """Forward pass through BERT encoder"""
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
#         # Mean pooling
#         token_embeddings = outputs.last_hidden_state
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#             input_mask_expanded.sum(1), min=1e-9
#         )
        
#         # L2 normalize
#         embeddings = F.normalize(embeddings, p=2, dim=1)
#         return embeddings
    
#     def encode(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
#         """Encode sentences - same as training"""
#         self.eval()
#         if isinstance(sentences, str):
#             sentences = [sentences]
        
#         # Add 'query: ' prefix for e5-base-v2
#         processed_sentences = [f"query: {s.strip()}" for s in sentences]
        
#         all_embeddings = []
#         with torch.no_grad():
#             for i in range(0, len(processed_sentences), batch_size):
#                 batch_sentences = processed_sentences[i:i + batch_size]
                
#                 # Tokenize
#                 tokens = self.tokenizer(
#                     batch_sentences,
#                     truncation=True,
#                     padding=True,
#                     max_length=128,
#                     return_tensors='pt'
#                 ).to(self.encoder.device)
                
#                 # Get embeddings
#                 embeddings = self.forward(tokens['input_ids'], tokens['attention_mask'])
#                 all_embeddings.append(embeddings.cpu().numpy())
        
#         return np.vstack(all_embeddings)

# # ============================================================================
# # RETRIEVER MODEL (Wrapper)
# # ============================================================================

# class RetrieverModel:
#     """Wrapper for trained retriever model"""
    
#     def __init__(self, model_path: str, device: str = "cpu"):
#         print(f"\n🤖 Loading retriever model...")
#         print(f"   Device: {device}")
#         self.device = device
#         self.model = CustomSentenceTransformer("intfloat/e5-base-v2").to(device)
        
#         # Load trained weights
#         print(f"   Loading weights from: {model_path}")
#         try:
#             state_dict = torch.load(model_path, map_location=device)
#             self.model.load_state_dict(state_dict)
#             print(f"   ✅ Trained weights loaded")
#         except Exception as e:
#             print(f"   ⚠️ Warning: Could not load trained weights: {e}")
#             print(f"   Using base e5-base-v2 model instead")
        
#         self.model.eval()
    
#     def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
#         """Encode documents"""
#         return self.model.encode(documents, batch_size=batch_size)

# # ============================================================================
# # MAIN: BUILD FAISS INDEX
# # ============================================================================

# def build_faiss_index():
#     """Main function to build FAISS index from scratch"""
    
#     print("=" * 80)
#     print("🏗️  BUILDING FAISS INDEX FROM SCRATCH")
#     print("=" * 80)
    
#     # ========================================================================
#     # STEP 1: LOAD KNOWLEDGE BASE
#     # ========================================================================
#     print(f"\n📖 STEP 1: Loading knowledge base...")
#     print(f"   File: {KB_JSONL_FILE}")
    
#     if not os.path.exists(KB_JSONL_FILE):
#         print(f"   ❌ ERROR: File not found!")
#         print(f"   Please copy your knowledge base to: {KB_JSONL_FILE}")
#         return False
    
#     kb_data = []
#     with open(KB_JSONL_FILE, 'r', encoding='utf-8') as f:
#         for line_num, line in enumerate(f, 1):
#             try:
#                 kb_data.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 print(f"   ⚠️ Warning: Skipping invalid JSON on line {line_num}: {e}")
    
#     print(f"   ✅ Loaded {len(kb_data)} documents")
    
#     if len(kb_data) == 0:
#         print(f"   ❌ ERROR: Knowledge base is empty!")
#         return False
    
#     # ========================================================================
#     # STEP 2: PREPARE DOCUMENTS FOR ENCODING
#     # ========================================================================
#     print(f"\n📝 STEP 2: Preparing documents for encoding...")
    
#     documents = []
#     for i, item in enumerate(kb_data):
#         # Combine instruction + response for embedding (same as training)
#         instruction = item.get('instruction', '')
#         response = item.get('response', '')
        
#         # Create combined text
#         if instruction and response:
#             text = f"{instruction} {response}"
#         elif instruction:
#             text = instruction
#         elif response:
#             text = response
#         else:
#             print(f"   ⚠️ Warning: Document {i} has no content, using placeholder")
#             text = "empty document"
        
#         documents.append(text)
    
#     print(f"   ✅ Prepared {len(documents)} documents for encoding")
#     print(f"   Average length: {sum(len(d) for d in documents) / len(documents):.1f} chars")
    
#     # ========================================================================
#     # STEP 3: LOAD RETRIEVER AND ENCODE DOCUMENTS
#     # ========================================================================
#     print(f"\n🔮 STEP 3: Encoding documents with trained retriever...")
    
#     if not os.path.exists(RETRIEVER_MODEL_PATH):
#         print(f"   ❌ ERROR: Retriever model not found!")
#         print(f"   Please copy your trained model to: {RETRIEVER_MODEL_PATH}")
#         return False
    
#     # Load retriever
#     retriever = RetrieverModel(RETRIEVER_MODEL_PATH, device=DEVICE)
    
#     # Encode all documents
#     print(f"   Encoding {len(documents)} documents...")
#     print(f"   Batch size: {BATCH_SIZE}")
#     print(f"   This may take a few minutes... ☕")
    
#     try:
#         embeddings = retriever.encode_documents(documents, batch_size=BATCH_SIZE)
#         print(f"   ✅ Encoded {embeddings.shape[0]} documents")
#         print(f"   Embedding dimension: {embeddings.shape[1]}")
#     except Exception as e:
#         print(f"   ❌ ERROR during encoding: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
    
#     # ========================================================================
#     # STEP 4: BUILD FAISS INDEX WITH PROPER SERIALIZATION
#     # ========================================================================
#     print(f"\n🔍 STEP 4: Building FAISS index...")
    
#     dimension = embeddings.shape[1]
#     print(f"   Dimension: {dimension}")
    
#     # Create FAISS index (Inner Product = Cosine similarity after normalization)
#     print(f"   Creating IndexFlatIP...")
#     index = faiss.IndexFlatIP(dimension)
    
#     # Normalize embeddings for cosine similarity
#     print(f"   Normalizing embeddings...")
#     faiss.normalize_L2(embeddings)
    
#     # Add to index
#     print(f"   Adding {embeddings.shape[0]} vectors to FAISS index...")
#     index.add(embeddings.astype('float32'))
    
#     print(f"   ✅ FAISS index built successfully")
#     print(f"   Total vectors: {index.ntotal}")
    
#     # ========================================================================
#     # STEP 5: SAVE WITH PROPER FAISS SERIALIZATION (VERSION COMPATIBLE!)
#     # ========================================================================
#     print(f"\n💾 STEP 5: Saving with FAISS serialization (version-compatible)...")
    
#     # Create models directory if it doesn't exist
#     os.makedirs(os.path.dirname(OUTPUT_PKL_FILE), exist_ok=True)
    
#     # ✅ PROPER WAY: Serialize FAISS index to bytes first
#     print(f"   Serializing FAISS index to bytes...")
#     try:
#         # Write FAISS index to bytes (works across FAISS versions!)
#         index_bytes = faiss.serialize_index(index)
        
#         # Now pickle the bytes + kb_data
#         print(f"   Pickling (index_bytes, kb_data) tuple...")
#         with open(OUTPUT_PKL_FILE, 'wb') as f:
#             pickle.dump((index_bytes, kb_data), f, protocol=pickle.HIGHEST_PROTOCOL)
        
#         file_size_mb = Path(OUTPUT_PKL_FILE).stat().st_size / (1024 * 1024)
#         print(f"   ✅ Saved: {OUTPUT_PKL_FILE}")
#         print(f"   File size: {file_size_mb:.2f} MB")
#     except Exception as e:
#         print(f"   ❌ ERROR saving pickle: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
    
#     # ========================================================================
#     # STEP 6: VERIFY SAVED FILE
#     # ========================================================================
#     print(f"\n✅ STEP 6: Verifying saved file...")
    
#     try:
#         with open(OUTPUT_PKL_FILE, 'rb') as f:
#             loaded_index_bytes, loaded_kb = pickle.load(f)
        
#         # Deserialize FAISS index from bytes
#         loaded_index = faiss.deserialize_index(loaded_index_bytes)
        
#         print(f"   ✅ Verification successful")
#         print(f"   Index vectors: {loaded_index.ntotal}")
#         print(f"   KB documents: {len(loaded_kb)}")
        
#         if loaded_index.ntotal != len(loaded_kb):
#             print(f"   ⚠️ WARNING: Size mismatch detected!")
#     except Exception as e:
#         print(f"   ❌ ERROR verifying file: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
    
#     # ========================================================================
#     # SUCCESS!
#     # ========================================================================
#     print("\n" + "=" * 80)
#     print("🎉 SUCCESS! FAISS INDEX BUILT AND SAVED")
#     print("=" * 80)
#     print(f"\n📊 Summary:")
#     print(f"   Documents: {len(kb_data)}")
#     print(f"   Vectors: {index.ntotal}")
#     print(f"   Dimension: {dimension}")
#     print(f"   File: {OUTPUT_PKL_FILE} ({file_size_mb:.2f} MB)")
#     print(f"\n🚀 Next steps:")
#     print(f"   1. Upload {OUTPUT_PKL_FILE} to HuggingFace Hub")
#     print(f"   2. Restart your backend")
#     print(f"   3. Test retrieval - should work now!")
#     print("=" * 80 + "\n")
    
#     return True

# # ============================================================================
# # RUN SCRIPT
# # ============================================================================

# if __name__ == "__main__":
#     success = build_faiss_index()
    
#     if not success:
#         print("\n" + "=" * 80)
#         print("❌ FAILED TO BUILD FAISS INDEX")
#         print("=" * 80)
#         print("\nPlease check:")
#         print("1. Knowledge base file exists: data/final_knowledge_base.jsonl")
#         print("2. Retriever model exists: app/models/best_retriever_model.pth")
#         print("3. You have enough RAM (embeddings need ~1GB for 10k docs)")
#         print("=" * 80 + "\n")
#         exit(1)