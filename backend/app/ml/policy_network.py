"""
BERT-based Policy Network for FETCH/NO_FETCH decisions
Trained with Reinforcement Learning (Policy Gradient + Entropy Regularization)

This is adapted from your RL.py with:
- PolicyNetwork class (BERT-based)
- State encoding from conversation history
- Action prediction (FETCH vs NO_FETCH)
- Module-level caching (load once on startup)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
from app.config import settings

# ============================================================================
# POLICY NETWORK (From RL.py)
# ============================================================================


class PolicyNetwork(nn.Module):
    """
    BERT-based Policy Network for deciding FETCH vs NO_FETCH actions.

    Architecture:
    - Base: BERT-base-uncased (pre-trained)
    - Input: Current query + conversation history + previous actions
    - Output: 2-class softmax (FETCH=0, NO_FETCH=1)
    - Special tokens: [FETCH], [NO_FETCH] for action encoding (encoded as plain text)

    Training Details:
    - Loss: Policy Gradient + Entropy Regularization
    - Optimizer: AdamW
    - Reward structure:
      * FETCH: +0.5 (always)
      * NO_FETCH + Good: +2.0
      * NO_FETCH + Bad: -0.5
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dropout_rate: float = 0.1,
        use_multilayer: bool = True,
        hidden_size: int = 128,
    ):
        super(PolicyNetwork, self).__init__()

        # Load pre-trained BERT and tokenizer
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ❗ IMPORTANT:
        # Do NOT add extra special tokens or resize embeddings here.
        # The saved checkpoint was trained with the ORIGINAL BERT vocab
        # (vocab_size=30522). Changing vocab size before loading will cause
        # the size mismatch error:
        #   saved=30522, current=30524

        self.use_multilayer = use_multilayer

        # ✅ FLEXIBLE CLASSIFIER ARCHITECTURE (with configurable hidden size)
        if use_multilayer:
            # Multi-layer classifier with specified hidden size (128 or 256)
            self.classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, 2),
            )
        else:
            # Single-layer classifier (fallback)
            self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def _init_action_embeddings(self):
        """
        (Currently unused)

        In an alternative setup, this could initialize random embeddings
        for [FETCH] and [NO_FETCH] tokens if they were added as true
        special tokens. For this checkpoint we DO NOT change the vocab
        size, so we leave this unused to avoid shape mismatches.
        """
        with torch.no_grad():
            fetch_id = self.tokenizer.convert_tokens_to_ids("[FETCH]")
            no_fetch_id = self.tokenizer.convert_tokens_to_ids("[NO_FETCH]")

            embedding_dim = self.bert.config.hidden_size

            self.bert.embeddings.word_embeddings.weight[fetch_id] = (
                torch.randn(embedding_dim) * 0.02
            )
            self.bert.embeddings.word_embeddings.weight[no_fetch_id] = (
                torch.randn(embedding_dim) * 0.02
            )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through BERT + classifier.

        Args:
            input_ids: Tokenized input IDs (shape: [batch_size, seq_len])
            attention_mask: Attention mask (shape: [batch_size, seq_len])

        Returns:
            logits: Raw logits (shape: [batch_size, 2])
            probs: Softmax probabilities (shape: [batch_size, 2])
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        cls_output = self.dropout(cls_output)

        # Classification
        logits = self.classifier(cls_output)

        # Softmax for probabilities
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def encode_state(
        self,
        state: Dict,
        max_length: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode conversation state into BERT input format.

        State structure:
        {
            'previous_queries': [query1, query2, ...],
            'previous_actions': ['FETCH', 'NO_FETCH', ...],
            'current_query': 'user query'
        }

        Encoding format:
        "Previous query 1: {q1} [Action: [FETCH]] Previous query 2: {q2} [Action: [NO_FETCH]] Current query: <query>"

        Args:
            state: State dictionary
            max_length: Maximum sequence length (default from config)

        Returns:
            dict: Tokenized inputs (input_ids, attention_mask)
        """
        if max_length is None:
            max_length = settings.POLICY_MAX_LEN

        # Build state text from conversation history
        state_text = ""

        # Add previous queries and their actions
        prev_queries = state.get("previous_queries", [])
        prev_actions = state.get("previous_actions", [])

        if prev_queries and prev_actions:
            for i, (prev_query, prev_action) in enumerate(
                zip(prev_queries, prev_actions)
            ):
                state_text += (
                    f"Previous query {i+1}: {prev_query} [Action: [{prev_action}]] "
                )

        # Add current query
        current_query = state.get("current_query", "")
        state_text += f"Current query: {current_query}"

        # Tokenize
        encoding = self.tokenizer(
            state_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        return encoding

    def predict_action(
        self,
        state: Dict,
        use_dropout: bool = False,
        num_samples: int = 10,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action probabilities for a given state.

        Args:
            state: Conversation state dictionary
            use_dropout: Whether to use MC Dropout for uncertainty estimation
            num_samples: Number of MC Dropout samples (if use_dropout=True)

        Returns:
            probs: Action probabilities (shape: [1, 2]) - [P(FETCH), P(NO_FETCH)]
            uncertainty: Standard deviation across samples (if use_dropout=True)
        """
        device = next(self.parameters()).device

        if use_dropout:
            # MC Dropout for uncertainty estimation
            self.train()  # Enable dropout during inference
            all_probs = []

            for _ in range(num_samples):
                with torch.no_grad():
                    encoding = self.encode_state(state)
                    input_ids = encoding["input_ids"].to(device)
                    attention_mask = encoding["attention_mask"].to(device)

                    _, probs = self.forward(input_ids, attention_mask)
                    all_probs.append(probs.cpu().numpy())

            # Average probabilities across samples
            avg_probs = np.mean(all_probs, axis=0)

            # Calculate uncertainty (standard deviation)
            uncertainty = np.std(all_probs, axis=0)

            return avg_probs, uncertainty

        else:
            # Standard inference (no uncertainty estimation)
            self.eval()
            with torch.no_grad():
                encoding = self.encode_state(state)
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

                _, probs = self.forward(input_ids, attention_mask)

                return probs.cpu().numpy(), None


# ============================================================================
# MODULE-LEVEL CACHING (Load once on import)
# ============================================================================

# Global variables for caching
POLICY_MODEL: Optional[PolicyNetwork] = None
POLICY_TOKENIZER: Optional[AutoTokenizer] = None


def load_policy_model() -> PolicyNetwork:
    """
    Load trained policy model (called once on startup).
    Downloads from HuggingFace Hub if not present locally.
    Uses module-level caching - model stays in RAM.

    Returns:
        PolicyNetwork: Loaded policy model
    """
    global POLICY_MODEL, POLICY_TOKENIZER

    if POLICY_MODEL is None:
        # Download model from HF Hub if needed (for deployment)
        settings.download_model_if_needed(
            hf_filename="models/policy_query_only.pt",
            local_path=settings.POLICY_MODEL_PATH,
        )

        print(f"Loading policy network from {settings.POLICY_MODEL_PATH}...")
        try:
            # Load checkpoint first to detect architecture
            checkpoint = torch.load(
                settings.POLICY_MODEL_PATH, map_location=settings.DEVICE
            )

            # Unwrap if saved as {"model_state_dict": ...}
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # ✅ AUTO-DETECT ARCHITECTURE from checkpoint keys
            has_multilayer = "classifier.0.weight" in state_dict

            if has_multilayer:
                hidden_size = state_dict["classifier.0.weight"].shape[0]
                print(
                    f"📊 Detected: Multi-layer classifier (hidden_size={hidden_size})"
                )
            else:
                hidden_size = 768  # not really used for single-layer
                print("📊 Detected: Single-layer classifier")

            # Create model instance with correct architecture
            POLICY_MODEL = PolicyNetwork(
                model_name="bert-base-uncased",
                dropout_rate=0.1,
                use_multilayer=has_multilayer,
                hidden_size=hidden_size,
            )

            # Align vocab size / embeddings with checkpoint
            saved_vocab_size = state_dict[
                "bert.embeddings.word_embeddings.weight"
            ].shape[0]
            current_vocab_size = (
                POLICY_MODEL.bert.embeddings.word_embeddings.num_embeddings
            )

            if saved_vocab_size != current_vocab_size:
                print(
                    f"⚠️ Vocab size mismatch: saved={saved_vocab_size}, current={current_vocab_size}"
                )
                print(
                    "✅ Resizing BERT embeddings to match saved checkpoint vocab size..."
                )
                POLICY_MODEL.bert.resize_token_embeddings(saved_vocab_size)

            # Move to device
            POLICY_MODEL = POLICY_MODEL.to(settings.DEVICE)

            # Load weights (shapes now match, so strict=False is just safety)
            POLICY_MODEL.load_state_dict(state_dict, strict=False)

            # Set to evaluation mode
            POLICY_MODEL.eval()

            # Cache tokenizer
            POLICY_TOKENIZER = POLICY_MODEL.tokenizer

            print("✅ Policy network loaded and cached")
            print(
                f"   Model vocab size: {POLICY_MODEL.bert.embeddings.word_embeddings.num_embeddings}"
            )
            print(f"   Tokenizer vocab size: {len(POLICY_MODEL.tokenizer)}")

        except FileNotFoundError:
            print(f"❌ Policy model file not found: {settings.POLICY_MODEL_PATH}")
            print(
                f"⚠️ Make sure models are uploaded to HuggingFace Hub: {settings.HF_MODEL_REPO}"
            )
            raise
        except Exception as e:
            print(f"❌ Failed to load policy model: {e}")
            import traceback

            traceback.print_exc()
            raise

    return POLICY_MODEL


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================


def create_state_from_history(
    current_query: str,
    conversation_history: List[Dict],
    max_history: int = 2,
) -> Dict:
    """
    Create state dictionary from conversation history.
    Extracts last N query-action pairs.

    Args:
        current_query: Current user query
        conversation_history: List of conversation turns
            Each turn: {'role': 'user'/'assistant', 'content': '...', 'metadata': {...}}
        max_history: Maximum number of previous turns to include (default: 2)

    Returns:
        dict: State dictionary for policy network
    """
    state = {
        "current_query": current_query,
        "previous_queries": [],
        "previous_actions": [],
    }

    if not conversation_history:
        return state

    # Extract last N conversation turns (user + assistant pairs)
    relevant_history = conversation_history[-(max_history * 2) :]

    for i, turn in enumerate(relevant_history):
        # User turns
        if turn.get("role") == "user":
            query = turn.get("content", "")
            state["previous_queries"].append(query)

            # Look for corresponding assistant turn
            if i + 1 < len(relevant_history):
                bot_turn = relevant_history[i + 1]
                if bot_turn.get("role") == "assistant":
                    metadata = bot_turn.get("metadata", {})
                    action = metadata.get("policy_action", "FETCH")
                    state["previous_actions"].append(action)

    return state


def predict_policy_action(
    query: str,
    history: List[Dict] = None,
    return_probs: bool = False,
) -> Dict:
    """
    Predict FETCH/NO_FETCH action for a query.

    Args:
        query: User query text
        history: Conversation history (optional)
        return_probs: Whether to return full probability distribution

    Returns:
        dict: Prediction results
        {
            'action': 'FETCH' or 'NO_FETCH',
            'confidence': float (0-1),
            'fetch_prob': float,
            'no_fetch_prob': float,
            'should_retrieve': bool
        }
    """
    # Load model (cached after first call)
    model = load_policy_model()

    # Create state from history
    if history is None:
        history = []

    state = create_state_from_history(query, history)

    # Predict action
    probs, _ = model.predict_action(state, use_dropout=False)

    # Extract probabilities
    fetch_prob = float(probs[0][0])
    no_fetch_prob = float(probs[0][1])

    # Determine action (argmax)
    action_idx = int(np.argmax(probs[0]))
    action = "FETCH" if action_idx == 0 else "NO_FETCH"
    confidence = float(probs[0][action_idx])

    # Check confidence threshold
    should_retrieve = (action == "FETCH") or (
        action == "NO_FETCH" and confidence < settings.CONFIDENCE_THRESHOLD
    )

    result = {
        "action": action,
        "confidence": confidence,
        "should_retrieve": should_retrieve,
        "policy_decision": action,
    }

    if return_probs:
        result["fetch_prob"] = fetch_prob
        result["no_fetch_prob"] = no_fetch_prob

    return result


# ============================================================================
# USAGE EXAMPLE (for reference)
# ============================================================================
"""
# In your service file:
from app.ml.policy_network import predict_policy_action

# Predict action
history = [
    {'role': 'user', 'content': 'What is my balance?'},
    {'role': 'assistant', 'content': '$1000', 'metadata': {'policy_action': 'FETCH'}}
]

result = predict_policy_action(
    query="Thank you!",
    history=history,
    return_probs=True
)

print(result)
# {
#     'action': 'NO_FETCH',
#     'confidence': 0.95,
#     'should_retrieve': False,
#     'fetch_prob': 0.05,
#     'no_fetch_prob': 0.95
# }
"""