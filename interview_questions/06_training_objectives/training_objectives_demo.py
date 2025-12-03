"""
Training Objectives for LLM Interview Preparation
=================================================

This demo covers key training objectives used in modern LLMs:
1. Masked Language Modeling (MLM) - BERT-style
2. Causal Language Modeling (CLM) - GPT-style
3. Next Sentence Prediction (NSP)

Key Interview Questions:
- Q7: How does masked language modeling work?
- Q9: What's the difference between MLM and causal LM?
- Q11: Why do we use different masking strategies?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from typing import List, Tuple, Dict

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create output directory for visualizations
output_dir = Path("/Users/zack/dev/ml-refresher/data/interview_viz")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LLM TRAINING OBJECTIVES DEMO")
print("=" * 80)


# ============================================================================
# PART 1: VOCABULARY AND TOKENIZATION SETUP
# ============================================================================

class SimpleTokenizer:
    """
    Simple tokenizer for demonstration purposes.

    Interview Tip: Real LLMs use subword tokenizers (BPE, WordPiece, SentencePiece)
    but this simplified version helps understand the core concepts.
    """
    def __init__(self):
        # Special tokens used in LLM training
        self.special_tokens = {
            '[PAD]': 0,   # Padding token
            '[UNK]': 1,   # Unknown token
            '[CLS]': 2,   # Classification token (BERT)
            '[SEP]': 3,   # Separator token
            '[MASK]': 4,  # Mask token for MLM
        }

        # Simple vocabulary (in practice, this would be 30k-50k tokens)
        self.vocab = {
            **self.special_tokens,
            'the': 5, 'a': 6, 'is': 7, 'was': 8, 'are': 9,
            'cat': 10, 'dog': 11, 'sat': 12, 'mat': 13, 'on': 14,
            'quick': 15, 'brown': 16, 'fox': 17, 'jumps': 18, 'over': 19,
            'lazy': 20, 'language': 21, 'model': 22, 'learning': 23,
            'machine': 24, 'deep': 25, 'neural': 26, 'network': 27,
            'transformer': 28, 'attention': 29, 'bert': 30, 'gpt': 31,
        }

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = text.lower().split()
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text"""
        return ' '.join([self.id_to_token.get(id, '[UNK]') for id in ids])

tokenizer = SimpleTokenizer()

print("\n" + "=" * 80)
print("TOKENIZER SETUP")
print("=" * 80)
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Special tokens: {list(tokenizer.special_tokens.keys())}")
print(f"\nExample encoding:")
example_text = "the cat sat on the mat"
encoded = tokenizer.encode(example_text)
print(f"Text: '{example_text}'")
print(f"Encoded: {encoded}")
print(f"Decoded: '{tokenizer.decode(encoded)}'")


# ============================================================================
# PART 2: MASKED LANGUAGE MODELING (MLM) - BERT STYLE
# ============================================================================

print("\n" + "=" * 80)
print("MASKED LANGUAGE MODELING (MLM)")
print("=" * 80)
print("""
MLM is the training objective used by BERT and similar bidirectional models.

KEY CONCEPTS:
1. Random tokens are masked in the input
2. Model must predict the original token
3. Uses bidirectional context (can see both left and right)
4. Masking strategy (BERT paper):
   - 80% of time: Replace with [MASK]
   - 10% of time: Replace with random token
   - 10% of time: Keep unchanged

WHY THIS STRATEGY?
- 80% [MASK]: Main training signal
- 10% random: Prevents model from relying on [MASK] token
- 10% unchanged: Encourages model to learn representations for all tokens

INTERVIEW TIP: Be able to explain why we don't mask 100% with [MASK]!
""")


class MLMDataProcessor:
    """
    Processes data for Masked Language Modeling.

    This is a critical component for BERT-style models.
    """
    def __init__(self, tokenizer: SimpleTokenizer, mask_prob: float = 0.15):
        """
        Args:
            tokenizer: Tokenizer instance
            mask_prob: Probability of masking each token (BERT uses 15%)
        """
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.vocab['[MASK]']
        self.pad_token_id = tokenizer.vocab['[PAD]']
        self.special_token_ids = set(tokenizer.special_tokens.values())

    def create_mlm_batch(
        self,
        text: str,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create a batch for MLM training.

        Returns:
            input_ids: Token IDs with masking applied
            labels: Original token IDs (only for masked positions)
            attention_mask: 1 for real tokens, 0 for padding
        """
        # Encode text
        tokens = tokenizer.encode(text)
        original_tokens = tokens.copy()

        # Create labels (-100 is ignored by PyTorch loss functions)
        labels = [-100] * len(tokens)

        # Apply masking
        masked_positions = []
        for i, token_id in enumerate(tokens):
            # Don't mask special tokens
            if token_id in self.special_token_ids:
                continue

            # Randomly decide if this token should be masked
            if random.random() < self.mask_prob:
                masked_positions.append(i)
                labels[i] = token_id  # Store original token for loss calculation

                prob = random.random()
                if prob < 0.8:
                    # 80% of time: replace with [MASK]
                    tokens[i] = self.mask_token_id
                elif prob < 0.9:
                    # 10% of time: replace with random token
                    tokens[i] = random.randint(5, self.tokenizer.vocab_size - 1)
                # else: 10% of time: keep unchanged (do nothing)

        if verbose:
            print(f"\nOriginal text: '{text}'")
            print(f"Original tokens: {original_tokens}")
            print(f"Original decoded: '{tokenizer.decode(original_tokens)}'")
            print(f"\nMasked positions: {masked_positions} ({len(masked_positions)}/{len(tokens)} = {len(masked_positions)/len(tokens)*100:.1f}%)")
            print(f"Input tokens: {tokens}")
            print(f"Input decoded: '{tokenizer.decode(tokens)}'")
            print(f"\nLabels (only masked positions have values != -100):")
            for i, (label, orig) in enumerate(zip(labels, original_tokens)):
                if label != -100:
                    print(f"  Position {i}: Predict '{tokenizer.id_to_token[label]}' (ID: {label})")

        # Convert to tensors
        input_ids = torch.tensor([tokens])
        labels_tensor = torch.tensor([labels])
        attention_mask = torch.ones_like(input_ids)

        return input_ids, labels_tensor, attention_mask


# Demonstrate MLM with examples
mlm_processor = MLMDataProcessor(tokenizer, mask_prob=0.15)

print("\n" + "-" * 80)
print("EXAMPLE 1: Simple sentence")
print("-" * 80)
example1 = "the quick brown fox jumps over the lazy dog"
input_ids1, labels1, mask1 = mlm_processor.create_mlm_batch(example1)

print("\n" + "-" * 80)
print("EXAMPLE 2: Technical sentence")
print("-" * 80)
example2 = "the transformer model uses attention mechanism"
input_ids2, labels2, mask2 = mlm_processor.create_mlm_batch(example2)


class SimpleBERTModel(nn.Module):
    """
    Simplified BERT model for demonstration.

    In a real BERT model:
    - 12-24 transformer layers
    - 768-1024 hidden dimensions
    - Multi-head attention
    - Layer normalization, residual connections, etc.
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 64):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        # MLM head: projects hidden states back to vocabulary
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        Forward pass for MLM.

        Returns logits for each position in vocabulary space.
        """
        # Embed tokens
        embeddings = self.embeddings(input_ids)

        # Apply transformer (bidirectional - can see entire context)
        hidden_states = self.transformer(embeddings)

        # Project to vocabulary for prediction
        logits = self.mlm_head(hidden_states)

        return logits


# Create model and compute loss
bert_model = SimpleBERTModel(tokenizer.vocab_size)

print("\n" + "-" * 80)
print("MLM LOSS COMPUTATION")
print("-" * 80)

# Forward pass
logits = bert_model(input_ids1)
print(f"Input shape: {input_ids1.shape}")
print(f"Logits shape: {logits.shape} (batch_size, seq_len, vocab_size)")

# Compute loss (only on masked positions due to labels=-100)
loss_fn = nn.CrossEntropyLoss()  # Ignores -100 labels by default
loss = loss_fn(logits.view(-1, tokenizer.vocab_size), labels1.view(-1))
print(f"\nMLM Loss: {loss.item():.4f}")

print("""
LOSS COMPUTATION DETAILS:
- Loss is computed ONLY on masked positions
- Non-masked positions have label = -100 (ignored)
- This is key: model learns to predict masked tokens using bidirectional context
- At inference: no masking, model outputs representations for downstream tasks
""")


# ============================================================================
# PART 3: CAUSAL LANGUAGE MODELING (CLM) - GPT STYLE
# ============================================================================

print("\n" + "=" * 80)
print("CAUSAL LANGUAGE MODELING (CLM)")
print("=" * 80)
print("""
CLM is the training objective used by GPT and similar autoregressive models.

KEY CONCEPTS:
1. Predict the next token given previous tokens
2. Uses causal (autoregressive) attention - can only see left context
3. Each position predicts the next token
4. No masking needed in input (masking in attention mechanism)

DIFFERENCES FROM MLM:
- MLM: Bidirectional, mask some tokens, predict masked ones
- CLM: Unidirectional, no masking in input, predict next token at each position

INTERVIEW TIP: CLM is simpler but very effective! Powers GPT-3, GPT-4, etc.
""")


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create causal attention mask.

    Position i can only attend to positions <= i.
    This prevents the model from "cheating" by looking at future tokens.

    Returns:
        mask: Shape (seq_len, seq_len), True where attention is allowed
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask


# Visualize causal mask
seq_len = 8
causal_mask = create_causal_mask(seq_len)

print("\n" + "-" * 80)
print("CAUSAL ATTENTION MASK VISUALIZATION")
print("-" * 80)
print("""
In causal attention:
- Each position can attend to itself and previous positions
- Cannot attend to future positions (upper triangle is masked)
- This enforces left-to-right, autoregressive generation
""")

plt.figure(figsize=(10, 8))
sns.heatmap(
    causal_mask.numpy(),
    cmap='RdYlGn',
    cbar_kws={'label': 'Can Attend'},
    square=True,
    linewidths=0.5,
    annot=True,
    fmt='d',
    xticklabels=[f'Pos {i}' for i in range(seq_len)],
    yticklabels=[f'Pos {i}' for i in range(seq_len)]
)
plt.title('Causal Attention Mask\n(1 = can attend, 0 = masked)', fontsize=14, fontweight='bold')
plt.xlabel('Key Position (attending TO)', fontsize=12)
plt.ylabel('Query Position (attending FROM)', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / 'causal_attention_mask.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'causal_attention_mask.png'}")
plt.close()


class CLMDataProcessor:
    """
    Processes data for Causal Language Modeling.

    Much simpler than MLM - no masking needed in input!
    """
    def __init__(self, tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer

    def create_clm_batch(
        self,
        text: str,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a batch for CLM training.

        Input:  [token_1, token_2, token_3, token_4]
        Labels: [token_2, token_3, token_4, token_5]

        Each position predicts the next token.
        """
        # Encode text
        tokens = tokenizer.encode(text)

        # Input: all tokens except last
        input_ids = tokens[:-1]

        # Labels: all tokens except first (shifted by 1)
        labels = tokens[1:]

        if verbose:
            print(f"\nOriginal text: '{text}'")
            print(f"All tokens: {tokens}")
            print(f"Decoded: '{tokenizer.decode(tokens)}'")
            print(f"\nCLM Training Setup:")
            print(f"Input IDs:  {input_ids}")
            print(f"Labels:     {labels}")
            print(f"\nPrediction targets at each position:")
            for i, (inp, label) in enumerate(zip(input_ids, labels)):
                inp_word = tokenizer.id_to_token[inp]
                label_word = tokenizer.id_to_token[label]
                print(f"  Position {i}: Given '{inp_word}' (and all previous), predict '{label_word}'")

        return torch.tensor([input_ids]), torch.tensor([labels])


# Demonstrate CLM
clm_processor = CLMDataProcessor(tokenizer)

print("\n" + "-" * 80)
print("CLM EXAMPLE 1")
print("-" * 80)
input_ids_clm1, labels_clm1 = clm_processor.create_clm_batch(example1)

print("\n" + "-" * 80)
print("CLM EXAMPLE 2")
print("-" * 80)
input_ids_clm2, labels_clm2 = clm_processor.create_clm_batch(example2)


class SimpleGPTModel(nn.Module):
    """
    Simplified GPT model for demonstration.

    Key difference from BERT: Uses causal (masked) attention.
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 64):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        # Causal transformer decoder layer
        self.transformer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )

        # LM head: projects hidden states to vocabulary
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        Forward pass for CLM.

        Uses causal attention mask to prevent seeing future tokens.
        """
        seq_len = input_ids.size(1)

        # Embed tokens
        embeddings = self.embeddings(input_ids)

        # Create causal mask
        causal_mask = ~create_causal_mask(seq_len)  # PyTorch uses True for masked positions

        # Apply transformer with causal attention
        hidden_states = self.transformer(
            embeddings,
            embeddings,
            tgt_mask=causal_mask
        )

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return logits


# Create GPT model and compute loss
gpt_model = SimpleGPTModel(tokenizer.vocab_size)

print("\n" + "-" * 80)
print("CLM LOSS COMPUTATION")
print("-" * 80)

# Forward pass
logits_clm = gpt_model(input_ids_clm1)
print(f"Input shape: {input_ids_clm1.shape}")
print(f"Logits shape: {logits_clm.shape} (batch_size, seq_len, vocab_size)")
print(f"Labels shape: {labels_clm1.shape}")

# Compute loss (on ALL positions - each predicts next token)
loss_clm = loss_fn(logits_clm.view(-1, tokenizer.vocab_size), labels_clm1.view(-1))
print(f"\nCLM Loss: {loss_clm.item():.4f}")

print("""
LOSS COMPUTATION DETAILS:
- Loss is computed on ALL positions (unlike MLM)
- Each position predicts the next token
- Model learns from every token in the sequence
- At inference: generate token by token, feeding output back as input
""")


# ============================================================================
# PART 4: SIDE-BY-SIDE COMPARISON - MLM VS CLM
# ============================================================================

print("\n" + "=" * 80)
print("MLM VS CLM: SIDE-BY-SIDE COMPARISON")
print("=" * 80)

comparison_text = "the cat sat on the mat"
comparison_tokens = tokenizer.encode(comparison_text)

# MLM processing
mlm_input_ids, mlm_labels, _ = mlm_processor.create_mlm_batch(comparison_text, verbose=False)

# CLM processing
clm_input_ids, clm_labels = clm_processor.create_clm_batch(comparison_text, verbose=False)

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# MLM visualization
ax1 = axes[0]
tokens_display = [tokenizer.id_to_token[tid] for tid in comparison_tokens]
mlm_input_display = [tokenizer.id_to_token[tid] for tid in mlm_input_ids[0].tolist()]
mlm_mask = (mlm_labels[0] != -100).numpy()

colors_mlm = ['lightcoral' if masked else 'lightgreen' for masked in mlm_mask]
y_pos = np.arange(len(tokens_display))

ax1.barh(y_pos, [1] * len(tokens_display), color=colors_mlm, alpha=0.6)
for i, (orig, masked, is_masked) in enumerate(zip(tokens_display, mlm_input_display, mlm_mask)):
    label = f"{orig}\n→ {masked}" if is_masked else orig
    ax1.text(0.5, i, label, ha='center', va='center', fontsize=11, fontweight='bold')

ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"Pos {i}" for i in range(len(tokens_display))])
ax1.set_xlim(0, 1)
ax1.set_xticks([])
ax1.set_title('Masked Language Modeling (MLM) - BERT Style\nRed = Masked positions (predict these), Green = Context',
              fontsize=13, fontweight='bold', pad=20)
ax1.invert_yaxis()

# CLM visualization
ax2 = axes[1]
clm_input_display = [tokenizer.id_to_token[tid] for tid in clm_input_ids[0].tolist()]
clm_label_display = [tokenizer.id_to_token[tid] for tid in clm_labels[0].tolist()]

colors_clm = ['lightblue'] * len(clm_input_display)
y_pos_clm = np.arange(len(clm_input_display))

ax2.barh(y_pos_clm, [1] * len(clm_input_display), color=colors_clm, alpha=0.6)
for i, (inp, target) in enumerate(zip(clm_input_display, clm_label_display)):
    label = f"{inp}\n→ {target}"
    ax2.text(0.5, i, label, ha='center', va='center', fontsize=11, fontweight='bold')

ax2.set_yticks(y_pos_clm)
ax2.set_yticklabels([f"Pos {i}" for i in range(len(clm_input_display))])
ax2.set_xlim(0, 1)
ax2.set_xticks([])
ax2.set_title('Causal Language Modeling (CLM) - GPT Style\nBlue = Predict next token at each position',
              fontsize=13, fontweight='bold', pad=20)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'mlm_vs_clm_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comparison: {output_dir / 'mlm_vs_clm_comparison.png'}")
plt.close()

# Print comparison table
print("\n" + "-" * 80)
print("COMPARISON TABLE")
print("-" * 80)
comparison_data = {
    'Aspect': [
        'Direction',
        'Masking',
        'Loss Computed On',
        'Training Efficiency',
        'Best For',
        'Examples',
        'Attention Pattern',
        'Generation'
    ],
    'MLM (BERT)': [
        'Bidirectional',
        '15% of tokens masked',
        'Only masked positions',
        'Lower (only ~15% tokens)',
        'Understanding, classification',
        'BERT, RoBERTa, ALBERT',
        'Full attention',
        'Not naturally generative'
    ],
    'CLM (GPT)': [
        'Unidirectional (left-to-right)',
        'No masking in input',
        'All positions (predict next)',
        'Higher (uses all tokens)',
        'Generation, completion',
        'GPT-2, GPT-3, GPT-4',
        'Causal (triangular) mask',
        'Natural for generation'
    ]
}

for i in range(len(comparison_data['Aspect'])):
    print(f"\n{comparison_data['Aspect'][i]}:")
    print(f"  MLM: {comparison_data['MLM (BERT)'][i]}")
    print(f"  CLM: {comparison_data['CLM (GPT)'][i]}")


# ============================================================================
# PART 5: NEXT SENTENCE PREDICTION (NSP)
# ============================================================================

print("\n" + "=" * 80)
print("NEXT SENTENCE PREDICTION (NSP)")
print("=" * 80)
print("""
NSP is an additional training objective used in original BERT.

KEY CONCEPTS:
1. Given two sentences A and B, predict if B follows A in original text
2. Binary classification: IsNext (1) or NotNext (0)
3. Uses [CLS] token representation for classification
4. 50% of time B actually follows A, 50% it's a random sentence

PURPOSE:
- Learn sentence-level relationships
- Useful for tasks like QA where understanding sentence pairs matters

CONTROVERSY:
- Later work (RoBERTa) showed NSP might not be necessary
- Removing it didn't hurt and sometimes helped performance
- Modern models often skip NSP

INTERVIEW TIP: Know that NSP exists but is less important than MLM/CLM!
""")


class NSPDataProcessor:
    """
    Processes data for Next Sentence Prediction.
    """
    def __init__(self, tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer
        self.cls_token_id = tokenizer.vocab['[CLS]']
        self.sep_token_id = tokenizer.vocab['[SEP]']

    def create_nsp_pair(
        self,
        sentence_a: str,
        sentence_b: str,
        is_next: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create NSP training example.

        Format: [CLS] sentence_a [SEP] sentence_b [SEP]
        Label: 1 if is_next, 0 otherwise
        """
        # Encode sentences
        tokens_a = tokenizer.encode(sentence_a)
        tokens_b = tokenizer.encode(sentence_b)

        # Combine with special tokens
        tokens = [self.cls_token_id] + tokens_a + [self.sep_token_id] + tokens_b + [self.sep_token_id]

        label = 1 if is_next else 0

        return torch.tensor([tokens]), torch.tensor([label])


# Create NSP examples
nsp_processor = NSPDataProcessor(tokenizer)

print("\n" + "-" * 80)
print("NSP EXAMPLE 1: Positive pair (IsNext)")
print("-" * 80)
sent_a1 = "the cat sat on the mat"
sent_b1 = "the dog was lazy"  # Could be next sentence
nsp_input1, nsp_label1 = nsp_processor.create_nsp_pair(sent_a1, sent_b1, is_next=True)
print(f"Sentence A: '{sent_a1}'")
print(f"Sentence B: '{sent_b1}'")
print(f"Combined tokens: {nsp_input1[0].tolist()}")
print(f"Decoded: '{tokenizer.decode(nsp_input1[0].tolist())}'")
print(f"Label: {nsp_label1.item()} (IsNext)")

print("\n" + "-" * 80)
print("NSP EXAMPLE 2: Negative pair (NotNext)")
print("-" * 80)
sent_a2 = "the cat sat on the mat"
sent_b2 = "transformer model uses attention"  # Random, unrelated
nsp_input2, nsp_label2 = nsp_processor.create_nsp_pair(sent_a2, sent_b2, is_next=False)
print(f"Sentence A: '{sent_a2}'")
print(f"Sentence B: '{sent_b2}'")
print(f"Combined tokens: {nsp_input2[0].tolist()}")
print(f"Decoded: '{tokenizer.decode(nsp_input2[0].tolist())}'")
print(f"Label: {nsp_label2.item()} (NotNext)")


class BERTWithNSP(nn.Module):
    """
    BERT model with NSP head.

    Uses [CLS] token representation for sentence pair classification.
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 64):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )

        # MLM head
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)

        # NSP head (binary classification)
        self.nsp_head = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        hidden_states = self.transformer(embeddings)

        # MLM predictions for all positions
        mlm_logits = self.mlm_head(hidden_states)

        # NSP prediction from [CLS] token (position 0)
        cls_hidden = hidden_states[:, 0, :]
        nsp_logits = self.nsp_head(cls_hidden)

        return mlm_logits, nsp_logits


# Demonstrate NSP
bert_nsp_model = BERTWithNSP(tokenizer.vocab_size)

print("\n" + "-" * 80)
print("NSP LOSS COMPUTATION")
print("-" * 80)

mlm_logits, nsp_logits = bert_nsp_model(nsp_input1)
print(f"Input shape: {nsp_input1.shape}")
print(f"MLM logits shape: {mlm_logits.shape}")
print(f"NSP logits shape: {nsp_logits.shape} (batch_size, 2)")

# NSP loss
nsp_loss = loss_fn(nsp_logits, nsp_label1)
print(f"\nNSP Loss: {nsp_loss.item():.4f}")
print(f"NSP Prediction: {torch.argmax(nsp_logits, dim=1).item()}")
print(f"NSP Label: {nsp_label1.item()}")

print("""
COMBINED TRAINING:
- Original BERT trained with BOTH MLM and NSP simultaneously
- Total loss = MLM loss + NSP loss
- Multi-task learning helps create better representations
""")


# ============================================================================
# PART 6: MASKING PATTERNS VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("MASKING PATTERNS VISUALIZATION")
print("=" * 80)

# Create longer example for visualization
long_text = "the quick brown fox jumps over the lazy dog the cat sat on the mat"
long_tokens = tokenizer.encode(long_text)

# Apply MLM masking multiple times to show randomness
num_examples = 5
masking_results = []

for i in range(num_examples):
    input_ids, labels, _ = mlm_processor.create_mlm_batch(long_text, verbose=False)
    masked = (labels[0] != -100).numpy()
    masking_results.append(masked)

# Visualize masking patterns
fig, axes = plt.subplots(num_examples, 1, figsize=(16, 10))

for i, (ax, masked) in enumerate(zip(axes, masking_results)):
    # Create color map: red for masked, green for unmasked
    colors = ['red' if m else 'green' for m in masked]

    # Bar plot
    ax.bar(range(len(masked)), [1] * len(masked), color=colors, alpha=0.6, edgecolor='black')

    # Add token labels
    for j, token_id in enumerate(long_tokens):
        token = tokenizer.id_to_token[token_id]
        ax.text(j, 0.5, token, ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_xlim(-0.5, len(masked) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(range(len(masked)))
    ax.set_xticklabels([f"{i}" for i in range(len(masked))], fontsize=8)
    ax.set_title(f'Masking Pattern {i+1} ({masked.sum()}/{len(masked)} tokens masked = {masked.sum()/len(masked)*100:.1f}%)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Token Position' if i == num_examples - 1 else '', fontsize=10)

plt.suptitle('MLM Masking Patterns (15% probability per token)\nRed = Masked, Green = Unmasked',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'masking_patterns.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved masking patterns: {output_dir / 'masking_patterns.png'}")
plt.close()


# Visualize the 80-10-10 rule
print("\n" + "-" * 80)
print("80-10-10 MASKING STRATEGY BREAKDOWN")
print("-" * 80)

# Simulate many maskings to show distribution
mask_types = {'[MASK]': 0, 'random': 0, 'unchanged': 0}
num_simulations = 1000

for _ in range(num_simulations):
    prob = random.random()
    if prob < 0.8:
        mask_types['[MASK]'] += 1
    elif prob < 0.9:
        mask_types['random'] += 1
    else:
        mask_types['unchanged'] += 1

# Visualize distribution
fig, ax = plt.subplots(figsize=(10, 6))
colors_strategy = ['#e74c3c', '#3498db', '#2ecc71']
bars = ax.bar(mask_types.keys(), mask_types.values(), color=colors_strategy, alpha=0.7, edgecolor='black', linewidth=2)

for bar, (key, value) in zip(bars, mask_types.items()):
    height = bar.get_height()
    percentage = (value / num_simulations) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Count (out of 1000)', fontsize=12, fontweight='bold')
ax.set_xlabel('Masking Strategy', fontsize=12, fontweight='bold')
ax.set_title('BERT Masking Strategy Distribution\n80% [MASK], 10% Random Token, 10% Unchanged',
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, max(mask_types.values()) * 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'masking_strategy_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved masking strategy: {output_dir / 'masking_strategy_distribution.png'}")
plt.close()


# ============================================================================
# PART 7: INTERVIEW PREP SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("INTERVIEW PREPARATION SUMMARY")
print("=" * 80)

interview_qa = {
    "Q7: How does masked language modeling work?": """
    Answer:
    - Randomly mask 15% of tokens in the input
    - Model must predict original token using bidirectional context
    - Masking strategy: 80% [MASK], 10% random token, 10% unchanged
    - Loss computed only on masked positions
    - Used by BERT, RoBERTa, ALBERT
    - Good for understanding tasks (classification, NER, etc.)
    """,

    "Q9: What's the difference between MLM and Causal LM?": """
    Answer:
    MLM (BERT-style):
    - Bidirectional: sees full context
    - Masks random tokens, predicts them
    - Loss on ~15% of tokens only
    - Better for understanding tasks

    Causal LM (GPT-style):
    - Unidirectional: left-to-right only
    - Predicts next token at each position
    - Loss on all tokens
    - Better for generation tasks
    - Uses causal attention mask

    Key insight: MLM sees future tokens (bidirectional) while CLM doesn't (causal)
    """,

    "Q11: Why do we use different masking strategies (80-10-10)?": """
    Answer:
    - 80% [MASK]: Main training signal, model learns to predict masked tokens
    - 10% random token: Prevents overfitting to [MASK] token, model can't rely on it
    - 10% unchanged: Forces model to learn representations for all tokens, not just masked ones

    Without this strategy:
    - Model might only learn to predict when it sees [MASK]
    - At inference, there's no [MASK] token, causing train/test mismatch
    - This strategy makes training more robust

    Trade-off: More complex but better generalization
    """,
}

for question, answer in interview_qa.items():
    print(f"\n{question}")
    print(answer)

print("\n" + "=" * 80)
print("KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 80)
print("""
1. MLM (BERT):
   - Bidirectional, mask random tokens
   - Good for understanding and classification
   - 80-10-10 masking strategy is crucial
   - Loss only on masked positions

2. CLM (GPT):
   - Unidirectional, predict next token
   - Good for generation
   - Simpler but very effective
   - Loss on all positions
   - Uses causal attention mask

3. NSP (BERT):
   - Additional objective for sentence pairs
   - Binary classification: IsNext or NotNext
   - Later shown to be less important
   - Many modern models skip it

4. Attention Patterns:
   - MLM: Full attention matrix (bidirectional)
   - CLM: Triangular attention matrix (causal)
   - This is the fundamental architectural difference

5. Interview Red Flags to Avoid:
   ✗ "BERT uses masking in attention" (No! It's bidirectional)
   ✗ "GPT can see future tokens" (No! It's causal)
   ✗ "All masked tokens use [MASK]" (No! 80-10-10 rule)
   ✗ "NSP is essential" (No! It's optional)

6. Good Interview Answers:
   ✓ Explain bidirectional vs causal attention
   ✓ Describe the 80-10-10 masking strategy
   ✓ Know when to use MLM vs CLM
   ✓ Understand the loss computation differences
""")

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print(f"\nAll visualizations saved to: {output_dir}")
print("\nGenerated files:")
print("  1. causal_attention_mask.png - Shows causal masking pattern")
print("  2. mlm_vs_clm_comparison.png - Side-by-side comparison")
print("  3. masking_patterns.png - Multiple masking examples")
print("  4. masking_strategy_distribution.png - 80-10-10 breakdown")
print("\n" + "=" * 80)
