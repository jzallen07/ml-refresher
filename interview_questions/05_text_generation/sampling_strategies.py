"""
Text Generation Sampling Strategies Demo
=========================================

This module demonstrates various text generation strategies used in Large Language Models.
Perfect for interview preparation covering questions about:
- Q5: Greedy decoding vs beam search
- Q6: Temperature in LLM sampling
- Q12: Top-k and top-p sampling

Author: ML Interview Prep
Date: 2024
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os


# ==============================================================================
# SETUP: Mock Vocabulary and Probability Distributions
# ==============================================================================

class MockVocabulary:
    """
    A mock vocabulary for demonstration purposes.
    In a real LLM, this would be the tokenizer's vocabulary.
    """

    def __init__(self):
        self.vocab = [
            '<pad>', '<sos>', '<eos>', 'the', 'a', 'cat', 'dog', 'sat',
            'on', 'mat', 'ran', 'jumped', 'quickly', 'slowly', 'and',
            'or', 'but', 'very', 'quite', 'really', 'happy', 'sad',
            'big', 'small', 'red', 'blue', 'green', 'yellow', 'house',
            'tree', 'car', 'bike', 'book', 'pen', 'table', 'chair'
        ]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def decode(self, idx: int) -> str:
        return self.idx_to_word.get(idx, '<unk>')

    def encode(self, word: str) -> int:
        return self.word_to_idx.get(word, 0)


def create_mock_logits(vocab_size: int, seed: Optional[int] = None) -> torch.Tensor:
    """
    Create mock logits (unnormalized log probabilities) that simulate
    a language model's output at a single timestep.

    In a real LLM, these would come from the final linear layer before softmax.

    Args:
        vocab_size: Size of vocabulary
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (vocab_size,) representing logits
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Create logits with some structure (not completely random)
    # Higher values for common words, lower for rare words
    logits = torch.randn(vocab_size) * 2.0

    # Make some tokens more likely (simulate language model behavior)
    # Indices 3-10 are common words in our vocab
    logits[3:11] += 3.0  # Boost common words
    logits[0:3] -= 5.0   # Suppress special tokens

    return logits


# ==============================================================================
# STRATEGY 1: GREEDY DECODING
# ==============================================================================

def greedy_decode(logits: torch.Tensor, vocab: MockVocabulary) -> Tuple[int, str]:
    """
    GREEDY DECODING: Always select the token with highest probability.

    Algorithm:
    1. Apply softmax to logits to get probabilities
    2. Select argmax (token with highest probability)

    Pros:
    - Fast and deterministic
    - Simple to implement
    - Works well for factual/objective tasks

    Cons:
    - No exploration of alternative sequences
    - Can lead to repetitive text
    - May miss globally optimal sequences
    - No diversity in generated text

    Interview Talking Points:
    - "Greedy decoding is the simplest strategy but can be myopic"
    - "It makes locally optimal choices without considering future consequences"
    - "Used when you need deterministic, predictable outputs"

    Args:
        logits: Unnormalized log probabilities from model
        vocab: Vocabulary for decoding

    Returns:
        (token_id, token_string) tuple
    """
    print("\n" + "="*70)
    print("GREEDY DECODING")
    print("="*70)

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # Get the token with maximum probability
    token_id = torch.argmax(probs).item()
    token_str = vocab.decode(token_id)

    print(f"Logits shape: {logits.shape}")
    print(f"Max probability: {probs[token_id]:.4f}")
    print(f"Selected token ID: {token_id}")
    print(f"Selected token: '{token_str}'")

    # Show top 5 alternatives
    print("\nTop 5 alternatives (not explored in greedy):")
    top_probs, top_indices = torch.topk(probs, k=5)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = vocab.decode(idx.item())
        selected = "✓ SELECTED" if i == 0 else ""
        print(f"  {i+1}. '{token}' (prob={prob:.4f}) {selected}")

    return token_id, token_str


# ==============================================================================
# STRATEGY 2: BEAM SEARCH
# ==============================================================================

@dataclass
class Beam:
    """Represents a single beam (hypothesis) in beam search."""
    tokens: List[int]          # Sequence of token IDs
    score: float               # Cumulative log probability
    finished: bool = False     # Whether sequence ended with <eos>

    def __lt__(self, other):
        """For sorting beams by score."""
        return self.score < other.score


def beam_search(
    logits_sequence: List[torch.Tensor],
    vocab: MockVocabulary,
    beam_width: int = 3,
    length_penalty: float = 1.0
) -> List[Beam]:
    """
    BEAM SEARCH: Keep track of top-k most likely sequences at each step.

    Algorithm:
    1. Initialize with <sos> token
    2. At each step:
       - For each beam, get top-k next tokens
       - Create k new candidate sequences per beam
       - Keep only top beam_width candidates overall
    3. Return beams when all finish or max length reached

    Key Concepts:
    - Beam width: Number of hypotheses to maintain
    - Length penalty: Prevents bias toward shorter sequences
    - Score normalization: Divide by length^penalty

    Pros:
    - Better than greedy: explores multiple paths
    - Can find higher probability sequences
    - Configurable exploration vs exploitation

    Cons:
    - More computationally expensive (k times greedy)
    - Still may miss globally optimal sequence
    - Can lead to generic/safe outputs
    - Tends to prefer shorter sequences without length penalty

    Interview Talking Points:
    - "Beam search is a middle ground between greedy and exhaustive search"
    - "Beam width is a key hyperparameter: larger = better quality but slower"
    - "Length penalty prevents bias toward short sequences"
    - "Used in machine translation, summarization, etc."

    Args:
        logits_sequence: List of logit tensors (one per timestep)
        vocab: Vocabulary for decoding
        beam_width: Number of beams to maintain (k)
        length_penalty: Alpha for length normalization (score / len^alpha)

    Returns:
        List of top beams sorted by score
    """
    print("\n" + "="*70)
    print(f"BEAM SEARCH (beam_width={beam_width}, length_penalty={length_penalty})")
    print("="*70)

    # Initialize: Start with a single beam containing <sos> token
    sos_token = vocab.encode('<sos>')
    eos_token = vocab.encode('<eos>')

    beams = [Beam(tokens=[sos_token], score=0.0)]

    print(f"\nInitial beam: [<sos>] (token_id={sos_token})")
    print(f"EOS token ID: {eos_token}")

    # Process each timestep
    for step, logits in enumerate(logits_sequence):
        print(f"\n--- Step {step + 1} ---")

        # Convert logits to log probabilities (for numerical stability)
        log_probs = F.log_softmax(logits, dim=-1)

        # Generate candidates from all active beams
        candidates = []

        for beam_idx, beam in enumerate(beams):
            if beam.finished:
                # Keep finished beams as-is
                candidates.append(beam)
                continue

            # Get top-k tokens for this beam
            top_log_probs, top_indices = torch.topk(log_probs, k=beam_width)

            # Create new candidate beams
            for log_prob, token_id in zip(top_log_probs, top_indices):
                new_tokens = beam.tokens + [token_id.item()]
                new_score = beam.score + log_prob.item()

                # Check if this token is <eos>
                finished = (token_id.item() == eos_token)

                new_beam = Beam(
                    tokens=new_tokens,
                    score=new_score,
                    finished=finished
                )
                candidates.append(new_beam)

                if beam_idx == 0:  # Show details for first beam
                    token_str = vocab.decode(token_id.item())
                    print(f"  Beam {beam_idx} + '{token_str}': score={new_score:.3f}")

        # Apply length penalty and select top beam_width candidates
        # Length penalty: normalized_score = score / (length ** length_penalty)
        scored_candidates = []
        for candidate in candidates:
            length = len(candidate.tokens)
            normalized_score = candidate.score / (length ** length_penalty)
            scored_candidates.append((normalized_score, candidate))

        # Sort by normalized score (higher is better)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Keep top beam_width
        beams = [candidate for _, candidate in scored_candidates[:beam_width]]

        print(f"\nTop {beam_width} beams after step {step + 1}:")
        for i, beam in enumerate(beams):
            tokens_str = [vocab.decode(t) for t in beam.tokens]
            status = "FINISHED" if beam.finished else "active"
            print(f"  {i+1}. {tokens_str} (score={beam.score:.3f}, {status})")

        # Early stopping if all beams are finished
        if all(beam.finished for beam in beams):
            print("\nAll beams finished! Stopping early.")
            break

    print(f"\n{'='*70}")
    print("FINAL BEAMS (sorted by score):")
    print('='*70)
    for i, beam in enumerate(beams):
        tokens_str = [vocab.decode(t) for t in beam.tokens]
        length = len(beam.tokens)
        normalized_score = beam.score / (length ** length_penalty)
        print(f"\nBeam {i+1}:")
        print(f"  Tokens: {tokens_str}")
        print(f"  Score: {beam.score:.4f}")
        print(f"  Length: {length}")
        print(f"  Normalized score: {normalized_score:.4f}")

    return beams


# ==============================================================================
# STRATEGY 3: TEMPERATURE SCALING
# ==============================================================================

def temperature_sampling(
    logits: torch.Tensor,
    vocab: MockVocabulary,
    temperature: float = 1.0,
    num_samples: int = 5
) -> List[Tuple[int, str, float]]:
    """
    TEMPERATURE SCALING: Control randomness/creativity of sampling.

    Algorithm:
    1. Divide logits by temperature: scaled_logits = logits / T
    2. Apply softmax to get probabilities
    3. Sample from the resulting distribution

    Temperature Effects:
    - T = 1.0: Use original distribution (standard sampling)
    - T → 0: Approaches greedy (deterministic, peaked distribution)
    - T > 1: More uniform distribution (more random, creative)
    - T < 1: More peaked distribution (more focused, conservative)

    Mathematical Intuition:
    - Softmax: p_i = exp(logit_i) / Σ exp(logit_j)
    - With temp: p_i = exp(logit_i/T) / Σ exp(logit_j/T)
    - Lower T makes large logits dominate exponentially more
    - Higher T makes all logits more similar after exp()

    Use Cases:
    - T=0.7: Factual tasks (Q&A, math, coding)
    - T=1.0: Balanced tasks (general conversation)
    - T=1.2-1.5: Creative tasks (story writing, brainstorming)

    Interview Talking Points:
    - "Temperature controls the exploration-exploitation tradeoff"
    - "Low temp = more deterministic, high temp = more diverse"
    - "It's like adjusting confidence: low temp = very confident picks"
    - "Different tasks need different temperatures"

    Args:
        logits: Unnormalized log probabilities
        vocab: Vocabulary for decoding
        temperature: Scaling factor (T > 0)
        num_samples: Number of samples to draw

    Returns:
        List of (token_id, token_str, probability) tuples
    """
    print("\n" + "="*70)
    print(f"TEMPERATURE SAMPLING (T={temperature})")
    print("="*70)

    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    print(f"Original logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"Scaled logits range: [{scaled_logits.min():.2f}, {scaled_logits.max():.2f}]")
    print(f"Probability entropy: {-(probs * torch.log(probs + 1e-10)).sum():.4f}")

    # Sample multiple times to show distribution
    samples = []
    print(f"\nDrawing {num_samples} samples:")

    for i in range(num_samples):
        # Sample from categorical distribution
        token_id = torch.multinomial(probs, num_samples=1).item()
        token_str = vocab.decode(token_id)
        prob = probs[token_id].item()

        samples.append((token_id, token_str, prob))
        print(f"  Sample {i+1}: '{token_str}' (prob={prob:.4f})")

    # Show distribution statistics
    print("\nTop 10 most likely tokens:")
    top_probs, top_indices = torch.topk(probs, k=10)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = vocab.decode(idx.item())
        print(f"  {i+1}. '{token}' (prob={prob:.4f})")

    return samples


# ==============================================================================
# STRATEGY 4: TOP-K SAMPLING
# ==============================================================================

def top_k_sampling(
    logits: torch.Tensor,
    vocab: MockVocabulary,
    k: int = 10,
    temperature: float = 1.0,
    num_samples: int = 5
) -> List[Tuple[int, str, float]]:
    """
    TOP-K SAMPLING: Sample from only the k most likely tokens.

    Algorithm:
    1. Apply temperature scaling (optional)
    2. Find top-k tokens by probability
    3. Set all other tokens' probabilities to 0
    4. Renormalize the top-k probabilities
    5. Sample from this restricted distribution

    Key Concepts:
    - k is a hyperparameter: larger k = more diversity
    - Truncates the "long tail" of unlikely tokens
    - Prevents sampling of very low probability (nonsensical) tokens

    Pros:
    - Prevents unlikely/nonsensical tokens
    - More diverse than greedy or beam search
    - Simple and effective
    - Consistent quality across different contexts

    Cons:
    - Fixed k may not fit all contexts
    - When distribution is flat, top-k may be too restrictive
    - When distribution is peaked, top-k may include bad tokens

    Interview Talking Points:
    - "Top-k prevents the model from saying ridiculous things"
    - "It's like giving the model a curated menu of reasonable options"
    - "Fixed k is both strength and weakness - doesn't adapt to context"
    - "Introduced in the paper 'Hierarchical Neural Story Generation' (2018)"

    Args:
        logits: Unnormalized log probabilities
        vocab: Vocabulary for decoding
        k: Number of top tokens to consider
        temperature: Temperature scaling factor
        num_samples: Number of samples to draw

    Returns:
        List of (token_id, token_str, probability) tuples
    """
    print("\n" + "="*70)
    print(f"TOP-K SAMPLING (k={k}, T={temperature})")
    print("="*70)

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Get probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Get top-k probabilities and indices
    top_k_probs, top_k_indices = torch.topk(probs, k=k)

    print(f"Original vocab size: {len(probs)}")
    print(f"Restricted to top-{k} tokens")
    print(f"Top-{k} probability mass: {top_k_probs.sum():.4f}")
    print(f"Excluded probability mass: {1 - top_k_probs.sum():.4f}")

    # Create a new probability distribution with only top-k
    # Set all other probs to 0 and renormalize
    restricted_probs = torch.zeros_like(probs)
    restricted_probs[top_k_indices] = top_k_probs

    # Renormalize (should already sum to ~1, but for numerical stability)
    restricted_probs = restricted_probs / restricted_probs.sum()

    # Sample multiple times
    samples = []
    print(f"\nDrawing {num_samples} samples:")

    for i in range(num_samples):
        token_id = torch.multinomial(restricted_probs, num_samples=1).item()
        token_str = vocab.decode(token_id)
        prob = restricted_probs[token_id].item()

        samples.append((token_id, token_str, prob))
        print(f"  Sample {i+1}: '{token_str}' (prob={prob:.4f})")

    # Show the top-k tokens
    print(f"\nTop-{k} candidate tokens:")
    for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
        token = vocab.decode(idx.item())
        print(f"  {i+1}. '{token}' (prob={prob:.4f})")

    return samples


# ==============================================================================
# STRATEGY 5: TOP-P (NUCLEUS) SAMPLING
# ==============================================================================

def top_p_sampling(
    logits: torch.Tensor,
    vocab: MockVocabulary,
    p: float = 0.9,
    temperature: float = 1.0,
    num_samples: int = 5
) -> List[Tuple[int, str, float]]:
    """
    TOP-P (NUCLEUS) SAMPLING: Sample from smallest set of tokens whose
    cumulative probability exceeds p.

    Algorithm:
    1. Apply temperature scaling (optional)
    2. Sort tokens by probability (descending)
    3. Compute cumulative probability
    4. Find smallest set where cumulative prob >= p
    5. Sample from this "nucleus" set

    Key Concepts:
    - Adaptive: nucleus size varies with distribution shape
    - p is typically 0.9-0.95 (90-95% probability mass)
    - Also called "nucleus sampling"

    Advantages over Top-K:
    - Adapts to context: fewer tokens when confident, more when uncertain
    - When distribution is peaked: nucleus is small (focused)
    - When distribution is flat: nucleus is large (diverse)

    Pros:
    - Adapts to model confidence
    - More context-aware than top-k
    - Maintains quality while allowing creativity
    - Currently preferred method in many LLMs

    Cons:
    - Slightly more complex than top-k
    - Can still include unlikely tokens in flat distributions

    Interview Talking Points:
    - "Top-p adapts to model confidence - that's its key advantage"
    - "It's like having a dynamic top-k based on certainty"
    - "Introduced in 'The Curious Case of Neural Text Degeneration' (2019)"
    - "Most modern LLMs (GPT-3/4, Claude) use top-p by default"
    - "Can be combined with temperature for fine-grained control"

    Args:
        logits: Unnormalized log probabilities
        vocab: Vocabulary for decoding
        p: Cumulative probability threshold (0 < p <= 1)
        temperature: Temperature scaling factor
        num_samples: Number of samples to draw

    Returns:
        List of (token_id, token_str, probability) tuples
    """
    print("\n" + "="*70)
    print(f"TOP-P (NUCLEUS) SAMPLING (p={p}, T={temperature})")
    print("="*70)

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Get probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cutoff index where cumulative prob exceeds p
    # Remove tokens with cumulative probability above p
    sorted_indices_to_remove = cumulative_probs > p

    # Shift right to keep at least one token and the first token above threshold
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    # Create mask for original indices
    indices_to_remove = sorted_indices[sorted_indices_to_remove]

    # Create restricted probability distribution
    restricted_probs = probs.clone()
    restricted_probs[indices_to_remove] = 0.0

    # Count nucleus size
    nucleus_size = (restricted_probs > 0).sum().item()

    print(f"Original vocab size: {len(probs)}")
    print(f"Nucleus size: {nucleus_size} tokens")
    print(f"Nucleus probability mass: {restricted_probs.sum():.4f}")
    print(f"Excluded probability mass: {1 - restricted_probs.sum():.4f}")
    print(f"Nucleus size is {nucleus_size/len(probs)*100:.1f}% of vocabulary")

    # Renormalize
    restricted_probs = restricted_probs / restricted_probs.sum()

    # Sample multiple times
    samples = []
    print(f"\nDrawing {num_samples} samples:")

    for i in range(num_samples):
        token_id = torch.multinomial(restricted_probs, num_samples=1).item()
        token_str = vocab.decode(token_id)
        prob = restricted_probs[token_id].item()

        samples.append((token_id, token_str, prob))
        print(f"  Sample {i+1}: '{token_str}' (prob={prob:.4f})")

    # Show the nucleus tokens (up to 15)
    nucleus_indices = (restricted_probs > 0).nonzero(as_tuple=True)[0]
    nucleus_probs = restricted_probs[nucleus_indices]

    # Sort nucleus by probability
    sorted_nucleus_probs, sorted_nucleus_idx = torch.sort(nucleus_probs, descending=True)
    sorted_nucleus_indices = nucleus_indices[sorted_nucleus_idx]

    print(f"\nTop tokens in nucleus (up to 15):")
    for i, (prob, idx) in enumerate(zip(sorted_nucleus_probs[:15], sorted_nucleus_indices[:15])):
        token = vocab.decode(idx.item())
        cumulative = sorted_nucleus_probs[:i+1].sum().item()
        print(f"  {i+1}. '{token}' (prob={prob:.4f}, cumulative={cumulative:.4f})")

    return samples


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def visualize_temperature_effects(
    logits: torch.Tensor,
    vocab: MockVocabulary,
    temperatures: List[float],
    save_path: str
):
    """
    Visualize how temperature affects probability distributions.

    Creates a plot showing:
    - Original logits
    - Probability distributions at different temperatures
    - Entropy at each temperature

    Args:
        logits: Original logits
        vocab: Vocabulary for labeling
        temperatures: List of temperatures to visualize
        save_path: Path to save the figure
    """
    print("\n" + "="*70)
    print("VISUALIZING TEMPERATURE EFFECTS")
    print("="*70)

    # Select top-k tokens to visualize (for clarity)
    k = 15
    probs_T1 = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs_T1, k=k)

    # Get token labels
    token_labels = [vocab.decode(idx.item()) for idx in top_indices]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temperature Effects on Probability Distribution', fontsize=16, fontweight='bold')

    # Plot for each temperature
    entropies = []

    for idx, temp in enumerate(temperatures):
        ax = axes[idx // 2, idx % 2]

        # Compute probabilities at this temperature
        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=-1)

        # Extract probabilities for top-k tokens
        top_probs_temp = probs[top_indices]

        # Compute entropy: -Σ p(x) log p(x)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)

        # Create bar plot
        bars = ax.bar(range(k), top_probs_temp.numpy(), alpha=0.7, color='steelblue')
        ax.set_xlabel('Token', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Temperature = {temp} (Entropy = {entropy:.3f})', fontsize=14, fontweight='bold')
        ax.set_xticks(range(k))
        ax.set_xticklabels(token_labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Highlight top token
        bars[0].set_color('crimson')
        bars[0].set_alpha(1.0)

        # Add probability labels on bars
        for i, (bar, prob) in enumerate(zip(bars, top_probs_temp)):
            if i < 5:  # Label top 5
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.3f}',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")

    # Print entropy analysis
    print("\nEntropy Analysis:")
    print("(Higher entropy = more uniform/random distribution)")
    for temp, entropy in zip(temperatures, entropies):
        print(f"  T={temp}: Entropy={entropy:.4f}")

    plt.close()


def visualize_sampling_comparison(
    logits: torch.Tensor,
    vocab: MockVocabulary,
    save_path: str
):
    """
    Compare probability distributions for different sampling strategies.

    Creates a multi-panel plot comparing:
    - Original distribution
    - Top-k filtered distribution
    - Top-p (nucleus) filtered distribution
    - Temperature-scaled distribution

    Args:
        logits: Original logits
        vocab: Vocabulary for labeling
        save_path: Path to save the figure
    """
    print("\n" + "="*70)
    print("VISUALIZING SAMPLING STRATEGY COMPARISON")
    print("="*70)

    # Parameters
    k = 10
    p = 0.9
    temperature = 1.2

    # Original probabilities
    probs_original = F.softmax(logits, dim=-1)

    # Get top tokens for consistent x-axis
    top_probs, top_indices = torch.topk(probs_original, k=20)
    token_labels = [vocab.decode(idx.item()) for idx in top_indices]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparison of Sampling Strategies', fontsize=16, fontweight='bold')

    # 1. Original distribution
    ax = axes[0, 0]
    probs_display = probs_original[top_indices].numpy()
    bars = ax.bar(range(len(probs_display)), probs_display, alpha=0.7, color='steelblue')
    bars[0].set_color('crimson')
    ax.set_title('Original Distribution (Softmax)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Token', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # 2. Top-k sampling
    ax = axes[0, 1]
    probs_topk = probs_original.clone()
    top_k_probs, top_k_indices = torch.topk(probs_original, k=k)
    mask = torch.zeros_like(probs_topk, dtype=torch.bool)
    mask[top_k_indices] = True
    probs_topk[~mask] = 0
    probs_topk = probs_topk / probs_topk.sum()  # Renormalize

    probs_display = probs_topk[top_indices].numpy()
    bars = ax.bar(range(len(probs_display)), probs_display, alpha=0.7, color='forestgreen')
    for i in range(min(k, len(bars))):
        bars[i].set_alpha(1.0)
    ax.set_title(f'Top-k Sampling (k={k})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Token', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(x=k-0.5, color='red', linestyle='--', linewidth=2, label=f'Top-{k} cutoff')
    ax.legend()

    # 3. Top-p (nucleus) sampling
    ax = axes[1, 0]
    sorted_probs, sorted_indices = torch.sort(probs_original, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    probs_topp = probs_original.clone()
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs_topp[indices_to_remove] = 0
    nucleus_size = (probs_topp > 0).sum().item()
    probs_topp = probs_topp / probs_topp.sum()  # Renormalize

    probs_display = probs_topp[top_indices].numpy()
    bars = ax.bar(range(len(probs_display)), probs_display, alpha=0.7, color='darkorange')
    for i, bar in enumerate(bars):
        if probs_display[i] > 0:
            bar.set_alpha(1.0)
    ax.set_title(f'Top-p Sampling (p={p}, nucleus_size={nucleus_size})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Token', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # 4. Temperature scaling
    ax = axes[1, 1]
    scaled_logits = logits / temperature
    probs_temp = F.softmax(scaled_logits, dim=-1)

    probs_display = probs_temp[top_indices].numpy()
    bars = ax.bar(range(len(probs_display)), probs_display, alpha=0.7, color='mediumpurple')
    bars[0].set_color('crimson')
    ax.set_title(f'Temperature Scaling (T={temperature})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Token', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")
    plt.close()


def visualize_beam_search_tree(save_path: str):
    """
    Create a conceptual visualization of how beam search explores the search space.

    Shows:
    - Search tree with multiple beams
    - Pruned paths vs kept paths
    - Comparison with greedy search

    Args:
        save_path: Path to save the figure
    """
    print("\n" + "="*70)
    print("VISUALIZING BEAM SEARCH TREE")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Greedy Decoding vs Beam Search', fontsize=16, fontweight='bold')

    # Greedy search visualization
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Greedy Decoding (Single Path)', fontsize=14, fontweight='bold')

    # Draw greedy path
    positions = [(1, 5), (3, 6), (5, 7), (7, 8), (9, 7)]
    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=3, alpha=0.8)
        ax.plot(x1, y1, 'ro', markersize=15)
    ax.plot(positions[-1][0], positions[-1][1], 'ro', markersize=15)

    # Add labels
    ax.text(1, 5, 'START', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(9, 7, 'END', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Show unexplored alternatives
    unexplored = [(3, 4), (3, 5), (5, 5), (5, 6), (7, 6), (7, 7)]
    for x, y in unexplored:
        ax.plot(x, y, 'o', color='lightgray', markersize=10, alpha=0.5)

    ax.text(5, 1, 'Only explores highest\nprobability path', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Beam search visualization
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Beam Search (beam_width=3)', fontsize=14, fontweight='bold')

    # Draw multiple beams
    beam_paths = [
        [(1, 5), (3, 6), (5, 7), (7, 8), (9, 7)],   # Beam 1
        [(1, 5), (3, 5), (5, 6), (7, 7), (9, 6)],   # Beam 2
        [(1, 5), (3, 4), (5, 5), (7, 6), (9, 5)],   # Beam 3
    ]
    colors = ['crimson', 'forestgreen', 'darkorange']

    for path, color in zip(beam_paths, colors):
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=2.5, alpha=0.7)
            ax.plot(x1, y1, 'o', color=color, markersize=12)
        ax.plot(path[-1][0], path[-1][1], 'o', color=color, markersize=12)

    # Add labels
    ax.text(1, 5, 'START', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.text(9, 7, 'Path 1', ha='center', va='center', fontsize=8, color='white')
    ax.text(9, 6, 'Path 2', ha='center', va='center', fontsize=8, color='white')
    ax.text(9, 5, 'Path 3', ha='center', va='center', fontsize=8, color='white')

    ax.text(5, 1, 'Explores top-k paths\nin parallel', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")
    plt.close()


# ==============================================================================
# MAIN DEMONSTRATION
# ==============================================================================

def main():
    """
    Main demonstration function that runs all sampling strategies and creates
    visualizations for interview preparation.
    """
    print("="*70)
    print("TEXT GENERATION SAMPLING STRATEGIES DEMO")
    print("LLM Interview Preparation")
    print("="*70)

    # Create output directory
    output_dir = "/Users/zack/dev/ml-refresher/data/interview_viz/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Initialize mock vocabulary
    vocab = MockVocabulary()
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Sample tokens: {vocab.vocab[:10]}")

    # Create mock logits for a single timestep
    logits = create_mock_logits(len(vocab), seed=42)

    print("\n" + "="*70)
    print("PART 1: SINGLE TIMESTEP DEMONSTRATIONS")
    print("="*70)

    # 1. Greedy Decoding
    greedy_token_id, greedy_token = greedy_decode(logits, vocab)

    # 2. Temperature Sampling (multiple temperatures)
    temps = [0.5, 1.0, 1.5]
    print("\n" + "="*70)
    print("COMPARING DIFFERENT TEMPERATURES")
    print("="*70)

    for temp in temps:
        temperature_sampling(logits, vocab, temperature=temp, num_samples=3)

    # 3. Top-k Sampling
    top_k_sampling(logits, vocab, k=10, temperature=1.0, num_samples=5)

    # 4. Top-p Sampling
    top_p_sampling(logits, vocab, p=0.9, temperature=1.0, num_samples=5)

    print("\n" + "="*70)
    print("PART 2: MULTI-STEP BEAM SEARCH DEMONSTRATION")
    print("="*70)

    # Create a sequence of logits for beam search (simulate 4 timesteps)
    logits_sequence = [
        create_mock_logits(len(vocab), seed=42 + i) for i in range(4)
    ]

    # 5. Beam Search
    beams = beam_search(logits_sequence, vocab, beam_width=3, length_penalty=0.6)

    print("\n" + "="*70)
    print("PART 3: VISUALIZATIONS")
    print("="*70)

    # Visualization 1: Temperature effects
    visualize_temperature_effects(
        logits, vocab,
        temperatures=[0.5, 0.8, 1.0, 1.5],
        save_path=os.path.join(output_dir, "temperature_effects.png")
    )

    # Visualization 2: Strategy comparison
    visualize_sampling_comparison(
        logits, vocab,
        save_path=os.path.join(output_dir, "sampling_comparison.png")
    )

    # Visualization 3: Beam search tree
    visualize_beam_search_tree(
        save_path=os.path.join(output_dir, "beam_search_tree.png")
    )

    print("\n" + "="*70)
    print("INTERVIEW KEY TAKEAWAYS")
    print("="*70)
    print("""
1. GREEDY DECODING:
   - Simplest but myopic
   - Always picks highest probability token
   - Good for: deterministic tasks (math, code)
   - Bad for: creative generation

2. BEAM SEARCH:
   - Keeps top-k sequences at each step
   - Better than greedy for finding high-quality sequences
   - Requires length penalty to avoid short sequences
   - Good for: translation, summarization
   - Computational cost: O(k × vocab_size) per step

3. TEMPERATURE:
   - Controls randomness/creativity
   - T→0: deterministic (like greedy)
   - T>1: more random/creative
   - T<1: more focused/conservative
   - Always used with other sampling methods

4. TOP-K SAMPLING:
   - Fixed cutoff: only sample from top-k tokens
   - Prevents unlikely/nonsensical tokens
   - Cons: k doesn't adapt to context
   - Good for: general text generation

5. TOP-P (NUCLEUS) SAMPLING:
   - Adaptive cutoff based on cumulative probability
   - Nucleus grows/shrinks with model confidence
   - More flexible than top-k
   - Current best practice in most LLMs
   - Often combined with temperature

PRACTICAL TIPS:
- Combine strategies: temperature + top-p is common
- Different tasks need different settings
- Factual tasks: low temp, greedy/beam search
- Creative tasks: higher temp, top-p
- GPT-3 default: temperature=1.0, top-p=1.0 (no truncation)
- Claude default: similar to top-p with temperature
    """)

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for filename in os.listdir(output_dir):
        if filename.endswith('.png'):
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
