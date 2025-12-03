"""
Comprehensive Tokenization Demo for LLM Interview Preparation
==============================================================

This module demonstrates key tokenization concepts for ML/LLM interviews:
- Q1: What is tokenization?
- Q16: How do LLMs handle Out-of-Vocabulary (OOV) words?

Topics covered:
1. Word-level tokenization
2. Character-level tokenization
3. Byte-Pair Encoding (BPE) from scratch
4. Subword tokenization
5. OOV handling with different approaches
6. Vocabulary building process

Author: Interview Preparation Material
Date: 2024
"""

from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import re


# =============================================================================
# SECTION 1: BASIC TOKENIZATION APPROACHES
# =============================================================================

class WordLevelTokenizer:
    """
    Word-level tokenization: Split text into words.

    Pros:
    - Simple and intuitive
    - Preserves word boundaries
    - Fast

    Cons:
    - Large vocabulary size
    - Can't handle OOV (Out-of-Vocabulary) words
    - Poor generalization to unseen words
    - Requires extensive vocabulary for good coverage
    """

    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.next_id = 4

    def train(self, texts: List[str]):
        """Build vocabulary from training texts."""
        print("\n" + "="*70)
        print("WORD-LEVEL TOKENIZATION - TRAINING")
        print("="*70)

        word_counts = Counter()
        for text in texts:
            # Simple word splitting (lowercase, split on whitespace/punctuation)
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts.update(words)

        print(f"\nTotal unique words found: {len(word_counts)}")
        print(f"Most common words: {word_counts.most_common(10)}")

        # Add words to vocabulary
        for word, count in word_counts.items():
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.id_to_token[self.next_id] = word
                self.next_id += 1

        print(f"Final vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        words = re.findall(r'\b\w+\b', text.lower())
        # Use <UNK> token for words not in vocabulary (OOV handling)
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return " ".join([self.id_to_token.get(tid, "<UNK>") for tid in token_ids])


class CharLevelTokenizer:
    """
    Character-level tokenization: Split text into individual characters.

    Pros:
    - Very small vocabulary (typically 50-300 characters)
    - No OOV problem (can represent any text)
    - Good for morphologically rich languages

    Cons:
    - Very long sequences (increases computational cost)
    - Model needs to learn word structure from scratch
    - Loses semantic word boundaries
    """

    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.next_id = 4

    def train(self, texts: List[str]):
        """Build character vocabulary from training texts."""
        print("\n" + "="*70)
        print("CHARACTER-LEVEL TOKENIZATION - TRAINING")
        print("="*70)

        char_counts = Counter()
        for text in texts:
            char_counts.update(text)

        print(f"\nTotal unique characters found: {len(char_counts)}")
        print(f"Characters: {sorted(char_counts.keys())}")

        # Add characters to vocabulary
        for char, count in char_counts.items():
            if char not in self.vocab:
                self.vocab[char] = self.next_id
                self.id_to_token[self.next_id] = char
                self.next_id += 1

        print(f"Final vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return [self.vocab.get(char, self.vocab["<UNK>"]) for char in text]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return "".join([self.id_to_token.get(tid, "<UNK>") for tid in token_ids])


# =============================================================================
# SECTION 2: BYTE-PAIR ENCODING (BPE) - THE HEART OF MODERN TOKENIZATION
# =============================================================================

class SimpleBPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer - Manual Implementation from Scratch

    BPE is the foundation of modern tokenization used in GPT, BERT, and other LLMs.

    Algorithm:
    1. Start with character-level vocabulary
    2. Iteratively merge the most frequent pair of consecutive tokens
    3. Repeat until desired vocabulary size is reached

    Pros:
    - Balanced vocabulary size (between char and word level)
    - Handles OOV words by breaking into subwords
    - Data-driven approach (learns from corpus)
    - Good compression ratio

    Cons:
    - Tokenization is not always linguistically meaningful
    - Training can be slow on large corpora
    - Greedy algorithm (not optimal)

    This is how GPT, RoBERTa, and many other models handle tokenization!
    """

    def __init__(self, vocab_size: int = 300):
        """
        Args:
            vocab_size: Target vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []  # List of (pair, merged_token) tuples
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def _get_word_frequencies(self, texts: List[str]) -> Dict[Tuple[str, ...], int]:
        """
        Count word frequencies and represent each word as a tuple of characters.
        We add a special end-of-word marker '</w>' to distinguish word boundaries.

        Example: "hello" -> ('h', 'e', 'l', 'l', 'o', '</w>')
        """
        word_freqs = defaultdict(int)

        for text in texts:
            # Extract words
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                # Represent word as tuple of chars with end marker
                word_tuple = tuple(word) + ('</w>',)
                word_freqs[word_tuple] += 1

        return word_freqs

    def _get_pair_frequencies(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """
        Count how often each pair of consecutive tokens appears.

        Example: For ('h', 'e', 'l', 'l', 'o', '</w>'):
        Pairs: ('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o'), ('o', '</w>')
        """
        pair_freqs = Counter()

        for word, freq in word_freqs.items():
            # Get all consecutive pairs in this word
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq

        return pair_freqs

    def _merge_pair(self, word: Tuple[str, ...], pair: Tuple[str, str],
                    merged: str) -> Tuple[str, ...]:
        """
        Merge all instances of a pair in a word.

        Example:
            word = ('h', 'e', 'l', 'l', 'o', '</w>')
            pair = ('l', 'l')
            merged = 'll'
            result = ('h', 'e', 'll', 'o', '</w>')
        """
        new_word = []
        i = 0

        while i < len(word):
            # Check if current position matches the pair
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged)
                i += 2  # Skip both tokens
            else:
                new_word.append(word[i])
                i += 1

        return tuple(new_word)

    def train(self, texts: List[str], verbose: bool = True):
        """
        Train BPE tokenizer by iteratively merging most frequent pairs.

        This is the core BPE algorithm!
        """
        if verbose:
            print("\n" + "="*70)
            print("BYTE-PAIR ENCODING (BPE) - TRAINING")
            print("="*70)
            print(f"\nTarget vocabulary size: {self.vocab_size}")

        # Step 1: Get initial word frequencies (character-level)
        word_freqs = self._get_word_frequencies(texts)

        if verbose:
            print(f"\nInitial words (character-level representation):")
            for word, freq in list(word_freqs.items())[:5]:
                print(f"  {''.join(word):20s} -> {word} (freq: {freq})")

        # Step 2: Build initial vocabulary from characters
        base_vocab = set()
        for word in word_freqs.keys():
            base_vocab.update(word)

        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        for token in sorted(base_vocab):
            self.vocab[token] = len(self.vocab)

        initial_vocab_size = len(self.vocab)
        if verbose:
            print(f"\nInitial vocabulary size (character-level): {initial_vocab_size}")
            print(f"Sample tokens: {list(self.vocab.keys())[:20]}")

        # Step 3: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - initial_vocab_size
        if verbose:
            print(f"\nPerforming {num_merges} merge operations...")
            print("-" * 70)

        for merge_idx in range(num_merges):
            # Count pair frequencies
            pair_freqs = self._get_pair_frequencies(word_freqs)

            if not pair_freqs:
                if verbose:
                    print(f"\nNo more pairs to merge. Stopping at {len(self.vocab)} tokens.")
                break

            # Get most frequent pair
            most_frequent_pair = pair_freqs.most_common(1)[0]
            pair, freq = most_frequent_pair

            # Create merged token
            merged_token = ''.join(pair)

            # Store merge operation
            self.merges.append((pair, merged_token))

            # Add merged token to vocabulary
            self.vocab[merged_token] = len(self.vocab)

            if verbose and (merge_idx < 10 or merge_idx % 50 == 0):
                print(f"Merge {merge_idx + 1:3d}: {pair[0]:10s} + {pair[1]:10s} "
                      f"-> {merged_token:15s} (freq: {freq:5d})")

            # Update word frequencies with merged token
            new_word_freqs = {}
            for word, word_freq in word_freqs.items():
                new_word = self._merge_pair(word, pair, merged_token)
                new_word_freqs[new_word] = word_freq
            word_freqs = new_word_freqs

        if verbose:
            print("-" * 70)
            print(f"\n✓ Training complete!")
            print(f"  Final vocabulary size: {len(self.vocab)}")
            print(f"  Total merges performed: {len(self.merges)}")
            print(f"\nSample merged tokens:")
            for token in list(self.vocab.keys())[-10:]:
                print(f"  {token}")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned BPE merges.

        This shows how BPE handles OOV words: it breaks them into known subwords!
        """
        # Start with character-level representation
        word_tokens = list(word) + ['</w>']

        # Apply merges in the order they were learned
        for pair, merged_token in self.merges:
            i = 0
            while i < len(word_tokens) - 1:
                if (word_tokens[i], word_tokens[i + 1]) == pair:
                    # Merge this pair
                    word_tokens = (word_tokens[:i] +
                                  [merged_token] +
                                  word_tokens[i + 2:])
                else:
                    i += 1

        return word_tokens

    def encode(self, text: str, verbose: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Demonstrates OOV handling: Even unseen words can be tokenized!
        """
        words = re.findall(r'\b\w+\b', text.lower())

        all_tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            all_tokens.extend(word_tokens)

        if verbose:
            print(f"\nTokenization breakdown:")
            print(f"  Original text: {text}")
            print(f"  Tokens: {all_tokens}")

        # Convert tokens to IDs (use <UNK> for any tokens not in vocab)
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in all_tokens]

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(tid, "<UNK>") for tid in token_ids]

        # Join tokens and remove end-of-word markers
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text

    def show_vocabulary_sample(self, n: int = 30):
        """Display a sample of the learned vocabulary."""
        print(f"\n{'='*70}")
        print(f"VOCABULARY SAMPLE (showing {n} tokens)")
        print('='*70)

        tokens = list(self.vocab.keys())

        print("\nSpecial tokens:")
        for token in tokens[:4]:
            print(f"  {token:20s} -> ID {self.vocab[token]}")

        print("\nCharacter tokens:")
        char_tokens = [t for t in tokens[4:] if len(t) == 1][:10]
        for token in char_tokens:
            print(f"  '{token}':20s -> ID {self.vocab[token]}")

        print("\nSubword tokens (merged):")
        subword_tokens = [t for t in tokens if len(t) > 1 and t != '</w>'][-n:]
        for token in subword_tokens:
            print(f"  {token:20s} -> ID {self.vocab[token]}")


# =============================================================================
# SECTION 3: COMPARISON AND OOV HANDLING DEMONSTRATION
# =============================================================================

def compare_tokenization_methods():
    """
    Compare different tokenization methods on the same text.
    Demonstrates how each method handles vocabulary and sequence length.
    """
    print("\n" + "="*70)
    print("TOKENIZATION METHOD COMPARISON")
    print("="*70)

    # Training corpus
    training_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming the world",
        "Natural language processing enables computers to understand text",
        "Tokenization is a fundamental step in text processing",
        "Deep learning models require careful preprocessing",
        "The transformer architecture revolutionized NLP",
        "Attention mechanisms allow models to focus on relevant parts",
        "Large language models are trained on massive datasets",
    ]

    # Test text with OOV words
    test_text = "The extraordinary supercomputer processes unimaginable amounts of information"

    print("\nTraining corpus:")
    for i, text in enumerate(training_texts, 1):
        print(f"  {i}. {text}")

    print(f"\n{'='*70}")
    print("Test text (contains OOV words):")
    print(f"  \"{test_text}\"")
    print('='*70)

    # Method 1: Word-level
    print("\n\n" + "+"*70)
    print("METHOD 1: WORD-LEVEL TOKENIZATION")
    print("+"*70)

    word_tokenizer = WordLevelTokenizer()
    word_tokenizer.train(training_texts)

    word_ids = word_tokenizer.encode(test_text)
    word_decoded = word_tokenizer.decode(word_ids)

    print(f"\nTest text encoding:")
    print(f"  Token IDs: {word_ids}")
    print(f"  Number of tokens: {len(word_ids)}")
    print(f"  Decoded: {word_decoded}")

    # Count UNK tokens
    unk_count = word_ids.count(word_tokenizer.vocab["<UNK>"])
    print(f"\n  ⚠️  OOV words replaced with <UNK>: {unk_count} out of {len(word_ids)} tokens")
    print(f"  📊 Vocabulary size: {len(word_tokenizer.vocab)}")

    # Method 2: Character-level
    print("\n\n" + "+"*70)
    print("METHOD 2: CHARACTER-LEVEL TOKENIZATION")
    print("+"*70)

    char_tokenizer = CharLevelTokenizer()
    char_tokenizer.train(training_texts)

    char_ids = char_tokenizer.encode(test_text)
    char_decoded = char_tokenizer.decode(char_ids)

    print(f"\nTest text encoding:")
    print(f"  Token IDs: {char_ids[:50]}... (truncated)")
    print(f"  Number of tokens: {len(char_ids)}")
    print(f"  Decoded: {char_decoded}")

    unk_count = char_ids.count(char_tokenizer.vocab["<UNK>"])
    print(f"\n  ✓ OOV words: {unk_count} (character-level handles all text!)")
    print(f"  📊 Vocabulary size: {len(char_tokenizer.vocab)}")

    # Method 3: BPE
    print("\n\n" + "+"*70)
    print("METHOD 3: BYTE-PAIR ENCODING (BPE)")
    print("+"*70)

    bpe_tokenizer = SimpleBPETokenizer(vocab_size=200)
    bpe_tokenizer.train(training_texts, verbose=True)

    print(f"\nTest text encoding:")
    bpe_ids = bpe_tokenizer.encode(test_text, verbose=True)
    bpe_decoded = bpe_tokenizer.decode(bpe_ids)

    print(f"\n  Token IDs: {bpe_ids}")
    print(f"  Number of tokens: {len(bpe_ids)}")
    print(f"  Decoded: {bpe_decoded}")

    unk_count = bpe_ids.count(bpe_tokenizer.vocab["<UNK>"])
    print(f"\n  ✓ OOV words: {unk_count} (BPE breaks unknown words into subwords!)")
    print(f"  📊 Vocabulary size: {len(bpe_tokenizer.vocab)}")

    # Summary comparison
    print("\n\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    comparison_data = [
        ("Method", "Vocab Size", "Token Count", "OOV Handling"),
        ("-" * 20, "-" * 12, "-" * 12, "-" * 30),
        ("Word-level", len(word_tokenizer.vocab), len(word_ids),
         f"{unk_count} <UNK> tokens (poor)"),
        ("Character-level", len(char_tokenizer.vocab), len(char_ids),
         "Perfect (no OOV)"),
        ("BPE (subword)", len(bpe_tokenizer.vocab), len(bpe_ids),
         "Excellent (breaks into subwords)"),
    ]

    for row in comparison_data:
        print(f"{row[0]:20s} | {str(row[1]):12s} | {str(row[2]):12s} | {row[3]:30s}")

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
    1. Word-level:
       - Largest vocabulary, but can't handle unseen words
       - Best for: Fixed vocabulary tasks, simple applications

    2. Character-level:
       - Smallest vocabulary, handles all text
       - Very long sequences (computational cost)
       - Best for: Character-level tasks, morphologically rich languages

    3. BPE (Subword):
       - Balanced vocabulary size and sequence length
       - Handles OOV by breaking into known subwords
       - Best for: Modern LLMs (GPT, BERT, etc.)
       - This is the STANDARD for most LLMs today!
    """)


# =============================================================================
# SECTION 4: OOV HANDLING DEMONSTRATION
# =============================================================================

def demonstrate_oov_handling():
    """
    Detailed demonstration of how different tokenizers handle OOV words.

    This is crucial for interview question: "How do LLMs handle OOV words?"
    """
    print("\n\n" + "="*70)
    print("OUT-OF-VOCABULARY (OOV) HANDLING DEMONSTRATION")
    print("="*70)

    # Simple training corpus
    training_texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a cat and a dog are friends",
    ]

    # Test cases with increasing OOV complexity
    test_cases = [
        ("the cat sat", "All words in vocabulary"),
        ("the elephant sat", "One OOV word: 'elephant'"),
        ("the cat slept", "One OOV word: 'slept'"),
        ("extraordinary elephants communicate", "All OOV words"),
        ("antidisestablishmentarianism", "Very long OOV word"),
    ]

    print("\nTraining corpus (small, limited vocabulary):")
    for text in training_texts:
        print(f"  - {text}")

    # Train BPE tokenizer
    print("\n" + "-"*70)
    bpe = SimpleBPETokenizer(vocab_size=100)
    bpe.train(training_texts, verbose=False)
    print(f"BPE vocabulary size: {len(bpe.vocab)}")

    # Show how BPE breaks down OOV words
    print("\n" + "="*70)
    print("HOW BPE HANDLES OOV WORDS")
    print("="*70)

    for test_text, description in test_cases:
        print(f"\n{'-'*70}")
        print(f"Test: {description}")
        print(f"Text: \"{test_text}\"")
        print(f"{'-'*70}")

        # Tokenize
        words = test_text.split()
        for word in words:
            tokens = bpe._tokenize_word(word)
            token_ids = [bpe.vocab.get(t, bpe.vocab["<UNK>"]) for t in tokens]

            in_vocab = all(t in bpe.vocab for t in tokens)
            status = "✓ IN VOCAB" if in_vocab else "⚠ OOV (decomposed)"

            print(f"\n  Word: '{word}' {status}")
            print(f"    Subword tokens: {tokens}")
            print(f"    Token IDs: {token_ids}")

            if not in_vocab:
                print(f"    → BPE broke this OOV word into {len(tokens)} known subwords!")

    print("\n" + "="*70)
    print("KEY CONCEPT: SUBWORD TOKENIZATION FOR OOV HANDLING")
    print("="*70)
    print("""
    How LLMs handle OOV words (Interview Answer):

    1. Traditional word-level: Replace OOV with <UNK> token (loses information)

    2. Modern approach (BPE/WordPiece/SentencePiece):
       - Break OOV words into known subword units
       - NEVER need <UNK> token (in practice)
       - Examples:
         * "unhappiness" → ["un", "happi", "ness"]
         * "COVID-19" → ["CO", "VID", "-", "19"]
         * "transformer" → ["trans", "former"]

    3. Why this works:
       - Most words share common prefixes, suffixes, roots
       - Model can compose meaning from subwords
       - Open vocabulary: Can represent any text
       - More efficient than character-level (shorter sequences)

    4. Used in:
       - GPT (BPE)
       - BERT (WordPiece, similar to BPE)
       - T5, ALBERT (SentencePiece)
       - RoBERTa (Byte-level BPE)
    """)


# =============================================================================
# SECTION 5: VOCABULARY BUILDING VISUALIZATION
# =============================================================================

def visualize_vocabulary_building():
    """
    Step-by-step visualization of how BPE builds vocabulary.
    Great for understanding the algorithm deeply.
    """
    print("\n\n" + "="*70)
    print("BPE VOCABULARY BUILDING - STEP-BY-STEP VISUALIZATION")
    print("="*70)

    # Very simple corpus for clear visualization
    texts = ["low", "lower", "newest", "widest"]

    print("\nTraining corpus:")
    for word in texts:
        print(f"  - {word}")

    print("\n" + "="*70)
    print("STEP-BY-STEP MERGE PROCESS")
    print("="*70)

    # Manual demonstration (simplified)
    print("\nInitial state (character-level):")
    initial_words = {
        ('l', 'o', 'w', '</w>'): 1,
        ('l', 'o', 'w', 'e', 'r', '</w>'): 1,
        ('n', 'e', 'w', 'e', 's', 't', '</w>'): 1,
        ('w', 'i', 'd', 'e', 's', 't', '</w>'): 1,
    }

    for word, freq in initial_words.items():
        print(f"  {''.join(word):15s} -> {word}")

    print("\n" + "-"*70)
    print("Let's trace the first few merges:")
    print("-"*70)

    # Train with verbose output
    bpe = SimpleBPETokenizer(vocab_size=50)

    # Manually show merge process
    word_freqs = bpe._get_word_frequencies(texts)

    for merge_step in range(5):
        pair_freqs = bpe._get_pair_frequencies(word_freqs)

        if not pair_freqs:
            break

        most_frequent = pair_freqs.most_common(1)[0]
        pair, freq = most_frequent
        merged = ''.join(pair)

        print(f"\nMerge {merge_step + 1}:")
        print(f"  Most frequent pair: {pair[0]} + {pair[1]} = '{merged}' (appears {freq} times)")

        # Show before/after for each word
        print(f"  Effects on vocabulary:")
        new_word_freqs = {}
        for word, word_freq in word_freqs.items():
            if pair[0] in word and pair[1] in word:
                new_word = bpe._merge_pair(word, pair, merged)
                if word != new_word:
                    print(f"    {''.join(word):20s} -> {''.join(new_word)}")
                new_word_freqs[new_word] = word_freq
            else:
                new_word_freqs[word] = word_freq

        word_freqs = new_word_freqs
        bpe.merges.append((pair, merged))

    print("\n" + "="*70)
    print("FINAL VOCABULARY STRUCTURE")
    print("="*70)

    print("\nVocabulary layers:")
    print("  Layer 1 (Base): Individual characters")
    print("  Layer 2: Common pairs (e.g., 'es', 'st', 'er')")
    print("  Layer 3: Longer subwords (e.g., 'est', 'low', 'new')")
    print("  Layer 4+: Full words (e.g., 'lowest', 'newest')")

    print("\nThis hierarchical structure allows:")
    print("  ✓ Efficient representation of common words")
    print("  ✓ Graceful handling of rare/OOV words")
    print("  ✓ Balanced between vocabulary size and sequence length")


# =============================================================================
# SECTION 6: PRACTICAL INTERVIEW INSIGHTS
# =============================================================================

def interview_key_points():
    """
    Summary of key points for interviews.
    """
    print("\n\n" + "="*70)
    print("INTERVIEW KEY POINTS SUMMARY")
    print("="*70)

    print("""
    Q1: What is tokenization?
    -------------------------
    Answer:
    Tokenization is the process of breaking down text into smaller units called
    tokens, which are the basic units that machine learning models process.

    Three main approaches:
    1. Word-level: Split on whitespace/punctuation
       - Pro: Intuitive, preserves word semantics
       - Con: Large vocabulary, poor OOV handling

    2. Character-level: Individual characters
       - Pro: Small vocabulary, no OOV problem
       - Con: Very long sequences, loses word structure

    3. Subword-level (BPE/WordPiece): Data-driven
       - Pro: Balanced, handles OOV, used in modern LLMs
       - Con: Not linguistically motivated

    Modern LLMs use subword tokenization (BPE variants).


    Q16: How do LLMs handle OOV (Out-of-Vocabulary) words?
    --------------------------------------------------------
    Answer:
    Modern LLMs use subword tokenization (BPE, WordPiece, SentencePiece) which
    handles OOV words by breaking them into known subword units.

    Process:
    1. Word not in vocabulary → Break into smaller subwords
    2. Continue until all pieces are in vocabulary
    3. In worst case → Break down to characters (always in vocab)

    Example: "unhappiness" (OOV) → ["un", "happiness"] or ["un", "happi", "ness"]

    Advantages:
    - Open vocabulary (can represent any text)
    - No information loss from <UNK> tokens
    - Model learns compositional semantics
    - More efficient than character-level

    This is why GPT can handle:
    - Rare words: "antidisestablishmentarianism"
    - Neologisms: "COVID-19", "blockchain"
    - Typos: "helllo" → ["hell", "lo"]
    - Code: "def_function_name" → ["def", "_", "function", "_", "name"]


    Implementation in practice:
    --------------------------
    - GPT: BPE (Byte-Pair Encoding)
    - BERT: WordPiece (similar to BPE)
    - T5/ALBERT: SentencePiece (works directly on Unicode)
    - RoBERTa: Byte-level BPE (handles any byte sequence)


    Important details for interviews:
    ---------------------------------
    1. Vocabulary size typical ranges:
       - GPT-2: 50,257 tokens
       - BERT: 30,522 tokens
       - GPT-3: 50,257 tokens

    2. Special tokens:
       - <PAD>: Padding for batch processing
       - <UNK>: Unknown (rarely used with BPE)
       - <BOS>/<EOS>: Begin/End of sequence
       - <SEP>/<CLS>: BERT-specific

    3. Trade-offs:
       - Larger vocab → shorter sequences, larger embedding matrix
       - Smaller vocab → longer sequences, more computation
       - Sweet spot: 30k-50k tokens for most applications

    4. Why this matters:
       - Affects model size (embedding matrix)
       - Affects inference speed (sequence length)
       - Affects model's ability to generalize
       - Critical for multilingual models
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Run all demonstrations.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║           TOKENIZATION DEMO FOR LLM INTERVIEW PREPARATION            ║
    ║                                                                       ║
    ║  This demo covers:                                                    ║
    ║  - Q1: What is tokenization?                                         ║
    ║  - Q16: How do LLMs handle OOV words?                                ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Run all demonstrations
    compare_tokenization_methods()
    demonstrate_oov_handling()
    visualize_vocabulary_building()

    # Show interview key points
    interview_key_points()

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nNext steps for interview preparation:")
    print("  1. Re-read the code and understand each section")
    print("  2. Practice explaining BPE algorithm in your own words")
    print("  3. Be ready to discuss trade-offs of different approaches")
    print("  4. Know vocabulary sizes of popular models (GPT, BERT)")
    print("  5. Understand why subword tokenization is the standard")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
