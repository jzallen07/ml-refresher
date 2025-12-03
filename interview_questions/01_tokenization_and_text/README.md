# Tokenization & Text Processing

## Interview Questions Covered
- **Q1**: What does tokenization entail, and why is it critical for LLMs?
- **Q16**: How do LLMs manage out-of-vocabulary (OOV) words?

---

## Q1: What does tokenization entail, and why is it critical for LLMs?

### Answer

**Tokenization** is the process of breaking down text into smaller units called **tokens**. These can be:
- **Words**: "Hello world" → ["Hello", "world"]
- **Subwords**: "artificial" → ["art", "ific", "ial"]
- **Characters**: "cat" → ["c", "a", "t"]

### Why It's Critical

1. **LLMs process numbers, not text**: Neural networks operate on numerical representations. Tokenization converts text into token IDs that map to embedding vectors.

2. **Vocabulary management**: A fixed vocabulary size (e.g., 50,000 tokens) keeps the model tractable. Subword tokenization balances vocabulary size with coverage.

3. **Handling diverse languages**: Subword methods like BPE can represent any text, including rare words, misspellings, and multiple languages.

4. **Computational efficiency**: Shorter sequences (fewer tokens) mean faster training and inference.

### Common Tokenization Methods

| Method | Description | Example |
|--------|-------------|---------|
| **Word-level** | Split on whitespace/punctuation | "don't" → ["don", "'", "t"] |
| **Character-level** | Each character is a token | "cat" → ["c", "a", "t"] |
| **BPE** (Byte-Pair Encoding) | Merge frequent character pairs iteratively | "lowest" → ["low", "est"] |
| **WordPiece** | Similar to BPE, used by BERT | "playing" → ["play", "##ing"] |
| **SentencePiece** | Language-agnostic, treats space as token | "▁Hello▁world" |

### Interview Tips

- Explain the tradeoff: larger vocabulary = better representation but more parameters
- Mention that GPT uses BPE, BERT uses WordPiece
- Know that tokenization affects context window (token count, not word count)

---

## Q16: How do LLMs manage out-of-vocabulary (OOV) words?

### Answer

LLMs use **subword tokenization** (like BPE) to handle OOV words by breaking them into known subword units.

### Example
```
"cryptocurrency" → ["crypto", "currency"]
"transformerization" → ["transform", "er", "ization"]
```

### Why This Works

1. **Compositionality**: Many words share roots, prefixes, and suffixes
2. **Graceful degradation**: Unknown words are represented as combinations of known parts
3. **No UNK tokens**: Unlike word-level tokenizers, subword methods rarely need an "unknown" token

### The BPE Algorithm (Simplified)

1. Start with character-level vocabulary
2. Count all adjacent character pairs in training data
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary size is reached

```python
# Starting vocabulary: ['l', 'o', 'w', 'e', 'r', 's', 't']
# Text: "low lower lowest"

# Iteration 1: "lo" appears most → merge
# Vocabulary: ['l', 'o', 'w', 'e', 'r', 's', 't', 'lo']

# Iteration 2: "low" appears most → merge
# Vocabulary: [..., 'lo', 'low']

# Continue until desired vocabulary size
```

### Interview Tips

- Emphasize that subword tokenization is the key innovation
- BPE was originally a compression algorithm, adapted for NLP
- Mention that this allows handling of typos, neologisms, and code

---

## Code Demo

See `tokenization_demo.py` for:
- Manual BPE implementation
- Comparing different tokenization strategies
- Visualizing token frequencies

```bash
python interview_questions/01_tokenization_and_text/tokenization_demo.py
```
