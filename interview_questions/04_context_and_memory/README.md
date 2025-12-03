# Context & Memory

## Interview Questions Covered
- **Q3**: What is the context window in LLMs, and why does it matter?

---

## Q3: What is the context window in LLMs?

### Answer

The **context window** (or context length) is the maximum number of tokens an LLM can process in a single forward pass. It defines the model's "memory" for understanding and generating text.

### Key Numbers to Know

| Model | Context Window |
|-------|---------------|
| GPT-3 | 4,096 tokens |
| GPT-4 | 8K / 32K / 128K tokens |
| Claude 2 | 100K tokens |
| Claude 3 | 200K tokens |
| Llama 2 | 4,096 tokens |
| Llama 3 | 8K / 128K tokens |

### Why Context Window Matters

1. **Coherence**: Longer context = better understanding of document structure
2. **Few-shot learning**: More examples fit in the prompt
3. **Long documents**: Can process entire books, codebases, conversations
4. **Task complexity**: Multi-step reasoning needs history

### The Quadratic Problem

Standard attention has O(n²) complexity:
- 4K tokens: 16M attention computations
- 32K tokens: 1B attention computations
- 128K tokens: 16B attention computations

This is why longer context windows require architectural innovations.

### Techniques to Extend Context

| Technique | Description |
|-----------|-------------|
| **Sparse Attention** | Only attend to subset of positions |
| **Linear Attention** | Approximate attention with O(n) |
| **Sliding Window** | Local attention + global tokens |
| **ALiBi** | Position bias instead of embeddings |
| **RoPE** | Rotary position embeddings (extrapolate) |
| **Ring Attention** | Distribute across devices |

### Memory vs Context Window

```
Context Window: What the model sees in ONE forward pass
Memory: Information retained across multiple interactions

LLMs have NO persistent memory between API calls!
Each request starts fresh with only the provided context.
```

### Practical Implications

**Token ≠ Word**
- English: ~1.3 tokens per word
- Code: More tokens (special characters)
- "ChatGPT" = 3 tokens: ["Chat", "G", "PT"]

**Context Budget**
```
Total context = System prompt + Conversation history + User query + Response
```

If you exceed the limit, oldest messages are typically truncated.

### Interview Tips

1. **Know the numbers**: GPT-4's context sizes, Claude's 200K
2. **Explain the tradeoff**: Longer context = more compute cost
3. **Mention solutions**: Sparse attention, RAG for unlimited "memory"
4. **Practical awareness**: Token counting, prompt optimization

---

## No Code Demo

This is a conceptual topic. Key points to remember:
- Context window is measured in **tokens**, not words
- Attention complexity is **O(n²)** which limits practical length
- Modern techniques (ALiBi, RoPE, Ring Attention) enable longer contexts
- RAG can provide "infinite" context by retrieving relevant information
