# Text Generation Strategies

## Interview Questions Covered
- **Q5**: How does beam search improve text generation compared to greedy decoding?
- **Q6**: What role does temperature play in controlling LLM output?
- **Q12**: How do top-k and top-p sampling differ in text generation?

---

## Q5: Beam Search vs Greedy Decoding

### Greedy Decoding

**Definition**: Always select the token with the highest probability at each step.

```python
# Greedy: Always pick argmax
next_token = argmax(probabilities)
```

**Problems**:
- Locally optimal ≠ globally optimal
- Can miss better sequences that start with lower-probability tokens
- Often produces repetitive, generic text

**Example**:
```
Greedy path: "The" → "dog" → "is" → "a" → "dog" (repetitive!)
Better path: "The" → "golden" → "retriever" → "bounded" → "joyfully"
```

### Beam Search

**Definition**: Keep track of top-k candidate sequences (beams) at each step.

```python
# Beam search with k=3
# Step 1: Top 3 first tokens
# Step 2: Expand each beam, keep top 3 overall
# Continue until EOS or max length
```

**Advantages**:
- Explores multiple hypotheses
- Finds higher-probability complete sequences
- Better for deterministic tasks (translation)

**Beam Size Tradeoffs**:
| Beam Size | Quality | Diversity | Speed |
|-----------|---------|-----------|-------|
| 1 (greedy) | Low | Low | Fast |
| 4-5 | Good | Medium | Medium |
| 10+ | Diminishing returns | Lower | Slow |

---

## Q6: Temperature in LLM Output

### What Temperature Does

Temperature (τ) scales the logits before softmax:

```python
probabilities = softmax(logits / temperature)
```

### Effect on Distribution

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| τ → 0 | Approaches argmax (deterministic) | Factual Q&A |
| τ = 0.3 | Conservative, predictable | Code generation |
| τ = 0.7-0.8 | Balanced | General chat |
| τ = 1.0 | Original distribution | Default |
| τ = 1.5+ | More random, creative | Brainstorming |

### Mathematical Intuition

```python
# Low temperature (0.3): Sharpens distribution
logits = [2.0, 1.0, 0.5]
# softmax(logits/0.3) ≈ [0.91, 0.07, 0.02]  # First token dominates

# High temperature (2.0): Flattens distribution
# softmax(logits/2.0) ≈ [0.42, 0.33, 0.25]  # More uniform
```

### Interview Tip

Temperature doesn't change which token is most likely—it changes HOW MUCH more likely. At τ=0, you always get the top token. At τ=∞, you get uniform random.

---

## Q12: Top-k vs Top-p (Nucleus) Sampling

### Top-k Sampling

**Definition**: Sample only from the k most probable tokens.

```python
# Top-k with k=50
top_k_tokens = sorted_by_probability[:50]
next_token = sample(top_k_tokens)
```

**Problem**: Fixed k doesn't adapt to confidence
- When model is confident: k=50 includes garbage tokens
- When model is uncertain: k=50 might exclude valid options

### Top-p (Nucleus) Sampling

**Definition**: Sample from smallest set of tokens whose cumulative probability exceeds p.

```python
# Top-p with p=0.9
cumsum = 0
nucleus = []
for token in sorted_by_probability:
    nucleus.append(token)
    cumsum += token.probability
    if cumsum >= 0.9:
        break
next_token = sample(nucleus)
```

**Advantage**: Adapts to model confidence
- Confident prediction: nucleus is small (few tokens)
- Uncertain prediction: nucleus is large (many options)

### Comparison

| Aspect | Top-k | Top-p |
|--------|-------|-------|
| Fixed parameter | k tokens | p probability mass |
| Adapts to confidence | No | Yes |
| Typical values | k=40-100 | p=0.9-0.95 |
| Risk | May include/exclude too many | More consistent |

### Combined Usage

In practice, both are often used together:
```python
# OpenAI API example
response = openai.chat.completions.create(
    model="gpt-4",
    temperature=0.7,
    top_p=0.9,      # Nucleus sampling
    # top_k not exposed in API but used internally
)
```

---

## Summary: When to Use What

| Task | Temperature | Sampling |
|------|-------------|----------|
| Code generation | 0.0-0.3 | Greedy or beam |
| Translation | 0.0 | Beam search (k=4-5) |
| Creative writing | 0.8-1.2 | Top-p (0.9) |
| Chatbot | 0.7 | Top-p (0.9) |
| Brainstorming | 1.0-1.5 | Top-p (0.95) |

---

## Code Demo

See `sampling_strategies.py` for:
- Implementation of greedy, beam, top-k, top-p
- Visualization of temperature effects
- Side-by-side comparison of outputs

```bash
python interview_questions/05_text_generation/sampling_strategies.py
```
