# Zero & Few-Shot Learning

## Interview Questions Covered
- **Q41**: What is zero-shot learning, and how do LLMs implement it?
- **Q44**: What is few-shot learning, and what are its benefits?

---

## Q41: Zero-Shot Learning

### Definition

**Zero-shot learning** allows models to perform tasks they weren't explicitly trained on, using only natural language instructions—no examples provided.

### How It Works

```
Prompt: "Classify the following review as positive or negative:
'This movie was absolutely terrible, I want my money back.'
Classification:"

Model output: "negative"
```

The model was never trained on this specific classification task, but understands:
1. What "classify" means
2. What "positive/negative" sentiment is
3. How to apply these concepts to the review

### Why LLMs Can Do Zero-Shot

1. **Massive pretraining**: Seen similar patterns in training data
2. **Instruction following**: Learned from diverse task descriptions
3. **Language understanding**: Grasps semantic meaning of task
4. **Emergent capability**: Appears at scale (>10B parameters)

### Zero-Shot Examples

| Task | Zero-Shot Prompt |
|------|------------------|
| Sentiment | "Is this review positive or negative?" |
| Translation | "Translate to French:" |
| Summarization | "Summarize in one sentence:" |
| Classification | "Categorize as sports/politics/tech:" |
| Question Answering | "Answer based on the context:" |

### Limitations

- Performance varies by task complexity
- May not follow exact output format
- Can hallucinate or give wrong answers confidently
- Worse than fine-tuned models for specific tasks

---

## Q44: Few-Shot Learning

### Definition

**Few-shot learning** (or in-context learning) provides a small number of examples in the prompt to guide the model's behavior.

### Format

```
Classify the sentiment:
Review: "Great product, highly recommend!" → Positive
Review: "Waste of money, broke after one day" → Negative
Review: "It's okay, nothing special" → Neutral
Review: "Best purchase I've ever made!" → ?
```

### Why Few-Shot Works

1. **Pattern recognition**: Model identifies input-output mapping
2. **Format specification**: Examples show desired output format
3. **Task disambiguation**: Clarifies ambiguous instructions
4. **No gradient updates**: All learning happens in forward pass

### How Many Shots?

| Shots | Name | Typical Use |
|-------|------|-------------|
| 0 | Zero-shot | Simple, well-defined tasks |
| 1 | One-shot | Need format example |
| 3-5 | Few-shot | Standard approach |
| 10-20 | Many-shot | Complex tasks, if context allows |

### Benefits of Few-Shot

1. **No training required**: Instant adaptation
2. **Data efficient**: Only need a handful of examples
3. **Flexible**: Easy to modify or correct
4. **Fast iteration**: Change examples, get different behavior

### Example Selection Matters

```python
# Bad: Similar examples
"Happy!" → Positive
"So happy!" → Positive
"Very happy!" → Positive

# Good: Diverse examples covering edge cases
"Amazing product!" → Positive
"Total garbage" → Negative
"Works as expected" → Neutral
"Not bad, not great" → Neutral
```

### Few-Shot vs Fine-Tuning

| Aspect | Few-Shot | Fine-Tuning |
|--------|----------|-------------|
| Data needed | 3-20 examples | 100s-1000s |
| Training time | None | Hours-days |
| Cost | Inference only | Training + inference |
| Flexibility | Change anytime | Requires retraining |
| Performance | Good | Often better |
| Generalization | May fail on edge cases | Better coverage |

### Best Practices

1. **Diverse examples**: Cover different cases
2. **Consistent format**: Same structure for all examples
3. **Representative**: Examples should match test distribution
4. **Order can matter**: Some models sensitive to example order
5. **Balance**: Equal representation of classes

---

## In-Context Learning Theory

Why does few-shot work without weight updates?

### Hypotheses

1. **Task recognition**: Examples help identify the task from pretraining
2. **Implicit fine-tuning**: Attention mechanisms act like gradient descent
3. **Bayesian inference**: Model infers task from prior over tasks
4. **Copying/retrieval**: Model retrieves similar patterns from memory

### Key Finding

Larger models are better at in-context learning:
- GPT-2 (1.5B): Limited few-shot ability
- GPT-3 (175B): Strong few-shot ability
- GPT-4: Even stronger

---

## Interview Tips

1. **Zero-shot**: No examples, just instructions—works due to pretraining
2. **Few-shot**: 3-10 examples in prompt—no weight updates
3. **Key insight**: This is "learning" without training
4. **Limitations**: Context window limits number of examples
5. **Practical**: Few-shot usually better than zero-shot

---

## No Code Demo

This is primarily a conceptual topic about prompting strategies. Related code is in:
- `13_prompt_engineering/` - Prompt templates and CoT
