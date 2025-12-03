# Generative vs Discriminative Models

## Interview Questions Covered
- **Q19**: What are the differences between generative and discriminative models in AI?
- **Q39**: What are the differences between autoregressive and autoencoding models in LLMs?

---

## Q19: Generative vs Discriminative Models

### Discriminative Models

**Learn**: P(Y | X) — the boundary between classes

**Goal**: Predict label given input

```
Input: "This movie is great!"
Output: P(positive | text) = 0.95
```

**Examples**:
- Logistic Regression
- SVM
- BERT (for classification)
- Traditional classifiers

### Generative Models

**Learn**: P(X) or P(X, Y) — the full data distribution

**Goal**: Generate new samples that look like training data

```
Input: "Once upon a"
Output: "time, there lived a princess..."
```

**Examples**:
- GPT (autoregressive LM)
- VAEs
- GANs
- Diffusion models

### Key Differences

| Aspect | Discriminative | Generative |
|--------|---------------|------------|
| **Models** | P(Y\|X) | P(X) or P(X,Y) |
| **Task** | Classification | Generation |
| **Training** | Labeled data | Often unsupervised |
| **Data needs** | Less data | More data |
| **Interpretability** | Decision boundary | Data distribution |
| **Flexibility** | One task | Multiple tasks via prompting |

### Why Generative Models Dominate Modern AI

1. **Scale**: Can train on unlimited unlabeled text
2. **Flexibility**: One model, many tasks (via prompting)
3. **Emergence**: New capabilities appear at scale
4. **Transfer**: Pre-training transfers to downstream tasks

### Mathematical Perspective

**Discriminative** (direct mapping):
```
f: X → Y
P(Y | X) = softmax(f(X))
```

**Generative** (model the joint):
```
P(X, Y) = P(X | Y) * P(Y)
P(Y | X) = P(X | Y) * P(Y) / P(X)  # Bayes rule
```

Or for language models:
```
P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ | x₁, ..., xᵢ₋₁)
```

---

## Q39: Autoregressive vs Autoencoding Models

### Autoregressive Models (AR)

**Pattern**: Predict next token given previous tokens

```
P(x₁, x₂, ..., xₙ) = P(x₁) * P(x₂|x₁) * P(x₃|x₁,x₂) * ...
```

**Examples**: GPT, LLaMA, Claude

**Architecture**: Decoder-only with causal (left-to-right) attention

```
Input:  "The cat sat on the"
Target: "cat sat on the mat"

Attention mask (causal):
    The cat sat on the
The  ✓   ✗   ✗  ✗  ✗
cat  ✓   ✓   ✗  ✗  ✗
sat  ✓   ✓   ✓  ✗  ✗
on   ✓   ✓   ✓  ✓  ✗
the  ✓   ✓   ✓  ✓  ✓
```

**Strengths**:
- Natural for generation
- Simple training objective
- Scales well

**Weaknesses**:
- Unidirectional context (can't see future)
- Can't fill in blanks naturally

### Autoencoding Models (AE)

**Pattern**: Reconstruct corrupted input

```
Input:  "The [MASK] sat on the [MASK]"
Output: "The cat sat on the mat"
```

**Examples**: BERT, RoBERTa, ALBERT

**Architecture**: Encoder-only with bidirectional attention

```
Attention mask (bidirectional):
    The [M] sat on the [M]
The  ✓   ✓   ✓  ✓  ✓   ✓
[M]  ✓   ✓   ✓  ✓  ✓   ✓
sat  ✓   ✓   ✓  ✓  ✓   ✓
on   ✓   ✓   ✓  ✓  ✓   ✓
the  ✓   ✓   ✓  ✓  ✓   ✓
[M]  ✓   ✓   ✓  ✓  ✓   ✓
```

**Strengths**:
- Bidirectional context
- Better for understanding tasks
- Good for classification, NER, QA

**Weaknesses**:
- Can't generate naturally
- Requires [MASK] token mismatch with inference

### Comparison Table

| Aspect | Autoregressive (GPT) | Autoencoding (BERT) |
|--------|---------------------|---------------------|
| **Direction** | Left-to-right | Bidirectional |
| **Training** | Next token prediction | Masked token prediction |
| **Attention** | Causal mask | Full attention |
| **Generation** | Natural | Awkward |
| **Understanding** | Good | Better |
| **Popular use** | Chatbots, text gen | Classification, NER |

### Hybrid: Encoder-Decoder (T5, BART)

Combines both:
```
Encoder: Bidirectional (like BERT)
Decoder: Autoregressive (like GPT)

Input (encoder):  "Translate: Hello world"
Output (decoder): "Bonjour monde"
```

### Why Autoregressive Won

1. **Simpler objective**: Next token prediction is elegant
2. **Scaling**: CLM scales better than MLM empirically
3. **Generation**: AR is natural for chat/completion
4. **Emergent abilities**: Appear more in AR models at scale

### Modern Understanding

The field has converged on **decoder-only autoregressive** models because:
- Training objective aligns with use case (generation)
- Simpler architecture scales better
- Instruction tuning makes them good at "understanding" too

---

## Interview Tips

1. **Generative vs Discriminative**: P(X) vs P(Y|X)
2. **Why generative dominates**: Scale, flexibility, emergence
3. **AR vs AE**: Left-to-right vs bidirectional
4. **GPT = AR, BERT = AE**: Know the difference
5. **Why AR won**: Simpler scaling, natural generation

---

## No Code Demo

This is primarily a conceptual topic about model paradigms. Related code is in:
- `02_attention_mechanisms/` - Causal vs bidirectional attention
- `06_training_objectives/` - MLM vs CLM training
