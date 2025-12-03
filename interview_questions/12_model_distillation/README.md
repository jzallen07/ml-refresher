# Model Distillation

## Interview Questions Covered
- **Q15**: What is model distillation, and how does it benefit LLMs?

---

## Q15: Model Distillation

### Definition

**Knowledge distillation** trains a smaller "student" model to mimic a larger "teacher" model's behavior, using soft probability distributions rather than hard labels.

### The Key Insight

Instead of training on one-hot labels:
```
Label: [0, 0, 1, 0, 0]  # "cat" is correct
```

Train on teacher's soft predictions:
```
Teacher output: [0.01, 0.05, 0.85, 0.05, 0.04]  # "cat" most likely, but "dog" has some probability
```

### Why Soft Labels Help

Soft labels contain "dark knowledge":
- Relationships between classes (cat is more similar to dog than to car)
- Confidence levels (this example is ambiguous)
- Structure in the data

### Distillation Loss

```python
# Standard cross-entropy with hard labels
L_hard = CrossEntropy(student_output, true_labels)

# Distillation loss with soft labels
L_soft = KL_Divergence(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
)

# Combined loss
L = α * L_soft * T² + (1 - α) * L_hard
```

Where T is "temperature" (higher T = softer distributions).

### Benefits

| Benefit | Description |
|---------|-------------|
| **Size reduction** | 10-100x smaller models |
| **Speed** | Faster inference |
| **Deployment** | Run on edge devices, phones |
| **Cost** | Lower serving costs |
| **Maintained quality** | Often 90-99% of teacher performance |

### Famous Examples

| Student | Teacher | Size Reduction |
|---------|---------|----------------|
| DistilBERT | BERT | 40% smaller, 60% faster |
| TinyBERT | BERT | 7.5x smaller |
| MiniLM | Large LM | 2-10x smaller |
| Alpaca | GPT-3.5 | Open-source from API |

### Distillation Strategies

#### 1. Output Distillation
Match final layer probabilities.

#### 2. Feature Distillation
Match intermediate layer representations.
```python
for layer in range(num_layers):
    L += MSE(student_hidden[layer], teacher_hidden[layer])
```

#### 3. Attention Distillation
Match attention patterns.
```python
L += MSE(student_attention, teacher_attention)
```

### Practical Considerations

**Temperature Selection**:
- T=1: Standard softmax
- T=2-5: Softer, more information transfer
- T=20: Very soft, might lose signal

**When to Use**:
- Need smaller model for deployment
- Teacher is too expensive to serve
- Want to create open-source version of proprietary model

**Limitations**:
- Need access to teacher (or its outputs)
- Student has capacity limits
- Some tasks don't transfer well

---

## Self-Distillation

Train a model to be its own teacher:
1. Train model normally
2. Generate soft labels from trained model
3. Retrain same architecture on soft labels

Surprisingly, this often improves performance!

---

## Interview Tips

1. **Core concept**: Soft labels contain more information than hard labels
2. **Temperature**: Higher T = softer distributions
3. **Famous example**: DistilBERT is 40% smaller, 97% performance
4. **Use cases**: Edge deployment, API cost reduction

---

## No Code Demo

This is primarily a conceptual topic. The key mathematical concepts (KL divergence, softmax temperature) are covered in:
- `09_loss_functions_and_math/` - KL divergence
- `05_text_generation/` - Temperature scaling
