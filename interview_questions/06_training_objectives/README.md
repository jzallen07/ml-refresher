# Training Objectives

## Interview Questions Covered
- **Q7**: What is masked language modeling, and how does it aid pretraining?
- **Q9**: How do autoregressive and masked models differ in LLM training?
- **Q11**: What is next sentence prediction, and how does it enhance LLMs?

---

## Q7: Masked Language Modeling (MLM)

### Definition

**MLM** randomly masks tokens in the input and trains the model to predict them from context.

```
Input:  "The cat [MASK] on the [MASK]"
Target: "The cat  sat  on the  mat"
```

### How It Works

1. **Masking**: Replace ~15% of tokens
   - 80%: Replace with [MASK]
   - 10%: Replace with random token
   - 10%: Keep original (prevents model from ignoring non-masked)

2. **Prediction**: Model predicts original token for each masked position

3. **Loss**: Cross-entropy on masked positions only

### Why It Works

- **Bidirectional context**: Uses BOTH left and right context
- **Self-supervised**: No labels needed, learns from raw text
- **Rich representations**: Must understand deep semantics to fill blanks

### Used By
- BERT, RoBERTa, ALBERT, ELECTRA, DeBERTa

### Limitations
- Can't generate text autoregressively
- [MASK] tokens don't exist at inference time (train-test mismatch)

---

## Q9: Autoregressive vs Masked Models

### Autoregressive (Causal) Language Modeling

**Definition**: Predict the next token given all previous tokens.

```
Input:  "The cat sat on the"
Target: "mat"
P(x_t | x_1, x_2, ..., x_{t-1})
```

**Characteristics**:
- Unidirectional (left-to-right only)
- Natural for generation
- No train-test mismatch

**Models**: GPT series, LLaMA, Claude

### Masked Language Modeling

**Definition**: Predict masked tokens from bidirectional context.

```
Input:  "The cat [MASK] on the mat"
Target: "sat"
P(x_mask | x_1, ..., x_{mask-1}, x_{mask+1}, ..., x_n)
```

**Characteristics**:
- Bidirectional context
- Better for understanding tasks
- Can't generate naturally

**Models**: BERT, RoBERTa

### Comparison Table

| Aspect | Autoregressive | Masked |
|--------|---------------|--------|
| Direction | Left-to-right | Bidirectional |
| Context | Past only | Full context |
| Generation | Natural | Requires tricks |
| Understanding | Weaker | Stronger |
| Training signal | Every token | ~15% of tokens |
| Example | GPT-4 | BERT |

### Why GPT Won

Despite MLM's advantages for understanding, autoregressive models dominate because:
1. Natural generation without architectural changes
2. Scale better with more compute
3. Unified architecture for all tasks (via prompting)
4. Emergent abilities at scale

---

## Q11: Next Sentence Prediction (NSP)

### Definition

**NSP** trains the model to predict whether sentence B follows sentence A.

```
Input A: "The cat sat on the mat."
Input B: "It was a sunny day."
Label: NotNext (0)

Input A: "The cat sat on the mat."
Input B: "The mat was very soft."
Label: IsNext (1)
```

### BERT's Training

BERT uses BOTH MLM and NSP:
1. MLM: Learn token-level representations
2. NSP: Learn sentence-level relationships

### How NSP Works

1. 50% of training pairs: B is actual next sentence (IsNext)
2. 50% of training pairs: B is random sentence (NotNext)
3. [CLS] token embedding used for binary classification

### Criticism and Alternatives

**Problems with NSP**:
- Topic prediction is too easy (random sentences often different topics)
- Hurts performance in some cases

**RoBERTa's Finding**: Removing NSP improves results!

**Alternatives**:
- **Sentence Order Prediction (SOP)**: ALBERT uses swapped vs original order
- **No sentence objective**: RoBERTa, just uses MLM

### When NSP Helps

- Tasks requiring sentence relationships
- Natural language inference
- Question answering with context

---

## Summary: Training Objectives Landscape

```
Self-Supervised Pretraining Objectives:

├── Token-Level
│   ├── Masked LM (BERT): Predict [MASK] tokens
│   ├── Causal LM (GPT): Predict next token
│   ├── Span Corruption (T5): Predict masked spans
│   └── Replaced Token Detection (ELECTRA): Real vs fake tokens
│
└── Sentence-Level
    ├── NSP (BERT): Is B next sentence?
    ├── SOP (ALBERT): Is order correct?
    └── Contrastive (SimCSE): Similar sentences close
```

---

## Code Demo

See `training_objectives_demo.py` for:
- Masked language modeling implementation
- Comparison of MLM vs causal LM
- Visualization of masking strategies

```bash
python interview_questions/06_training_objectives/training_objectives_demo.py
```
