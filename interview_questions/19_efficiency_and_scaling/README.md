# Efficiency & Scaling

## Interview Questions Covered
- **Q42**: What is speculative decoding, and how does it accelerate LLM inference?

---

## Q42: Speculative Decoding

### The Problem

Autoregressive LLM inference is slow:
- Generate one token at a time
- Each token requires full forward pass
- Memory-bound, not compute-bound
- GPU underutilized

```
Step 1: "The"     → full forward pass → "cat"
Step 2: "The cat" → full forward pass → "sat"
Step 3: "The cat sat" → full forward pass → "on"
...
```

### Speculative Decoding Solution

Use a small, fast **draft model** to guess multiple tokens, then **verify** with the large model in one pass.

```
Draft model (fast):
  "The" → "cat sat on the mat" (5 tokens in 5 fast passes)

Target model (slow):
  Verify all 5 tokens in ONE pass
  Accept: "cat sat on the" ✓
  Reject: "mat" → regenerate from "the"
```

### How It Works

```python
def speculative_decode(prompt, draft_model, target_model, k=5):
    while not done:
        # 1. Draft: Generate k tokens with small model
        draft_tokens = []
        draft_probs = []
        for _ in range(k):
            token, prob = draft_model.generate_next(prompt + draft_tokens)
            draft_tokens.append(token)
            draft_probs.append(prob)

        # 2. Verify: Score all k tokens in ONE forward pass
        target_probs = target_model.get_probs(prompt + draft_tokens)

        # 3. Accept/Reject using rejection sampling
        accepted = 0
        for i in range(k):
            # Accept with probability min(1, target_prob / draft_prob)
            if random() < target_probs[i] / draft_probs[i]:
                accepted += 1
            else:
                break  # Reject this and all following

        # 4. Add accepted tokens + sample correction token
        output.extend(draft_tokens[:accepted])
        if accepted < k:
            # Sample from residual distribution
            correction = sample_residual(target_probs[accepted], draft_probs[accepted])
            output.append(correction)
```

### Acceptance Rate

The acceptance rate depends on how well the draft model approximates the target:

| Alignment | Acceptance Rate | Speedup |
|-----------|-----------------|---------|
| Poor | ~20% | 1-1.5x |
| Good | ~60% | 2-3x |
| Excellent | ~80% | 3-4x |

### Why Verification is Exact

Using rejection sampling guarantees the output distribution is identical to the target model:
- If draft matches target perfectly: 100% accept
- If draft differs: proportionally higher rejection
- Residual sampling handles rejected cases

**Key insight**: No quality loss, only speedup!

### Draft Model Choices

| Approach | Draft Model | Pros/Cons |
|----------|-------------|-----------|
| **Separate model** | Smaller version (1B draft for 70B target) | Need to train/deploy extra model |
| **Early exit** | First N layers of target | Single model, may be less aligned |
| **Quantized** | 4-bit version of target | High alignment, still somewhat slow |
| **N-gram** | Lookup table | Extremely fast, low quality |

### Practical Speedups

| Target Model | Draft Model | Speedup |
|--------------|-------------|---------|
| LLaMA-2-70B | LLaMA-2-7B | 2-2.5x |
| GPT-4 | GPT-3.5 | ~2x (estimated) |
| Chinchilla-70B | Chinchilla-1.4B | 2.5x |

### Other Inference Optimizations

#### 1. KV Cache

Store key-value pairs to avoid recomputation:
```python
# Without cache: O(n²) per token
# With cache: O(n) per token

def generate_with_cache(prompt):
    kv_cache = None
    for _ in range(max_tokens):
        logits, kv_cache = model(last_token, past_kv=kv_cache)
        # Only process new token, reuse cached K,V
```

#### 2. Continuous Batching

Don't wait for all sequences to finish:
```
Batch starts: [seq1, seq2, seq3]
seq2 finishes → [seq1, seq4, seq3]  # Replace immediately
```

#### 3. PagedAttention (vLLM)

Memory-efficient KV cache management:
```
# Instead of contiguous memory per sequence
# Use paged memory like OS virtual memory
# Reduces memory fragmentation
```

#### 4. Quantization

Reduce precision for faster inference:
```python
# FP16 → INT8: ~2x speedup, ~1% quality loss
# FP16 → INT4: ~4x speedup, ~3-5% quality loss
model_int8 = quantize(model, dtype=torch.int8)
```

#### 5. Flash Attention

Memory-efficient attention computation:
```python
# Standard: O(n²) memory for attention matrix
# Flash Attention: O(n) memory via tiling
from flash_attn import flash_attn_func
```

### Scaling Laws

Chinchilla scaling laws:
```
Optimal model size ∝ compute^0.5
Optimal data size ∝ compute^0.5

For compute budget C:
  Optimal N (params) ≈ 0.05 * C^0.5
  Optimal D (tokens) ≈ 20 * N
```

**Implication**: Models were undertrained. Train smaller models on more data.

| Previous | Chinchilla-Optimal |
|----------|-------------------|
| GPT-3 175B on 300B tokens | 70B on 1.4T tokens |
| Gopher 280B on 300B tokens | 70B on 1.4T tokens |

---

## Additional Efficiency Topics

### Tensor Parallelism

Split layers across GPUs:
```
GPU 0: First half of attention heads
GPU 1: Second half of attention heads
→ Combine with all-reduce
```

### Pipeline Parallelism

Split layers across GPUs:
```
GPU 0: Layers 0-31
GPU 1: Layers 32-63
→ Micro-batching for efficiency
```

### Expert Parallelism (MoE)

Different experts on different GPUs:
```
GPU 0: Experts 0-3
GPU 1: Experts 4-7
→ All-to-all communication for routing
```

---

## Interview Tips

1. **Speculative decoding**: Draft → verify → accept/reject
2. **Key insight**: Verification is exact (no quality loss)
3. **Speedup**: 2-3x typical, depends on draft/target alignment
4. **KV cache**: Essential for autoregressive efficiency
5. **Chinchilla**: Smaller models, more data

---

## No Code Demo

Speculative decoding requires two models and is best demonstrated with actual LLM inference. Related concepts are in:
- `05_text_generation/` - Sampling strategies
- For production implementation, see vLLM or HuggingFace TGI
