# Fine-Tuning Methods

## Interview Questions Covered
- **Q4**: What distinguishes LoRA from QLoRA in fine-tuning LLMs?
- **Q14**: How can LLMs avoid catastrophic forgetting during fine-tuning?
- **Q35**: How does PEFT mitigate catastrophic forgetting?

---

## Q4: LoRA vs QLoRA

### LoRA (Low-Rank Adaptation)

**Core Idea**: Instead of updating all weights, add small trainable matrices.

```
Original: Y = W @ X
LoRA:     Y = W @ X + (A @ B) @ X

Where:
- W: Frozen pretrained weights (d × d)
- A: Trainable (d × r), randomly initialized
- B: Trainable (r × d), initialized to zero
- r: Rank (typically 4-64, much smaller than d)
```

### Why LoRA Works

1. **Parameter efficiency**: Only train r × (d + d) instead of d × d
   - GPT-3: 175B params → LoRA: ~10M trainable params

2. **No inference overhead**: Merge A @ B into W after training
   ```python
   W_merged = W + A @ B  # Same inference cost as original
   ```

3. **Task switching**: Swap LoRA modules for different tasks

### QLoRA (Quantized LoRA)

**QLoRA = LoRA + Quantization**

```
1. Quantize base model to 4-bit (NF4 format)
2. Apply LoRA on top of quantized weights
3. Train LoRA parameters in full precision
4. Gradients flow through dequantized weights
```

### QLoRA Innovations

1. **4-bit NormalFloat (NF4)**: Optimal quantization for normally distributed weights
2. **Double Quantization**: Quantize the quantization constants too
3. **Paged Optimizers**: Handle memory spikes with CPU offloading

### Comparison

| Aspect | LoRA | QLoRA |
|--------|------|-------|
| Base model precision | FP16/32 | 4-bit |
| Memory for 7B model | ~14GB | ~4GB |
| Memory for 70B model | ~140GB | ~35GB |
| Training speed | Faster | Slower (dequantization) |
| Quality | Baseline | ~Same as LoRA |

### Code Example

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.1,
)

model = get_peft_model(base_model, config)
# Trainable params: 0.1% of original
```

---

## Q14: Catastrophic Forgetting

### The Problem

When fine-tuning on new task, model "forgets" original capabilities:

```
Before fine-tuning: Good at everything
After fine-tuning on medical data: Great at medical, terrible at general
```

### Mitigation Strategies

#### 1. Rehearsal (Experience Replay)

Mix old and new data during training:
```python
batch = concat(new_task_samples, replay_buffer_samples)
```

#### 2. Elastic Weight Consolidation (EWC)

Penalize changes to important weights:
```python
loss = task_loss + λ * Σ F_i * (θ_i - θ*_i)²

# F_i: Fisher information (importance of weight i)
# θ*_i: Original weight value
```

#### 3. Progressive Networks

Add new modules, freeze old ones:
```
Original model → [Frozen]
New task → [New trainable layers] + lateral connections to frozen
```

#### 4. Parameter-Efficient Methods (LoRA, Adapters)

Don't modify original weights at all—add small trainable components.

---

## Q35: How PEFT Mitigates Forgetting

### PEFT (Parameter-Efficient Fine-Tuning)

**Key insight**: Freeze most parameters, train only small additions.

### PEFT Methods

| Method | What's Added | Trainable % |
|--------|--------------|-------------|
| **LoRA** | Low-rank matrices | 0.1-1% |
| **Adapters** | Small FFN modules | 1-5% |
| **Prefix Tuning** | Learnable prefixes | <1% |
| **Prompt Tuning** | Soft prompts | <0.1% |
| **IA³** | Learned scaling vectors | <0.01% |

### Why PEFT Prevents Forgetting

1. **Frozen backbone**: Original knowledge preserved exactly
2. **Small capacity**: Can't overfit to new task
3. **Modular**: Can remove adapter to recover original model
4. **Composable**: Stack multiple adapters for multiple tasks

### Adapter Architecture

```python
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck=64):
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)

    def forward(self, x):
        return x + self.up(gelu(self.down(x)))  # Residual
```

Inserted after attention and FFN in each transformer layer.

---

## Interview Tips

1. **Know the math**: LoRA is A @ B with small r
2. **Memory numbers**: QLoRA enables 70B on consumer GPU
3. **Forgetting solutions**: PEFT, rehearsal, EWC
4. **Practical**: LoRA is the most popular method

---

## Code Demo

See `lora_concept_demo.py` for:
- Low-rank matrix approximation visualization
- Simple LoRA implementation
- Parameter count comparison
- Forgetting demonstration

```bash
python interview_questions/11_fine_tuning_methods/lora_concept_demo.py
```
