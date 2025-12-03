# Seq2Seq Models

## Interview Questions Covered
- **Q8**: What are sequence-to-sequence models, and where are they applied?

---

## Q8: What are sequence-to-sequence models?

### Definition

**Sequence-to-sequence (Seq2Seq)** models transform an input sequence into an output sequence, potentially of different length. They consist of two main components:

1. **Encoder**: Processes input sequence into a representation
2. **Decoder**: Generates output sequence from that representation

### Architecture

```
Input: "How are you?"
        ↓
    [Encoder]
        ↓
  Context Vector(s)
        ↓
    [Decoder]
        ↓
Output: "Comment allez-vous?"
```

### Original RNN-based Seq2Seq

```python
# Encoder: Process input, produce final hidden state
for token in input_sequence:
    hidden = encoder_rnn(token, hidden)
context = hidden  # Final hidden state = "context vector"

# Decoder: Generate output token by token
output_sequence = []
for _ in range(max_length):
    output, hidden = decoder_rnn(prev_token, hidden, context)
    output_sequence.append(output)
```

### The Bottleneck Problem

Original Seq2Seq compressed the entire input into a single fixed-size vector. This caused:
- Information loss for long sequences
- Equal treatment of all input tokens
- Poor performance on long inputs

**Solution**: Attention mechanism (Bahdanau, 2014)
- Decoder attends to all encoder hidden states
- Weighted combination based on relevance
- No fixed bottleneck

### Transformer Seq2Seq

Modern Seq2Seq uses transformers:

```
Encoder:
- Self-attention over input
- Bidirectional (sees full input)
- Produces contextualized representations

Decoder:
- Masked self-attention (causal)
- Cross-attention to encoder outputs
- Generates autoregressively
```

### Applications

| Application | Input → Output |
|-------------|----------------|
| **Machine Translation** | English → French |
| **Summarization** | Long document → Summary |
| **Question Answering** | Context + Question → Answer |
| **Chatbots** | User message → Response |
| **Code Generation** | Description → Code |
| **Speech Recognition** | Audio → Text |
| **Text-to-Speech** | Text → Audio |

### Seq2Seq Model Examples

| Model | Architecture | Use Case |
|-------|-------------|----------|
| T5 | Encoder-Decoder Transformer | General text-to-text |
| BART | Encoder-Decoder Transformer | Summarization |
| mBART | Multilingual BART | Translation |
| Whisper | Encoder-Decoder | Speech-to-text |
| NLLB | Encoder-Decoder | 200 languages |

### Encoder-Only vs Decoder-Only vs Encoder-Decoder

```
Encoder-Only (BERT):
  Input → [Encoder] → Representations
  Best for: Classification, NER, embeddings

Decoder-Only (GPT):
  Input → [Decoder] → Next tokens
  Best for: Open-ended generation, chat

Encoder-Decoder (T5):
  Input → [Encoder] → [Decoder] → Output
  Best for: Structured transformations (translation, summarization)
```

### Interview Tips

1. **Know the components**: Encoder processes, decoder generates
2. **Explain the bottleneck**: Why attention was needed
3. **Modern context**: Transformers replaced RNNs
4. **Name applications**: Translation is the classic example

---

## No Code Demo

This is primarily a conceptual/architectural topic. Key implementations are covered in:
- `02_attention_mechanisms/` - Cross-attention
- `03_transformer_architecture/` - Encoder/decoder structure
