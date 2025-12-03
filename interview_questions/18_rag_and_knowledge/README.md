# RAG & Knowledge

## Interview Questions Covered
- **Q36**: What is Retrieval-Augmented Generation (RAG), and why is it important for LLMs?
- **Q40**: How do LLMs handle out-of-distribution data?

---

## Q36: Retrieval-Augmented Generation (RAG)

### Definition

**RAG** combines LLMs with external knowledge retrieval to generate more accurate, up-to-date, and verifiable responses.

### Why RAG?

LLMs have limitations:
1. **Knowledge cutoff**: Training data has a date
2. **Hallucinations**: Can make up facts confidently
3. **No citations**: Can't verify claims
4. **Domain knowledge**: May lack specialized info

RAG solves these by retrieving relevant documents before generating.

### RAG Pipeline

```
User Query: "What were Apple's Q3 2024 earnings?"
           ↓
    ┌─────────────────┐
    │   1. RETRIEVE   │ ← Search knowledge base
    └─────────────────┘
           ↓
    Relevant documents:
    - "Apple Q3 2024 report: Revenue $85.8B..."
    - "Tim Cook announced..."
           ↓
    ┌─────────────────┐
    │   2. AUGMENT    │ ← Add docs to prompt
    └─────────────────┘
           ↓
    Prompt: "Context: [documents]
            Question: What were Apple's Q3 2024 earnings?"
           ↓
    ┌─────────────────┐
    │   3. GENERATE   │ ← LLM generates answer
    └─────────────────┘
           ↓
    "According to the Q3 2024 report, Apple's
     revenue was $85.8 billion..."
```

### Retrieval Methods

#### 1. Dense Retrieval (Embeddings)

```python
# Index documents
doc_embeddings = embed_model(documents)  # Store in vector DB

# At query time
query_embedding = embed_model(query)
relevant_docs = vector_db.search(query_embedding, top_k=5)
```

**Vector Databases**: Pinecone, Weaviate, Chroma, FAISS

#### 2. Sparse Retrieval (BM25/TF-IDF)

```python
# Traditional keyword matching
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(query_tokens)
```

#### 3. Hybrid Retrieval

Combine dense + sparse for best results:
```python
final_score = α * dense_score + (1-α) * sparse_score
```

### Embedding Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| OpenAI ada-002 | 1536 | General purpose |
| Cohere embed | 1024/4096 | Multilingual |
| BGE | 768/1024 | Open source |
| E5 | 768/1024 | Open source |

### RAG Prompt Template

```
You are a helpful assistant. Answer based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have that information."

Context:
{retrieved_documents}

Question: {user_query}

Answer:
```

### Advanced RAG Techniques

#### 1. Query Rewriting

```
Original: "Tell me about that company's profits"
Rewritten: "What are Apple Inc's profit margins and revenue for fiscal year 2024?"
```

#### 2. Hypothetical Document Embeddings (HyDE)

```python
# Generate hypothetical answer
hypothetical = llm("Write a passage that answers: {query}")
# Use hypothetical to search (often finds better matches)
docs = search(embed(hypothetical))
```

#### 3. Reranking

```python
# Initial retrieval (fast, recall-focused)
candidates = retriever.get(query, top_k=100)
# Reranking (slow, precision-focused)
reranked = reranker.rank(query, candidates)[:10]
```

#### 4. Chunking Strategies

| Strategy | Description |
|----------|-------------|
| Fixed-size | 512 tokens per chunk |
| Sentence | Split on sentence boundaries |
| Semantic | Split on topic changes |
| Recursive | Hierarchical splitting |

### RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Knowledge update** | Instant (update DB) | Requires retraining |
| **Verifiability** | Can cite sources | No citations |
| **Cost** | Inference + retrieval | Training cost |
| **Customization** | Limited | High |
| **Hallucination** | Reduced | Still possible |

---

## Q40: Out-of-Distribution (OOD) Data

### The Problem

LLMs perform poorly on data very different from training:
- New topics after training cutoff
- Domain-specific jargon
- Different languages/dialects
- Novel formats

### How LLMs Handle OOD

#### 1. Graceful Degradation

Models can often "extrapolate" reasonably:
```
Q: "Explain quantum blockchain AI (made-up term)"
A: "While 'quantum blockchain AI' isn't a standard term,
    it might refer to combining..."
```

#### 2. Uncertainty Expression

Well-trained models express uncertainty:
```
"I'm not certain about events after my training cutoff..."
"I don't have reliable information about..."
```

#### 3. Leveraging Related Knowledge

Transfer from similar concepts:
```
Q: "How does the Zephyr-9 rocket work?"  # Fictional
A: Model uses knowledge of real rockets to give plausible answer
   (but may hallucinate details)
```

### Improving OOD Handling

#### 1. RAG

Retrieve relevant up-to-date information:
```python
if is_recent_topic(query):
    docs = retrieve_from_current_knowledge_base(query)
    return generate_with_context(query, docs)
```

#### 2. Tool Use

Let the model search the web:
```
Thought: I don't know about recent events
Action: WebSearch("latest developments in...")
Observation: [search results]
Answer: Based on recent information...
```

#### 3. Confidence Calibration

Train models to know what they don't know:
```python
# During training, include examples like:
"Q: What happened yesterday? A: I don't have real-time information."
```

#### 4. Domain Adaptation

Fine-tune on domain-specific data:
```python
# Medical domain adaptation
fine_tune(base_model, medical_corpus)
```

### OOD Detection

Identify when input is out-of-distribution:

```python
def is_ood(query_embedding, training_distribution):
    # Check if query is far from training data
    distance = cosine_distance(query_embedding, distribution_centroid)
    return distance > threshold
```

### Best Practices

1. **Prompt for honesty**: "Say 'I don't know' if uncertain"
2. **Use RAG for facts**: Don't rely on parametric knowledge alone
3. **Provide context**: Give the model domain-specific information
4. **Set expectations**: Users should know model limitations

---

## Interview Tips

1. **RAG pipeline**: Retrieve → Augment → Generate
2. **Why RAG**: Reduces hallucination, enables updates, provides citations
3. **Vector search**: Embeddings + similarity search
4. **OOD handling**: RAG, tool use, uncertainty expression
5. **RAG vs fine-tuning**: RAG for factual knowledge, fine-tuning for style/behavior

---

## Code Demo

See `simple_rag_demo.py` for:
- Basic RAG pipeline implementation
- Vector similarity search
- Chunk creation and retrieval
- Query augmentation

```bash
poetry run python interview_questions/18_rag_and_knowledge/simple_rag_demo.py
```
