# Graph Neural Networks

## Interview Questions Covered
- **Q51**: What are Graph Neural Networks, and how does message passing work?
- **Q52**: How do knowledge graph embeddings represent relational data?
- **Q53**: What is Graph RAG and how does it enhance retrieval?

---

## Q51: What are Graph Neural Networks, and how does message passing work?

### Answer

**Graph Neural Networks (GNNs)** extend deep learning to graph-structured data — social networks, molecules, knowledge graphs, and any domain where relationships between entities matter as much as the entities themselves.

### The Core Idea

Traditional neural networks expect fixed-size inputs (vectors, grids). GNNs handle arbitrary graph structures by learning node representations that capture both node features and neighborhood structure.

Key insight: a node's representation should be informed by its neighbors. "You are the average of the company you keep."

### Message Passing Framework

All modern GNNs follow the **message passing** paradigm:

```
For each layer:
  1. MESSAGE:   m_j→i = MSG(h_j, h_i, e_ji)     # each neighbor j sends a message
  2. AGGREGATE: M_i = AGG({m_j→i : j ∈ N(i)})    # collect all incoming messages
  3. UPDATE:    h_i' = UPD(h_i, M_i)              # update node representation
```

Where:
- `h_i` = node i's feature vector
- `N(i)` = neighbors of node i
- `e_ji` = edge features (optional)

After K layers, each node's representation captures information from its K-hop neighborhood.

### GCN (Graph Convolutional Network)

The simplest GNN — a spectral approach simplified to spatial operations:

```
H' = σ(D̃⁻¹/²ÃD̃⁻¹/²HW)
```

Where:
- `Ã = A + I` (adjacency + self-loops)
- `D̃` = degree matrix of Ã
- `H` = node feature matrix
- `W` = learnable weight matrix
- `σ` = activation (ReLU)

In practice: each node averages its neighbors' features (including itself), then applies a linear transform.

### GAT (Graph Attention Network)

Replaces fixed averaging with **learned attention weights** — different neighbors contribute differently:

```
α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
h_i' = σ(Σ_j α_ij · Wh_j)
```

Key connection: GAT applies the same attention mechanism from transformers to graph neighborhoods. Each node "attends" to its neighbors.

### GraphSAGE

**Sample and Aggregate** — designed for inductive learning on large graphs:

1. Sample a fixed number of neighbors (not all)
2. Aggregate using mean, LSTM, or max pooling
3. Concatenate with self-representation

Advantage: can generalize to unseen nodes (inductive), unlike GCN/GAT which are transductive.

### Interview Tips

- Connect GNNs to transformers: a transformer is a GNN on a fully-connected graph where attention weights are the edge weights
- Over-smoothing: too many GNN layers → all node representations converge (analogous to information bottleneck)
- Common applications: molecule property prediction, recommendation systems, traffic forecasting

---

## Q52: How do knowledge graph embeddings represent relational data?

### Answer

**Knowledge graph embeddings** learn continuous vector representations for entities and relations in a knowledge graph, enabling link prediction, entity classification, and knowledge base completion.

### Knowledge Graph Basics

A knowledge graph stores facts as **(head, relation, tail)** triples:
- (Einstein, born_in, Germany)
- (Germany, located_in, Europe)
- (Einstein, field, Physics)

The goal: embed entities and relations into a vector space where valid triples score higher than invalid ones.

### TransE

The foundational model — relations are translations in embedding space:

```
Scoring: f(h, r, t) = -||h + r - t||

Goal: h + r ≈ t for valid triples
```

Intuition: "Germany" + "capital_of" ≈ "Berlin"

Training uses **contrastive learning**: valid triples should score higher than corrupted triples (where head or tail is replaced with a random entity).

Limitation: TransE cannot model 1-to-N relations well (e.g., "born_in" with multiple people born in the same city).

### Beyond TransE

| Model | Relation Representation | Handles |
|-------|------------------------|---------|
| TransE | Translation vector | 1-to-1 |
| TransR | Translation in relation-specific space | 1-to-N |
| DistMult | Diagonal matrix (bilinear) | Symmetric |
| ComplEx | Complex-valued embeddings | Antisymmetric |
| RotatE | Rotation in complex plane | Composition |

### Interview Tips

- Connect to word embeddings: just as Word2Vec learns "king - man + woman = queen", TransE learns "Germany + capital_of = Berlin"
- Scale matters: real knowledge graphs have millions of entities — embedding dimension is typically 100-400
- Applications: drug discovery (predict protein interactions), recommendation (user-item-attribute graphs), question answering

---

## Q53: What is Graph RAG and how does it enhance retrieval?

### Answer

**Graph RAG** extends standard retrieval-augmented generation by organizing knowledge into a graph structure, enabling multi-hop reasoning and relationship-aware retrieval that flat document stores cannot provide.

### Standard RAG Limitations

Traditional RAG (retrieve chunks → stuff into context) struggles with:
1. **Multi-hop questions**: "What techniques from the field where Einstein worked are used in LLMs?" requires Einstein → Physics → attention mechanisms
2. **Global summarization**: "What are the main themes across all documents?" requires seeing the whole corpus
3. **Relationship queries**: "How does concept A relate to concept B?" requires traversing connections

### Graph RAG Architecture

```
1. INDEXING:
   Documents → Extract entities/relationships → Build knowledge graph

2. COMMUNITY DETECTION:
   Graph → Leiden algorithm → Hierarchical communities → Community summaries

3. RETRIEVAL:
   Query → Identify relevant communities/entities → Traverse graph → Gather context

4. GENERATION:
   Graph context + retrieved text → LLM → Answer with provenance
```

Key components:
- **Entity extraction**: LLM identifies entities and relationships from source documents
- **Community summaries**: pre-computed summaries at different granularity levels
- **Local search**: start from query entities, traverse neighborhood
- **Global search**: use community summaries for broad questions

### When to Use

| Scenario | Standard RAG | Graph RAG |
|----------|-------------|-----------|
| Simple factual lookup | Good | Overkill |
| Multi-hop reasoning | Poor | Strong |
| Relationship discovery | Poor | Strong |
| Global summarization | Poor | Strong |
| Low-latency required | Better | Slower |
| Small corpus (<100 docs) | Sufficient | Unnecessary |

### Interview Tips

- Graph RAG is complementary to vector RAG, not a replacement — hybrid approaches work best
- The indexing cost is significant (many LLM calls to extract entities), so it suits corpora that are queried repeatedly
- Connect to this codebase: the concept graph in ml-refresher is a lightweight version of Graph RAG applied to curriculum navigation
