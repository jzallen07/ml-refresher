"""
Simple RAG (Retrieval-Augmented Generation) Demo
================================================

This demo covers:
- Q36: What is RAG and why is it important for LLMs?
- Q40: How do LLMs handle out-of-distribution data?

RAG Pipeline: RETRIEVE → AUGMENT → GENERATE

Interview Context:
-----------------
RAG solves key LLM limitations:
1. Knowledge cutoff: Can access up-to-date information
2. Hallucinations: Grounds responses in retrieved documents
3. Verifiability: Can cite sources
4. Domain knowledge: Can access specialized information

This demo shows:
1. Document chunking strategies (fixed-size vs sentence-based)
2. Simple embedding creation (TF-IDF for demo purposes)
3. Vector similarity search (cosine similarity)
4. Query augmentation (adding retrieved context to prompt)
5. OOD handling through retrieval
6. Visualization of embedding space and retrieval scores
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import re
from pathlib import Path


# ============================================================================
# PART 1: KNOWLEDGE BASE
# ============================================================================

# Create a small knowledge base about Machine Learning topics
# This simulates a vector database with documents

KNOWLEDGE_BASE = [
    # Document 1: RAG Basics
    """Retrieval-Augmented Generation (RAG) is a technique that combines
    large language models with external knowledge retrieval. It works by
    first retrieving relevant documents from a knowledge base, then using
    those documents as context for the LLM to generate more accurate responses.
    This helps reduce hallucinations and provides up-to-date information.""",

    # Document 2: Vector Embeddings
    """Vector embeddings are dense numerical representations of text that
    capture semantic meaning. Documents and queries are converted to vectors
    in a high-dimensional space where semantically similar texts have similar
    vectors. This enables efficient similarity search using metrics like
    cosine similarity.""",

    # Document 3: Transformers
    """Transformer models use self-attention mechanisms to process sequences.
    They consist of encoder and decoder blocks with multi-head attention layers.
    The attention mechanism allows the model to focus on relevant parts of the
    input sequence. BERT uses only encoders, GPT uses only decoders.""",

    # Document 4: Fine-tuning
    """Fine-tuning adapts a pre-trained model to a specific task by continuing
    training on task-specific data. Methods include full fine-tuning, LoRA
    (Low-Rank Adaptation), and prompt tuning. Fine-tuning changes model weights
    while RAG keeps weights frozen and retrieves information instead.""",

    # Document 5: Vector Databases
    """Vector databases like Pinecone, Weaviate, and Chroma are optimized for
    storing and searching high-dimensional vectors. They use algorithms like
    HNSW (Hierarchical Navigable Small World) and IVF (Inverted File Index)
    for approximate nearest neighbor search. This enables fast retrieval even
    with millions of documents.""",

    # Document 6: Attention Mechanism
    """The attention mechanism computes weighted combinations of values based
    on query-key similarity. Self-attention allows each position to attend to
    all positions in the sequence. Multi-head attention runs several attention
    operations in parallel, allowing the model to focus on different aspects.""",

    # Document 7: Context Window
    """Context window refers to the maximum sequence length a model can process.
    For example, GPT-3 has a 4k token window, GPT-4 has up to 128k tokens.
    Longer context windows allow processing more information but increase
    computational cost quadratically due to self-attention.""",

    # Document 8: Out-of-Distribution Data
    """Out-of-distribution (OOD) data refers to inputs significantly different
    from training data. LLMs struggle with OOD data, often hallucinating or
    providing incorrect answers. RAG helps by retrieving current information.
    Good models should express uncertainty when facing OOD inputs.""",

    # Document 9: Embeddings vs Tokens
    """Tokens are discrete units (words or subwords) while embeddings are
    continuous vector representations. Tokenization splits text into tokens,
    then each token is mapped to a learned embedding vector. These embeddings
    capture semantic and syntactic properties of the tokens.""",

    # Document 10: Semantic Search
    """Semantic search finds results based on meaning rather than keyword matching.
    Unlike traditional search (BM25, TF-IDF) that matches exact terms, semantic
    search uses embeddings to find conceptually similar content. This enables
    finding "car" when searching for "automobile".""",
]


# ============================================================================
# PART 2: CHUNKING STRATEGIES
# ============================================================================

class DocumentChunker:
    """
    Different chunking strategies for documents.

    Interview Concept:
    ----------------
    Chunk size affects retrieval quality:
    - Too small: Loses context, retrieves irrelevant snippets
    - Too large: Dilutes relevance, wastes context window
    - Typical sizes: 200-512 tokens with 10-20% overlap
    """

    @staticmethod
    def fixed_size_chunking(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
        """
        Split text into fixed-size character chunks with overlap.

        Args:
            text: Input text to chunk
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters between chunks

        Interview Note:
        --------------
        Fixed-size chunking is simple but may split sentences awkwardly.
        Overlap helps maintain context at boundaries.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    @staticmethod
    def sentence_based_chunking(text: str, sentences_per_chunk: int = 3) -> List[str]:
        """
        Split text into chunks of complete sentences.

        Args:
            text: Input text to chunk
            sentences_per_chunk: Number of sentences per chunk

        Interview Note:
        --------------
        Sentence-based chunking preserves semantic boundaries.
        Better for Q&A where complete thoughts matter.
        """
        # Simple sentence splitting (naive - production would use spaCy/nltk)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk = '. '.join(chunk_sentences) + '.'
            chunks.append(chunk)

        return chunks


# ============================================================================
# PART 3: SIMPLE EMBEDDING MODEL
# ============================================================================

class SimpleEmbeddingModel:
    """
    Simple embedding model using TF-IDF for demonstration.

    Interview Context:
    -----------------
    In production, you'd use:
    - OpenAI ada-002 (1536 dims)
    - Sentence-BERT (768 dims)
    - BGE/E5 models (open source)

    TF-IDF works for demo but misses semantic similarity
    (e.g., "car" and "automobile" won't be close)
    """

    def __init__(self, max_features: int = 100):
        """
        Initialize TF-IDF vectorizer.

        Args:
            max_features: Maximum number of features (vocabulary size)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        self.is_fitted = False

    def fit(self, documents: List[str]):
        """
        Fit the vectorizer on documents.

        Interview Note:
        --------------
        This is like building a vocabulary. In production embedding models,
        this is done during pre-training on massive corpora.
        """
        self.vectorizer.fit(documents)
        self.is_fitted = True
        print(f"✓ Embedding model fitted on {len(documents)} documents")
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to embedding vectors.

        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")

        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]


# ============================================================================
# PART 4: VECTOR SIMILARITY SEARCH
# ============================================================================

class VectorDatabase:
    """
    Simple vector database for similarity search.

    Interview Context:
    -----------------
    This simulates vector databases like:
    - Pinecone: Cloud-hosted, auto-scaling
    - Weaviate: Open source, GraphQL API
    - Chroma: Lightweight, embedded
    - FAISS: Facebook's similarity search library

    Key operations:
    1. Index: Store document embeddings
    2. Search: Find k nearest neighbors
    3. Similarity metric: Usually cosine similarity
    """

    def __init__(self, embedding_model: SimpleEmbeddingModel):
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = None

    def index_documents(self, documents: List[str]):
        """
        Index documents by computing and storing their embeddings.

        Interview Note:
        --------------
        In production:
        - Documents are chunked first
        - Metadata is stored (source, page, timestamp)
        - Embeddings are stored in optimized data structures (HNSW graphs)
        - Can handle millions/billions of vectors
        """
        print(f"\n{'='*70}")
        print("STEP 1: INDEXING DOCUMENTS")
        print(f"{'='*70}")

        self.documents = documents
        self.embeddings = self.embedding_model.embed(documents)

        print(f"✓ Indexed {len(documents)} documents")
        print(f"✓ Embedding dimension: {self.embeddings.shape[1]}")
        print(f"✓ Total storage: {self.embeddings.nbytes / 1024:.2f} KB")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Search for most similar documents to query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of (doc_index, similarity_score, document_text)

        Interview Note:
        --------------
        Cosine similarity is most common:
        - sim(A, B) = A·B / (||A|| ||B||)
        - Range: [-1, 1], higher = more similar
        - Efficient to compute
        - Normalized (insensitive to vector magnitude)
        """
        print(f"\n{'='*70}")
        print("STEP 2: RETRIEVAL - Searching for relevant documents")
        print(f"{'='*70}")
        print(f"Query: \"{query}\"")

        # Embed the query
        query_embedding = self.embedding_model.embed_single(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarity with all documents
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        print(f"\nTop {top_k} most relevant documents:")
        print("-" * 70)

        for rank, idx in enumerate(top_indices, 1):
            score = similarities[idx]
            doc = self.documents[idx]
            results.append((idx, score, doc))

            print(f"\nRank {rank}: Document {idx}")
            print(f"Similarity Score: {score:.4f}")
            print(f"Preview: {doc[:150]}...")

        return results


# ============================================================================
# PART 5: RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """
    Complete RAG pipeline: Retrieve → Augment → Generate

    Interview Explanation:
    ---------------------
    RAG = Retrieval-Augmented Generation

    Traditional LLM: Query → LLM → Response
    RAG Pipeline: Query → Retrieve Docs → Augment Prompt → LLM → Response

    Benefits:
    - Reduces hallucinations (grounds in retrieved facts)
    - Enables up-to-date information (update DB, not model)
    - Provides citations (can show sources)
    - Domain expertise (add specialized documents)
    """

    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Step 1: RETRIEVE relevant documents.
        """
        results = self.vector_db.search(query, top_k=top_k)
        return [doc for _, _, doc in results]

    def augment(self, query: str, documents: List[str]) -> str:
        """
        Step 2: AUGMENT the query with retrieved context.

        Interview Note:
        --------------
        Prompt engineering is crucial:
        - Clear instruction to use only provided context
        - Explicit instruction to say "I don't know" if info not in context
        - Structured format (Context → Question → Answer)
        """
        # Create augmented prompt with retrieved context
        context = "\n\n".join([f"Document {i+1}: {doc}"
                               for i, doc in enumerate(documents)])

        augmented_prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information to answer, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""

        return augmented_prompt

    def generate(self, augmented_prompt: str) -> str:
        """
        Step 3: GENERATE response (simulated).

        Interview Note:
        --------------
        In production, this would call an LLM API:
        - OpenAI GPT-4
        - Anthropic Claude
        - Local model (Llama, Mistral)

        For this demo, we just show the augmented prompt that would be sent.
        """
        return augmented_prompt

    def query(self, user_query: str, top_k: int = 3) -> Dict:
        """
        Complete RAG pipeline execution.

        Returns:
            Dictionary with retrieved_docs and augmented_prompt
        """
        print(f"\n{'='*70}")
        print("COMPLETE RAG PIPELINE EXECUTION")
        print(f"{'='*70}")
        print(f"User Query: \"{user_query}\"")

        # Step 1: Retrieve
        retrieved_docs = self.retrieve(user_query, top_k=top_k)

        # Step 2: Augment
        print(f"\n{'='*70}")
        print("STEP 3: AUGMENTATION - Creating enhanced prompt")
        print(f"{'='*70}")
        augmented_prompt = self.augment(user_query, retrieved_docs)
        print(f"✓ Added {len(retrieved_docs)} documents as context")
        print(f"✓ Prompt length: {len(augmented_prompt)} characters")

        # Step 3: Generate (simulated)
        print(f"\n{'='*70}")
        print("STEP 4: GENERATION (simulated)")
        print(f"{'='*70}")
        print("In production, this augmented prompt would be sent to an LLM.")
        print("The LLM would generate an answer grounded in the retrieved context.")

        return {
            'query': user_query,
            'retrieved_docs': retrieved_docs,
            'augmented_prompt': augmented_prompt
        }


# ============================================================================
# PART 6: OUT-OF-DISTRIBUTION HANDLING
# ============================================================================

def demonstrate_ood_handling():
    """
    Demonstrate how RAG helps with out-of-distribution (OOD) queries.

    Interview Context - Q40: OOD Data Handling
    ------------------------------------------
    LLMs struggle with:
    1. Topics after training cutoff
    2. Domain-specific knowledge
    3. Novel terms/concepts
    4. Different distributions

    RAG helps by:
    - Retrieving current information
    - Providing domain expertise
    - Reducing reliance on parametric knowledge
    - Enabling graceful degradation
    """
    print(f"\n{'='*70}")
    print("OUT-OF-DISTRIBUTION (OOD) QUERY HANDLING")
    print(f"{'='*70}")

    # Simulate different query distributions
    queries = {
        'in_distribution': "What is RAG and how does it work?",
        'borderline': "How do embeddings relate to semantic search?",
        'out_of_distribution': "Explain quantum computing principles"  # Not in our KB
    }

    print("\nQuery Distribution Analysis:")
    print("-" * 70)

    for query_type, query in queries.items():
        print(f"\n{query_type.upper().replace('_', ' ')}:")
        print(f"Query: {query}")

        # In production, you'd compute embedding distance to training distribution
        # Here we just show the concept
        if query_type == 'out_of_distribution':
            print("❌ Query is OOD (far from training distribution)")
            print("   → RAG will still retrieve best matches")
            print("   → System should express uncertainty")
            print("   → Alternative: Web search tool, or return 'I don't know'")
        else:
            print("✓ Query is in/near training distribution")
            print("  → RAG will find relevant documents")
            print("  → High confidence response expected")


# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

def visualize_embedding_space(
    vector_db: VectorDatabase,
    query: str,
    output_path: str
):
    """
    Visualize document and query embeddings in 2D space.

    Interview Visual Aid:
    --------------------
    Shows how:
    - Documents cluster by topic
    - Query embedding is close to relevant documents
    - Similarity = geometric proximity in embedding space
    """
    # Get all embeddings
    all_embeddings = vector_db.embeddings
    query_embedding = vector_db.embedding_model.embed_single(query).reshape(1, -1)

    # Combine for PCA
    combined = np.vstack([all_embeddings, query_embedding])

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(combined)

    # Split back
    docs_2d = embeddings_2d[:-1]
    query_2d = embeddings_2d[-1]

    # Compute similarities for coloring
    query_embedding_orig = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding_orig, all_embeddings)[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot documents with color based on similarity
    scatter = ax.scatter(
        docs_2d[:, 0],
        docs_2d[:, 1],
        c=similarities,
        cmap='RdYlGn',
        s=200,
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5,
        label='Documents'
    )

    # Add document labels
    for i, (x, y) in enumerate(docs_2d):
        ax.annotate(
            f'Doc {i}',
            (x, y),
            fontsize=9,
            ha='center',
            va='center',
            fontweight='bold'
        )

    # Plot query
    ax.scatter(
        query_2d[0],
        query_2d[1],
        c='red',
        s=500,
        marker='*',
        edgecolors='darkred',
        linewidth=2,
        label='Query',
        zorder=5
    )

    ax.annotate(
        'Query',
        (query_2d[0], query_2d[1]),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=12,
        fontweight='bold',
        color='darkred',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cosine Similarity to Query', fontsize=12, fontweight='bold')

    # Styling
    ax.set_xlabel('PCA Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PCA Component 2', fontsize=12, fontweight='bold')
    ax.set_title(
        'RAG Embedding Space Visualization\n'
        'Documents colored by similarity to query',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add explained variance
    var_explained = pca.explained_variance_ratio_
    ax.text(
        0.02, 0.98,
        f'PCA Variance Explained:\n'
        f'PC1: {var_explained[0]:.1%}\n'
        f'PC2: {var_explained[1]:.1%}',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved embedding space visualization to {output_path}")
    plt.close()


def visualize_retrieval_scores(
    vector_db: VectorDatabase,
    query: str,
    output_path: str
):
    """
    Visualize similarity scores between query and all documents.

    Interview Visual Aid:
    --------------------
    Shows:
    - Which documents are most relevant (highest scores)
    - Score distribution (how selective is retrieval)
    - Top-k cutoff visualization
    """
    # Compute similarities
    query_embedding = vector_db.embedding_model.embed_single(query).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, vector_db.embeddings)[0]

    # Sort for visualization
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_similarities = similarities[sorted_indices]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Bar chart of all scores
    colors = ['green' if i < 3 else 'lightblue' for i in range(len(similarities))]
    bars = ax1.bar(
        range(len(sorted_similarities)),
        sorted_similarities,
        color=colors,
        edgecolor='black',
        linewidth=1.5
    )

    # Highlight top-3
    for i in range(min(3, len(bars))):
        bars[i].set_label('Top-3 Retrieved' if i == 0 else '')

    ax1.axhline(
        y=sorted_similarities[2] if len(sorted_similarities) > 2 else 0,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Top-k Cutoff (k=3)'
    )

    ax1.set_xlabel('Document Index (sorted by score)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cosine Similarity Score', fontsize=12, fontweight='bold')
    ax1.set_title(
        'Document Relevance Scores\n'
        f'Query: "{query[:40]}..."',
        fontsize=13,
        fontweight='bold'
    )
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, max(sorted_similarities) * 1.1])

    # Right plot: Top documents with detailed scores
    top_k = 5
    top_indices = sorted_indices[:top_k]
    top_scores = sorted_similarities[:top_k]

    y_pos = np.arange(top_k)
    bars2 = ax2.barh(y_pos, top_scores, color='green', alpha=0.6, edgecolor='black')

    # Add score labels
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        ax2.text(
            score + 0.01,
            i,
            f'{score:.4f}',
            va='center',
            fontweight='bold'
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'Doc {idx}' for idx in top_indices])
    ax2.invert_yaxis()
    ax2.set_xlabel('Cosine Similarity Score', fontsize=12, fontweight='bold')
    ax2.set_title(
        f'Top-{top_k} Retrieved Documents\n'
        'These would be used as context',
        fontsize=13,
        fontweight='bold'
    )
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved retrieval scores visualization to {output_path}")
    plt.close()


def compare_chunking_strategies():
    """
    Compare different chunking strategies and their effects.

    Interview Insight:
    -----------------
    Chunking affects retrieval quality:
    - Fixed-size: May split concepts, simple to implement
    - Sentence-based: Preserves meaning, variable sizes
    - Semantic: Best quality, requires more processing
    """
    print(f"\n{'='*70}")
    print("CHUNKING STRATEGY COMPARISON")
    print(f"{'='*70}")

    sample_text = KNOWLEDGE_BASE[0]  # Use first document

    chunker = DocumentChunker()

    # Strategy 1: Fixed-size
    fixed_chunks = chunker.fixed_size_chunking(sample_text, chunk_size=100, overlap=20)
    print(f"\nSTRATEGY 1: Fixed-Size Chunking (100 chars, 20 overlap)")
    print(f"Number of chunks: {len(fixed_chunks)}")
    print(f"Average chunk size: {np.mean([len(c) for c in fixed_chunks]):.1f} chars")
    print(f"Chunk size std dev: {np.std([len(c) for c in fixed_chunks]):.1f}")
    print(f"First chunk preview: {fixed_chunks[0][:80]}...")

    # Strategy 2: Sentence-based
    sentence_chunks = chunker.sentence_based_chunking(sample_text, sentences_per_chunk=2)
    print(f"\nSTRATEGY 2: Sentence-Based Chunking (2 sentences/chunk)")
    print(f"Number of chunks: {len(sentence_chunks)}")
    print(f"Average chunk size: {np.mean([len(c) for c in sentence_chunks]):.1f} chars")
    print(f"Chunk size std dev: {np.std([len(c) for c in sentence_chunks]):.1f}")
    print(f"First chunk preview: {sentence_chunks[0][:80]}...")

    print(f"\n{'='*70}")
    print("INTERVIEW INSIGHT:")
    print(f"{'='*70}")
    print("Fixed-size: Predictable, may split sentences (BAD for meaning)")
    print("Sentence-based: Preserves semantics (BETTER for Q&A)")
    print("Production: Use semantic chunking with embeddings or LangChain")


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """
    Main demonstration of RAG pipeline.
    """
    print("="*70)
    print("SIMPLE RAG (RETRIEVAL-AUGMENTED GENERATION) DEMONSTRATION")
    print("="*70)
    print("\nInterview Questions Covered:")
    print("- Q36: What is RAG and why is it important?")
    print("- Q40: How do LLMs handle out-of-distribution data?")
    print()
    print("Pipeline: RETRIEVE → AUGMENT → GENERATE")
    print("="*70)

    # Setup output directory
    output_dir = Path("/Users/zack/dev/ml-refresher/data/interview_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. COMPARE CHUNKING STRATEGIES
    # ========================================================================
    compare_chunking_strategies()

    # ========================================================================
    # 2. CREATE AND INDEX KNOWLEDGE BASE
    # ========================================================================
    print(f"\n{'='*70}")
    print("BUILDING RAG SYSTEM")
    print(f"{'='*70}")

    # Initialize embedding model
    embedding_model = SimpleEmbeddingModel(max_features=100)
    embedding_model.fit(KNOWLEDGE_BASE)

    # Create vector database and index documents
    vector_db = VectorDatabase(embedding_model)
    vector_db.index_documents(KNOWLEDGE_BASE)

    # ========================================================================
    # 3. CREATE RAG PIPELINE
    # ========================================================================
    rag = RAGPipeline(vector_db)

    # ========================================================================
    # 4. TEST RAG WITH DIFFERENT QUERIES
    # ========================================================================
    queries = [
        "How does RAG help reduce hallucinations?",
        "What are vector databases used for?",
        "Explain the attention mechanism in transformers"
    ]

    print(f"\n{'='*70}")
    print("TESTING RAG PIPELINE WITH MULTIPLE QUERIES")
    print(f"{'='*70}")

    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#'*70}")
        print(f"QUERY {i}/{len(queries)}")
        print(f"{'#'*70}")

        result = rag.query(query, top_k=3)

        print(f"\n{'='*70}")
        print("AUGMENTED PROMPT (would be sent to LLM):")
        print(f"{'='*70}")
        print(result['augmented_prompt'][:500] + "...")

        # Visualizations for first query
        if i == 1:
            print(f"\n{'='*70}")
            print("CREATING VISUALIZATIONS")
            print(f"{'='*70}")

            # Embedding space visualization
            viz_path_1 = output_dir / "11_rag_embedding_space.png"
            visualize_embedding_space(vector_db, query, str(viz_path_1))

            # Retrieval scores visualization
            viz_path_2 = output_dir / "12_rag_retrieval_scores.png"
            visualize_retrieval_scores(vector_db, query, str(viz_path_2))

    # ========================================================================
    # 5. DEMONSTRATE OOD HANDLING
    # ========================================================================
    demonstrate_ood_handling()

    # ========================================================================
    # 6. SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print("RAG DEMO COMPLETE - KEY TAKEAWAYS")
    print(f"{'='*70}")

    print("""
Interview Key Points:
--------------------

1. RAG PIPELINE (Q36):
   ✓ Retrieve: Find relevant docs using vector similarity
   ✓ Augment: Add docs to prompt as context
   ✓ Generate: LLM generates grounded response

2. WHY RAG?
   ✓ Reduces hallucinations (grounds in facts)
   ✓ Enables up-to-date info (update DB, not model)
   ✓ Provides citations (can show sources)
   ✓ Domain expertise (add specialized docs)

3. KEY COMPONENTS:
   ✓ Embedding model (TF-IDF, BERT, OpenAI)
   ✓ Vector database (Pinecone, Chroma, FAISS)
   ✓ Similarity search (cosine similarity)
   ✓ Prompt engineering (clear instructions)

4. OOD HANDLING (Q40):
   ✓ RAG helps with OOD by retrieving current info
   ✓ Reduces reliance on parametric knowledge
   ✓ Should express uncertainty when no relevant docs
   ✓ Alternative: Web search tools, API calls

5. CHUNKING STRATEGIES:
   ✓ Fixed-size: Simple, may split concepts
   ✓ Sentence-based: Preserves meaning
   ✓ Semantic: Best quality, more complex
   ✓ Typical: 200-512 tokens, 10-20% overlap

6. RAG vs FINE-TUNING:
   ✓ RAG: Facts, citations, easy updates
   ✓ Fine-tuning: Behavior, style, task adaptation
   ✓ Often used together!

7. PRODUCTION CONSIDERATIONS:
   ✓ Embedding quality matters most
   ✓ Hybrid retrieval (dense + sparse)
   ✓ Reranking for precision
   ✓ Query rewriting/expansion
   ✓ Metadata filtering
    """)

    print(f"{'='*70}")
    print("Visualizations saved to:")
    print(f"  - {output_dir / '11_rag_embedding_space.png'}")
    print(f"  - {output_dir / '12_rag_retrieval_scores.png'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
