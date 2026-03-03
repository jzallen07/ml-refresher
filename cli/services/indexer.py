from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch.nn.functional as F

from cli.chunking import Chunk, chunk_all_content

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = REPO_ROOT / ".ml-refresher"
RUBRICS_PATH = REPO_ROOT / "cli" / "rubrics" / "questions.json"

MATRYOSHKA_DIM = 256
TABLE_NAME = "chunks"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        kwargs = dict(
            model_name_or_path="nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="cpu",
        )
        try:
            _model = SentenceTransformer(**kwargs, local_files_only=True)
        except OSError:
            # First run: download from HF Hub
            _model = SentenceTransformer(**kwargs)
    return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    model = _get_model()
    prefixed = [f"search_document: {t}" for t in texts]
    embeddings = model.encode(prefixed, convert_to_tensor=True)
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :MATRYOSHKA_DIM]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


def embed_query(query: str) -> np.ndarray:
    model = _get_model()
    prefixed = f"search_query: {query}"
    embedding = model.encode([prefixed], convert_to_tensor=True)
    embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
    embedding = embedding[:, :MATRYOSHKA_DIM]
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy()[0]


def _chunk_to_record(chunk: Chunk, vector: np.ndarray) -> dict:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "enriched_text": chunk.enriched_text,
        "vector": vector.tolist(),
        "parent_id": chunk.parent_id or "",
        "level": chunk.level,
        "source_type": chunk.source_type,
        "file_path": chunk.file_path,
        "has_code": chunk.has_code,
        "content_type": chunk.content_type,
        "category": chunk.category,
        "question_id": chunk.question_id,
        "question_text": chunk.question_text,
        "section": chunk.section,
        "lesson_number": chunk.lesson_number,
        "lesson_title": chunk.lesson_title,
        "difficulty": chunk.difficulty,
        "function_name": chunk.function_name,
    }


def _extract_rubrics(chunks: list[Chunk]) -> list[dict]:
    """Extract structured rubrics from interview question parent chunks."""
    rubrics = []
    for chunk in chunks:
        if chunk.source_type != "interview_questions" or chunk.level != "parent":
            continue

        # Find child chunks for this parent
        children = [
            c for c in chunks
            if c.parent_id == chunk.id
        ]

        # Extract key concepts from child section titles and content
        key_concepts = []
        for child in children:
            if child.section.lower() not in ("answer", "interview tips", "example"):
                key_concepts.append(child.section)

        # Build compound question ID: category_qN
        compound_id = f"{chunk.category}_{chunk.question_id.lower()}"

        rubrics.append({
            "question_id": compound_id,
            "bare_id": chunk.question_id,
            "category": chunk.category,
            "question": chunk.question_text,
            "difficulty": chunk.difficulty or "intermediate",
            "rubric": {
                "key_concepts": key_concepts,
                "bonus_concepts": [],
                "common_mistakes": [],
            },
        })

    return rubrics


def build_index(force: bool = False) -> dict:
    import lancedb

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(INDEX_DIR / "lancedb"))

    # Check if index already exists
    if not force and TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        return {
            "status": "exists",
            "message": "Index already exists. Use --force to rebuild.",
            "chunk_count": table.count_rows(),
        }

    start = time.time()

    # 1. Chunk all content
    chunks = chunk_all_content()
    chunk_time = time.time() - start

    # 2. Embed in batches
    embed_start = time.time()
    enriched_texts = [c.enriched_text for c in chunks]
    batch_size = 32
    all_vectors = []
    for i in range(0, len(enriched_texts), batch_size):
        batch = enriched_texts[i : i + batch_size]
        vectors = embed_documents(batch)
        all_vectors.append(vectors)
    all_vectors = np.vstack(all_vectors)
    embed_time = time.time() - embed_start

    # 3. Store in LanceDB
    store_start = time.time()
    records = [
        _chunk_to_record(chunk, vec)
        for chunk, vec in zip(chunks, all_vectors)
    ]

    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    table = db.create_table(TABLE_NAME, records)

    # 4. Create FTS index on enriched_text
    table.create_fts_index("enriched_text")
    store_time = time.time() - store_start

    # 5. Extract rubrics
    rubrics = _extract_rubrics(chunks)
    RUBRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUBRICS_PATH.write_text(json.dumps(rubrics, indent=2))

    # 6. Load and validate concept graph (informational)
    graph_stats = None
    try:
        from cli.graph import ConceptGraph, GRAPH_PATH
        if GRAPH_PATH.exists():
            cg = ConceptGraph.load(GRAPH_PATH)
            if not cg.is_empty():
                graph_stats = cg.summary()
    except Exception:
        pass

    total_time = time.time() - start
    parents = sum(1 for c in chunks if c.level == "parent")
    children = sum(1 for c in chunks if c.level == "child")

    result = {
        "status": "built",
        "total_chunks": len(chunks),
        "parents": parents,
        "children": children,
        "rubrics_extracted": len(rubrics),
        "timing": {
            "chunking_s": round(chunk_time, 2),
            "embedding_s": round(embed_time, 2),
            "storage_s": round(store_time, 2),
            "total_s": round(total_time, 2),
        },
    }
    if graph_stats:
        result["concept_graph"] = graph_stats
    return result


def get_index_status() -> dict:
    import lancedb

    db_path = INDEX_DIR / "lancedb"
    if not db_path.exists():
        return {"status": "not_built", "message": "Index has not been built yet. Run 'mlr index build'."}

    db = lancedb.connect(str(db_path))
    if TABLE_NAME not in db.table_names():
        return {"status": "not_built", "message": "Index table not found. Run 'mlr index build'."}

    table = db.open_table(TABLE_NAME)
    total = table.count_rows()

    # Get counts by type
    df = table.to_pandas()
    parents = int((df["level"] == "parent").sum())
    children = int((df["level"] == "child").sum())

    source_counts = df["source_type"].value_counts().to_dict()

    rubrics_exist = RUBRICS_PATH.exists()
    rubric_count = 0
    if rubrics_exist:
        rubric_count = len(json.loads(RUBRICS_PATH.read_text()))

    return {
        "status": "ready",
        "total_chunks": total,
        "parents": parents,
        "children": children,
        "source_counts": source_counts,
        "rubrics": rubric_count,
        "index_path": str(db_path),
    }
