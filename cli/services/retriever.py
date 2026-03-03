from __future__ import annotations

from pathlib import Path

import lancedb

from cli.services.indexer import INDEX_DIR, TABLE_NAME, embed_query

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        try:
            _reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True
            )
        except OSError:
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _get_table():
    db = lancedb.connect(str(INDEX_DIR / "lancedb"))
    return db.open_table(TABLE_NAME)


def _enrich_with_graph(result: dict, cg) -> None:
    """Add graph context to a search result."""
    category = result.get("topic", "")
    question_id = result.get("metadata", {}).get("question_id", "")
    if not category or not question_id:
        return

    q_node_id = f"question:{category}_{question_id.lower()}"
    node = cg.get_node(q_node_id)
    if node is None:
        return

    prereqs = cg.get_prerequisites(q_node_id)
    deps = cg.get_dependents(q_node_id)
    concepts = cg.get_concepts_for_topic(category)

    result["graph_context"] = {
        "related_concepts": [c.get("label", c["id"]) for c in concepts],
        "prerequisite_concepts": [p.get("label", p["id"]) for p in prereqs],
        "dependent_topics": [d.get("label", d["id"]) for d in deps],
    }


def search(
    query: str,
    topic: str | None = None,
    source_type: str | None = None,
    has_code: bool | None = None,
    limit: int = 5,
    include_graph_context: bool = False,
) -> list[dict]:
    """5-stage search pipeline: hybrid retrieval, filtering, re-ranking, parent expansion, graph enrichment."""
    table = _get_table()

    # Stage 1: Hybrid retrieval (BM25 + dense + RRF)
    query_vector = embed_query(query)
    candidates_limit = 20

    search_builder = (
        table.search(query_type="hybrid")
        .vector(query_vector.tolist())
        .text(query)
        .limit(candidates_limit)
    )

    # Stage 2: Metadata filtering
    filters = []
    if topic:
        filters.append(f"category = '{topic}'")
    if source_type:
        filters.append(f"source_type = '{source_type}'")
    if has_code is not None:
        filters.append(f"has_code = {str(has_code).lower()}")

    if filters:
        where_clause = " AND ".join(filters)
        search_builder = search_builder.where(where_clause)

    candidates_df = search_builder.to_pandas()

    if candidates_df.empty:
        return []

    # Stage 3: Cross-encoder re-ranking
    reranker = _get_reranker()
    texts = candidates_df["text"].tolist()
    ids = candidates_df["id"].tolist()
    pairs = [(query, t) for t in texts]
    scores = reranker.predict(pairs)

    # Combine with original data
    ranked = sorted(
        zip(range(len(scores)), scores),
        key=lambda x: x[1],
        reverse=True,
    )
    top_indices = [idx for idx, _ in ranked[:limit]]
    top_scores = [float(score) for _, score in ranked[:limit]]

    # Stage 4: Parent expansion
    results = []
    parent_ids_to_fetch = set()
    for idx in top_indices:
        pid = candidates_df.iloc[idx]["parent_id"]
        if pid:
            parent_ids_to_fetch.add(pid)

    # Fetch parent chunks
    parent_texts = {}
    if parent_ids_to_fetch:
        all_df = table.to_pandas()
        for pid in parent_ids_to_fetch:
            matches = all_df[all_df["id"] == pid]
            if not matches.empty:
                parent_texts[pid] = matches.iloc[0]["text"]

    for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
        row = candidates_df.iloc[idx]
        parent_id = row["parent_id"]
        result = {
            "rank": rank + 1,
            "text": row["text"],
            "source_file": row["file_path"],
            "topic": row["category"] or row.get("lesson_title", ""),
            "relevance_score": round(score, 4),
            "parent_text": parent_texts.get(parent_id, ""),
            "metadata": {
                "id": row["id"],
                "source_type": row["source_type"],
                "level": row["level"],
                "section": row["section"],
                "question_id": row["question_id"],
                "has_code": bool(row["has_code"]),
                "content_type": row["content_type"],
                "lesson_number": int(row["lesson_number"]),
            },
        }
        results.append(result)

    # Stage 5: Graph enrichment
    if include_graph_context:
        try:
            from cli.graph import get_concept_graph
            cg = get_concept_graph()
            if not cg.is_empty():
                for result in results:
                    _enrich_with_graph(result, cg)
        except Exception:
            pass

    return results
