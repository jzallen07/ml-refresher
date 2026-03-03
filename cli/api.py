from __future__ import annotations

from cli.services.lessons import list_lessons, get_lesson, get_lesson_code_examples
from cli.services.questions import (
    list_topics,
    get_questions,
    get_question,
    get_question_with_rubric,
    get_rubrics_for_topic,
)
from cli.services.executor import run_code
from cli.services.diagrams import (
    list_diagrams,
    render as render_diagram,
)


class MLRefresherAPI:
    """Programmatic interface to all ML Refresher operations.

    Returns plain dicts — no Click, no formatting.
    This is the interface agent tool shims will call.
    """

    # -- Content --

    def list_all_topics(self, category: str | None = None) -> list[dict]:
        results: list[dict] = []
        if category != "interview":
            results.extend(list_lessons())
        if category != "pytorch":
            results.extend(list_topics())
        return results

    def get_lesson(self, lesson_id: str) -> dict | None:
        return get_lesson(lesson_id)

    def get_questions(self, topic: str) -> list[dict] | None:
        return get_questions(topic)

    def get_question(self, q_id: str) -> dict | None:
        return get_question(q_id)

    # -- Code --

    def run_code(
        self,
        code: str | None = None,
        file: str | None = None,
        timeout: int = 30,
    ) -> dict:
        return run_code(code=code, file=file, timeout=timeout)

    def get_code_examples(
        self, lesson_id: str, name: str | None = None
    ) -> list[dict]:
        return get_lesson_code_examples(lesson_id, name=name)

    # -- Diagrams --

    def list_diagrams(self) -> list[dict]:
        return list_diagrams()

    def get_question_with_rubric(self, q_id: str) -> dict | None:
        return get_question_with_rubric(q_id)

    # -- Concept Graph --

    def get_graph_summary(self) -> dict:
        from cli.graph import get_concept_graph
        cg = get_concept_graph()
        if cg.is_empty():
            return {"status": "empty", "message": "No concept graph loaded."}
        return {**cg.summary(), "status": "loaded"}

    def get_concept_neighbors(self, node_id: str, hops: int = 1) -> dict:
        from cli.graph import get_concept_graph
        cg = get_concept_graph()
        if cg.is_empty():
            return {"error": "Concept graph not loaded."}
        node = cg.get_node(node_id)
        if node is None:
            return {"error": f"Node '{node_id}' not found in graph."}
        neighbors = cg.get_neighbors(node_id, hops=hops)
        prerequisites = cg.get_prerequisites(node_id)
        dependents = cg.get_dependents(node_id)
        return {
            "node": node,
            "prerequisites": prerequisites,
            "dependents": dependents,
            "neighbors": neighbors,
        }

    def get_learning_path(self, topic: str | None = None, concept: str | None = None) -> dict:
        from cli.graph import get_concept_graph
        cg = get_concept_graph()
        if cg.is_empty():
            return {"error": "Concept graph not loaded."}
        path = cg.generate_learning_path(target_topic=topic, target_concept=concept)
        if not path:
            target = topic or concept
            return {"error": f"No learning path found for '{target}'."}
        return {
            "target": topic or concept,
            "path": path,
            "total_steps": len(path),
        }

    # -- Search --

    def search_content(
        self,
        query: str,
        topic: str | None = None,
        source_type: str | None = None,
        has_code: bool | None = None,
        limit: int = 5,
        include_graph_context: bool = False,
    ) -> list[dict]:
        from cli.services.retriever import search
        return search(
            query=query,
            topic=topic,
            source_type=source_type,
            has_code=has_code,
            limit=limit,
            include_graph_context=include_graph_context,
        )

    # -- Index --

    def build_index(self, force: bool = False) -> dict:
        from cli.services.indexer import build_index
        return build_index(force=force)

    def get_index_status(self) -> dict:
        from cli.services.indexer import get_index_status
        return get_index_status()

    # -- Diagrams --

    def render_diagram(
        self,
        diagram_type: str,
        annotations: dict[str, str] | None = None,
    ) -> dict | None:
        return render_diagram(diagram_type, annotations)

    # -- Progress --

    def get_progress(self, topic: str | None = None) -> dict:
        from cli.state.progress import get_progress
        return get_progress(topic)

    def update_progress(
        self,
        topic: str,
        event_type: str,
        score: float | None = None,
        concepts_tested: list[str] | None = None,
    ) -> dict:
        from cli.state.progress import update_progress
        return update_progress(topic, event_type, score, concepts_tested)

    def get_review_schedule(self) -> dict:
        from cli.state.progress import get_review_schedule
        return get_review_schedule()

    def next_question(
        self,
        topic: str,
        difficulty: str | None = None,
        exclude_ids: list[str] | None = None,
    ) -> dict:
        import json
        import random
        from cli.state.db import StateDB

        questions = get_questions(topic)
        if questions is None:
            return {"error": f"No questions found for topic '{topic}'"}

        exclude = set(exclude_ids or [])
        candidates = [q for q in questions if q["id"] not in exclude]
        if not candidates:
            return {"error": "No matching questions available", "all_exhausted": True}

        # Soft difficulty filter via rubrics
        rubrics = get_rubrics_for_topic(topic)
        if difficulty:
            filtered = [
                q for q in candidates
                if rubrics.get(q["id"], {}).get("difficulty") == difficulty
            ]
            if filtered:
                candidates = filtered

        # Load FSRS state
        db = StateDB()
        try:
            due_cards = db.get_due_cards(topic_id=topic)
            all_cards = db.get_all_cards(topic)
        finally:
            db.close()

        due_concepts = {c["concept"] for c in due_cards}
        weak_concepts = set()
        seen_concepts = set()
        for card in all_cards:
            seen_concepts.add(card["concept"])
            card_data = json.loads(card["card_json"])
            if card_data.get("stability") is not None and card_data["stability"] < 2.0:
                weak_concepts.add(card["concept"])

        # Load concept graph for prerequisite awareness
        try:
            from cli.graph import get_concept_graph
            cg = get_concept_graph()
            graph_available = not cg.is_empty()
        except Exception:
            cg = None
            graph_available = False

        # Score each candidate
        scored = []
        for q in candidates:
            rubric = rubrics.get(q["id"], {})
            concepts = rubric.get("rubric", {}).get("key_concepts", [])
            overdue = sum(1 for c in concepts if c in due_concepts)
            weak = sum(1 for c in concepts if c in weak_concepts)
            unseen = sum(1 for c in concepts if c not in seen_concepts)

            # Prerequisite penalty: deprioritize questions whose prereqs are weak
            prereq_penalty = 0.0
            weak_prereqs = []
            if graph_available:
                q_node = f"question:{topic}_{q['id'].lower()}"
                for prereq in cg.get_prerequisites(q_node):
                    prereq_topic = prereq.get("topic", "")
                    if prereq_topic:
                        db2 = StateDB()
                        try:
                            mastery = db2.get_topic_mastery(prereq_topic)
                        finally:
                            db2.close()
                        if mastery["avg_stability"] < 2.0 and mastery["card_count"] > 0:
                            prereq_penalty += 0.5
                            weak_prereqs.append(prereq_topic)

            scored.append((overdue, weak, unseen, prereq_penalty, weak_prereqs, q))

        # Sort: higher overdue/weak/unseen is better, lower prereq_penalty is better
        scored.sort(key=lambda x: (x[0], x[1], x[2], -x[3]), reverse=True)
        top = scored[0]

        # Random fallback if top score is all zeros
        if top[0] == 0 and top[1] == 0 and top[2] == 0:
            _, _, _, _, _, selected = random.choice(scored)
            reason = "random"
            detail = "No overdue, weak, or unseen concepts found"
            weak_prereqs = []
        else:
            _, _, _, _, weak_prereqs, selected = top
            parts = []
            if top[0]:
                parts.append(f"{top[0]} overdue")
            if top[1]:
                parts.append(f"{top[1]} weak")
            if top[2]:
                parts.append(f"{top[2]} unseen")
            reason = "adaptive"
            detail = f"Concept match: {', '.join(parts)}"

        compound_id = f"{topic}_{selected['id'].lower()}"
        result = get_question_with_rubric(compound_id)
        if not result:
            result = selected

        result["selection_reason"] = reason
        result["selection_detail"] = detail

        if weak_prereqs:
            result["weak_prerequisites"] = weak_prereqs
            result["prerequisite_suggestion"] = (
                f"Consider reviewing {', '.join(weak_prereqs)} first — "
                f"{'these topics are' if len(weak_prereqs) > 1 else 'this topic is'} "
                f"a prerequisite and could use more practice."
            )

        return result
