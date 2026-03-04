from __future__ import annotations

import json

import click

from cli.services.lessons import list_lessons, get_lesson, get_lesson_code_examples
from cli.services.questions import (
    list_topics as list_interview_topics,
    get_questions,
    get_question,
)


@click.group()
def content():
    """Browse lessons and interview questions."""


@content.command()
@click.option("--category", type=click.Choice(["pytorch", "interview"]), help="Filter by category.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def topics(category: str | None, as_json: bool):
    """List all available topics."""
    results: list[dict] = []

    if category != "interview":
        for lesson in list_lessons():
            results.append(lesson)

    if category != "pytorch":
        for topic in list_interview_topics():
            results.append(topic)

    if as_json:
        click.echo(json.dumps(results, indent=2))
        return

    pytorch = [t for t in results if t["category"] == "pytorch"]
    interview = [t for t in results if t["category"] == "interview"]

    if pytorch:
        click.echo("PyTorch Refresher")
        click.echo("=" * 40)
        for t in pytorch:
            click.echo(f"  {t['id']:<15} {t['title']}")
        click.echo()

    if interview:
        click.echo("Interview Questions")
        click.echo("=" * 40)
        for t in interview:
            q_label = f"({t['question_count']}Q)"
            click.echo(f"  {t['id']:<30} {t['title']:<35} {q_label}")


@content.command()
@click.argument("lesson_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def lesson(lesson_id: str, as_json: bool):
    """Get a lesson by ID (e.g. pytorch/01)."""
    result = get_lesson(lesson_id)
    if not result:
        raise click.ClickException(f"Lesson not found: {lesson_id}")

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"Lesson: {result.get('title', lesson_id)}")
    click.echo("=" * 60)

    if objectives := result.get("learning_objectives"):
        click.echo("\nLearning Objectives:")
        for i, obj in enumerate(objectives, 1):
            click.echo(f"  {i}. {obj}")

    if concepts := result.get("key_concepts"):
        click.echo("\nKey Concepts:")
        for c in concepts:
            click.echo(f"  - {c}")

    if sections := result.get("sections"):
        click.echo(f"\nCode Sections ({len(sections)}):")
        for s in sections:
            click.echo(f"  - {s['title']}")

    if code_file := result.get("code_file"):
        click.echo(f"\nCode file: {code_file}")


@content.command()
@click.argument("topic")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def questions(topic: str, as_json: bool):
    """List questions in a topic (e.g. attention_mechanisms)."""
    result = get_questions(topic)
    if result is None:
        raise click.ClickException(f"Topic not found: {topic}")

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"Questions in: {topic}")
    click.echo("=" * 60)
    for q in result:
        click.echo(f"  {q['id']:<6} {q['title']}")


@content.command()
@click.argument("q_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def question(q_id: str, as_json: bool):
    """Get a specific question by ID (e.g. Q2)."""
    result = get_question(q_id)
    if not result:
        raise click.ClickException(f"Question not found: {q_id}")

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"{result['id']}: {result['title']}")
    click.echo(f"Topic: {result.get('topic', 'unknown')}")
    click.echo("=" * 60)
    click.echo()
    click.echo(result["body"])


@content.command("next-question")
@click.argument("topic")
@click.option("--difficulty", type=click.Choice(["basic", "intermediate", "advanced"]), help="Filter by difficulty.")
@click.option("--exclude", multiple=True, help="Question IDs to exclude (e.g. --exclude Q1 --exclude Q3).")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def next_question(topic: str, difficulty: str | None, exclude: tuple[str, ...], as_json: bool):
    """Pick the best next question for a topic using FSRS-aware selection."""
    from cli.api import MLRefresherAPI

    api = MLRefresherAPI()
    result = api.next_question(topic, difficulty=difficulty, exclude_ids=list(exclude) if exclude else None)

    if "error" in result:
        raise click.ClickException(result["error"])

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"Question: {result.get('id', '?')}: {result.get('title', '?')}")
    click.echo(f"Selection: {result.get('selection_reason', '?')}")
    click.echo(f"Detail: {result.get('selection_detail', '?')}")
    if result.get("difficulty"):
        click.echo(f"Difficulty: {result['difficulty']}")


@content.command()
@click.option("--node", "node_id", help="Show neighbors of a specific node.")
@click.option("--topic", "topic_slug", help="Show concepts for a topic.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def graph(node_id: str | None, topic_slug: str | None, as_json: bool):
    """Browse the concept relationship graph."""
    from cli.api import MLRefresherAPI

    api = MLRefresherAPI()

    if node_id:
        result = api.get_concept_neighbors(node_id)
        if "error" in result:
            raise click.ClickException(result["error"])
        if as_json:
            click.echo(json.dumps(result, indent=2))
            return
        node = result["node"]
        click.echo(f"Node: {node.get('label', node_id)} ({node.get('type', '?')})")
        click.echo("=" * 60)
        if result["prerequisites"]:
            click.echo("\nPrerequisites:")
            for p in result["prerequisites"]:
                click.echo(f"  <- {p.get('label', p['id'])} [{p.get('edge_type', '?')}]")
        if result["dependents"]:
            click.echo("\nDependents:")
            for d in result["dependents"]:
                click.echo(f"  -> {d.get('label', d['id'])} [{d.get('edge_type', '?')}]")
        for edge_type, nodes in result["neighbors"].items():
            if edge_type in ("PREREQUISITE", "BUILDS_ON"):
                continue  # Already shown above
            click.echo(f"\n{edge_type}:")
            for n in nodes:
                direction = n.get("direction", "?")
                arrow = "<-" if direction == "incoming" else "->"
                click.echo(f"  {arrow} {n.get('label', n['id'])}")
    elif topic_slug:
        from cli.graph import get_concept_graph
        cg = get_concept_graph()
        concepts = cg.get_concepts_for_topic(topic_slug)
        if as_json:
            click.echo(json.dumps(concepts, indent=2))
            return
        click.echo(f"Concepts for: {topic_slug}")
        click.echo("=" * 40)
        for c in concepts:
            click.echo(f"  - {c.get('label', c['id'])}")
        if not concepts:
            click.echo("  (none)")
    else:
        result = api.get_graph_summary()
        if as_json:
            click.echo(json.dumps(result, indent=2))
            return
        if result.get("status") == "empty":
            click.echo("Concept graph not loaded.")
            return
        click.echo("Concept Graph Summary")
        click.echo("=" * 40)
        click.echo(f"Total nodes: {result['total_nodes']}")
        for ntype, count in sorted(result["nodes_by_type"].items()):
            click.echo(f"  {ntype}: {count}")
        click.echo(f"Total edges: {result['total_edges']}")
        for etype, count in sorted(result["edges_by_type"].items()):
            click.echo(f"  {etype}: {count}")


@content.command()
@click.option("--topic", "topic_slug", help="Learning path for a topic.")
@click.option("--concept", "concept_slug", help="Learning path for a concept.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def path(topic_slug: str | None, concept_slug: str | None, as_json: bool):
    """Show an ordered learning path with prerequisites first."""
    from cli.api import MLRefresherAPI

    if not topic_slug and not concept_slug:
        raise click.ClickException("Specify --topic or --concept.")

    api = MLRefresherAPI()
    result = api.get_learning_path(topic=topic_slug, concept=concept_slug)

    if "error" in result:
        raise click.ClickException(result["error"])

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    target = result["target"]
    path_items = result["path"]
    click.echo(f"Learning Path: {target}")
    click.echo(f"Steps: {result['total_steps']}")
    click.echo("=" * 60)
    for item in path_items:
        markers = {
            "foundation": "F",
            "target": "*",
            "intermediate": " ",
        }
        marker = markers.get(item["classification"], " ")
        prereqs = ""
        if item["prerequisites"]:
            prereqs = f"  (requires: {', '.join(item['prerequisites'])})"
        click.echo(f"  [{marker}] {item['order']}. {item['label']}{prereqs}")

    click.echo()
    click.echo("Legend: [F] = foundation (no prerequisites), [*] = target")


@content.command()
@click.argument("query")
@click.option("--topic", help="Filter by topic slug.")
@click.option("--source", "source_type", help="Filter by source type (interview_questions, pytorch_lesson, python_file).")
@click.option("--code-only", is_flag=True, help="Only return chunks containing code.")
@click.option("--limit", default=5, help="Number of results to return.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def search(query: str, topic: str | None, source_type: str | None, code_only: bool, limit: int, as_json: bool):
    """Search content using semantic + keyword search."""
    from cli.services.retriever import search as do_search

    results = do_search(
        query=query,
        topic=topic,
        source_type=source_type,
        has_code=True if code_only else None,
        limit=limit,
    )

    if not results:
        raise click.ClickException("No results found. Is the index built? Run 'mlr index build'.")

    if as_json:
        click.echo(json.dumps(results, indent=2))
        return

    click.echo(f"Search: {query}")
    click.echo(f"Results: {len(results)}")
    click.echo("=" * 60)

    for r in results:
        click.echo()
        click.echo(f"  #{r['rank']} [score: {r['relevance_score']:.4f}]")
        click.echo(f"  Source: {r['source_file']}")
        if r["topic"]:
            click.echo(f"  Topic: {r['topic']}")
        click.echo(f"  Section: {r['metadata']['section']}")
        click.echo()
        # Show preview (first 200 chars)
        preview = r["text"][:200].replace("\n", "\n    ")
        click.echo(f"    {preview}")
        if len(r["text"]) > 200:
            click.echo("    ...")
        click.echo()
