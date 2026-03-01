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
