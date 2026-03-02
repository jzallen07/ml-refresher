from __future__ import annotations

import json

import click


@click.group()
def progress():
    """Track learning progress and review schedule."""


@progress.command()
@click.option("--topic", default=None, help="Show progress for a specific topic.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def show(topic: str | None, as_json: bool):
    """Show learning progress."""
    from cli.state.progress import get_progress

    result = get_progress(topic)

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return

    if topic:
        click.echo(f"Topic: {result['display_name']}")
        click.echo(f"  Level:        {result['level']}")
        click.echo(f"  Interactions: {result['total_interactions']}")
        if result["recent_scores"]:
            scores = ", ".join(f"{s:.0%}" for s in result["recent_scores"])
            click.echo(f"  Recent:       {scores}")
        if result["average_score"] is not None:
            click.echo(f"  Average:      {result['average_score']:.0%}")
        if result["weak_concepts"]:
            click.echo(f"  Weak areas:   {', '.join(result['weak_concepts'])}")
        click.echo(f"  Due cards:    {result['due_cards']}")
        if result["last_session"]:
            click.echo(f"  Last session: {result['last_session']}")
    else:
        click.echo("Overall Progress")
        click.echo("=" * 40)
        click.echo(f"  Topics:     {result['topics_started']}/{result['total_topics']} started")
        for level, count in result["level_counts"].items():
            click.echo(f"  {level.capitalize():<14}{count}")
        if result["overall_average"] is not None:
            click.echo(f"  Average:    {result['overall_average']:.0%}")
        if result["weakest_topic"]:
            click.echo(f"  Weakest:    {result['weakest_topic']}")
        if result["strongest_topic"]:
            click.echo(f"  Strongest:  {result['strongest_topic']}")


@progress.command()
@click.argument("topic")
@click.argument("event_type")
@click.option("--score", type=float, default=None, help="Score (0.0–1.0).")
@click.option("--concepts", default=None, help="Comma-separated concepts tested.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def update(topic: str, event_type: str, score: float | None, concepts: str | None, as_json: bool):
    """Record a learning event for a topic."""
    from cli.state.progress import update_progress

    concepts_list = [c.strip() for c in concepts.split(",")] if concepts else None
    result = update_progress(topic, event_type, score, concepts_list)

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return

    click.echo(f"Recorded {event_type} for {topic}")
    if score is not None:
        click.echo(f"  Score: {score:.0%}")
    click.echo(f"  Level: {result['new_level']}")
    if result["level_changed"]:
        click.echo(f"  Level up! {result['old_level']} → {result['new_level']}")


@progress.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def review(as_json: bool):
    """Show cards due for review."""
    from cli.state.progress import get_review_schedule

    result = get_review_schedule()

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    if not result["cards"]:
        click.echo("No cards due for review.")
        return

    click.echo(f"Due for review: {result['total_due']} cards")
    click.echo("-" * 40)
    for card in result["cards"]:
        click.echo(f"  [{card['topic_id']}] {card['concept']}  (due: {card['due_date'][:10]})")
