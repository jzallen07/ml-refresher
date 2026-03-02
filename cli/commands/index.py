from __future__ import annotations

import json

import click


@click.group()
def index():
    """Manage the content search index."""


@index.command()
@click.option("--force", is_flag=True, help="Rebuild index even if it exists.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def build(force: bool, as_json: bool):
    """Build the search index from all content."""
    from cli.services.indexer import build_index

    if not as_json:
        click.echo("Building search index...")

    result = build_index(force=force)

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    if result["status"] == "exists":
        click.echo(result["message"])
        return

    click.echo(f"Index built successfully!")
    click.echo(f"  Total chunks: {result['total_chunks']}")
    click.echo(f"  Parents: {result['parents']}")
    click.echo(f"  Children: {result['children']}")
    click.echo(f"  Rubrics extracted: {result['rubrics_extracted']}")
    click.echo(f"  Time: {result['timing']['total_s']}s")


@index.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def status(as_json: bool):
    """Show the current index status."""
    from cli.services.indexer import get_index_status

    result = get_index_status()

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    if result["status"] == "not_built":
        click.echo(result["message"])
        return

    click.echo("Index Status: Ready")
    click.echo(f"  Total chunks: {result['total_chunks']}")
    click.echo(f"  Parents: {result['parents']}")
    click.echo(f"  Children: {result['children']}")
    click.echo(f"  Rubrics: {result['rubrics']}")
    if source_counts := result.get("source_counts"):
        click.echo("  Sources:")
        for src, count in source_counts.items():
            click.echo(f"    {src}: {count}")
