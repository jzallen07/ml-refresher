from __future__ import annotations

import json

import click

from cli.services.diagrams import list_diagrams as _list_diagrams, render


@click.group()
def diagram():
    """View ASCII architecture diagrams."""


@diagram.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def list_cmd(as_json: bool):
    """List available diagram types."""
    diagrams = _list_diagrams()

    if as_json:
        click.echo(json.dumps(diagrams, indent=2))
        return

    click.echo("Available Diagrams")
    click.echo("=" * 40)
    for d in diagrams:
        click.echo(f"  {d['type']:<25} {d['title']}")


@diagram.command()
@click.argument("diagram_type")
@click.option(
    "--annotate",
    help="Comma-separated key=value pairs (e.g. d_model=512,num_heads=8).",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def show(diagram_type: str, annotate: str | None, as_json: bool):
    """Render an ASCII diagram (e.g. transformer_full)."""
    annotations: dict[str, str] = {}
    if annotate:
        for pair in annotate.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                annotations[k.strip()] = v.strip()

    result = render(diagram_type, annotations)
    if not result:
        available = [d["type"] for d in _list_diagrams()]
        raise click.ClickException(
            f"Unknown diagram: {diagram_type}\nAvailable: {', '.join(available)}"
        )

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"{result['title']}")
    click.echo("=" * 60)
    if result["annotations"]:
        params = ", ".join(f"{k}={v}" for k, v in result["annotations"].items())
        click.echo(f"Parameters: {params}")
    click.echo(result["rendered"])
