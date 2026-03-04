from __future__ import annotations

import json

import click

from cli.services.executor import run_code
from cli.services.lessons import get_lesson_code_examples


@click.group()
def code():
    """Run Python code and view examples."""


@code.command()
@click.option("--code", "code_str", help="Python code to execute.")
@click.option("--file", "file_path", type=click.Path(), help="Python file to execute.")
@click.option("--timeout", default=30, help="Execution timeout in seconds.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def run(code_str: str | None, file_path: str | None, timeout: int, as_json: bool):
    """Execute Python code."""
    if not code_str and not file_path:
        raise click.ClickException("Provide --code or --file")

    result = run_code(code=code_str, file=file_path, timeout=timeout)

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    if result["success"]:
        if result.get("stdout"):
            click.echo(result["stdout"], nl=False)
        click.echo(f"\n[OK] Completed in {result.get('elapsed_seconds', '?')}s")
    else:
        if result.get("stderr"):
            click.echo(result["stderr"], nl=False, err=True)
        if result.get("error"):
            click.echo(result["error"], err=True)
        raise SystemExit(result.get("exit_code", 1))


@code.command()
@click.argument("lesson_id")
@click.option("--name", help="Filter by section name.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def example(lesson_id: str, name: str | None, as_json: bool):
    """Get code examples from a lesson (e.g. pytorch/01)."""
    sections = get_lesson_code_examples(lesson_id, name=name)
    if not sections:
        raise click.ClickException(f"No code examples found for: {lesson_id}")

    if as_json:
        click.echo(json.dumps(sections, indent=2))
        return

    for i, s in enumerate(sections):
        if i > 0:
            click.echo()
        click.echo(f"--- {s['title']} ---")
        click.echo(s["code"])
