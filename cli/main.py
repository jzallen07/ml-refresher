import click

from cli.commands.content import content
from cli.commands.code import code
from cli.commands.diagram import diagram
from cli.commands.index import index


@click.group()
@click.version_option(version="0.1.0", prog_name="mlr")
def mlr():
    """ML Refresher CLI — browse lessons, run code, and view diagrams."""


mlr.add_command(content)
mlr.add_command(code)
mlr.add_command(diagram)
mlr.add_command(index)
