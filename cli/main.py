import click

from cli.commands.content import content
from cli.commands.code import code
from cli.commands.diagram import diagram
from cli.commands.index import index
from cli.commands.progress import progress


@click.group()
@click.version_option(version="0.1.0", prog_name="mlr")
def mlr():
    """ML Refresher CLI — browse lessons, run code, and view diagrams."""


mlr.add_command(content)
mlr.add_command(code)
mlr.add_command(diagram)
mlr.add_command(index)
mlr.add_command(progress)


@mlr.command()
@click.argument("topic")
@click.option("--model", default=None, help="Model to use.")
def learn(topic, model):
    """Start an interactive learning session."""
    from tui.app import MLRefresherApp

    MLRefresherApp(mode="teacher", topic=topic, model=model).run()


@mlr.command()
@click.argument("topic")
@click.option("--model", default=None, help="Model to use.")
def interview(topic, model):
    """Start an interactive interview session."""
    from tui.app import MLRefresherApp

    MLRefresherApp(mode="interviewer", topic=topic, model=model).run()


@mlr.command()
@click.option("--model", default=None, help="Model to use.")
def tui(model):
    """Launch interactive TUI with mode/topic selection."""
    from tui.app import MLRefresherApp

    MLRefresherApp(model=model).run()
