from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from anthropic import AsyncAnthropic

from textual.message import Message

from cli.api import MLRefresherAPI
from agent.harness import build_registry, DEFAULT_MODEL
from agent.loop import AgentLoop
from agent.orchestrator import SessionOrchestrator

if TYPE_CHECKING:
    from tui.app import MLRefresherApp


class ToolStarted(Message):
    """Message posted when a tool begins execution."""

    def __init__(self, name: str, input: dict) -> None:
        super().__init__()
        self.name = name
        self.input = input


class ToolFinished(Message):
    """Message posted when a tool finishes execution."""

    def __init__(self, name: str, result: dict) -> None:
        super().__init__()
        self.name = name
        self.result = result


def make_callbacks(
    app: MLRefresherApp,
) -> tuple[Callable[[str], None], Callable[[str, dict], None], Callable[[str, dict], None]]:
    def on_text(chunk: str) -> None:
        if app._active_stream is not None:
            # write() is async; schedule it on the running event loop.
            # Safe because on_text is called from within an async @work worker.
            asyncio.ensure_future(app._active_stream.write(chunk))

    def on_tool_start(name: str, input: dict) -> None:
        app.post_message(ToolStarted(name, input))

    def on_tool_end(name: str, result: dict) -> None:
        app.post_message(ToolFinished(name, result))

    return on_text, on_tool_start, on_tool_end


def create_session(
    mode: str,
    topic: str,
    model: str | None,
    on_text: Callable[[str], None],
    on_tool_start: Callable[[str, dict], None],
    on_tool_end: Callable[[str, dict], None],
) -> tuple[SessionOrchestrator, MLRefresherAPI]:
    resolved_model = model or DEFAULT_MODEL
    client = AsyncAnthropic()
    api = MLRefresherAPI()
    registry = build_registry(api, client, resolved_model)

    agent = AgentLoop(
        client=client,
        model=resolved_model,
        registry=registry,
        on_text=on_text,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )

    return SessionOrchestrator(mode=mode, topic=topic, agent=agent), api
