from __future__ import annotations

import asyncio
import os
import sys

from openrouter import OpenRouter

from cli.api import MLRefresherAPI
from agent.tools import ToolRegistry
from agent.tools.content import make_content_tools
from agent.tools.code import make_code_tools
from agent.tools.state import make_state_tools
from agent.tools.presentation import make_presentation_tools
from agent.tools.visualization import make_visualization_tools
from agent.tools.assessment import make_assessment_tools
from agent.loop import AgentLoop
from agent.orchestrator import SessionOrchestrator


DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"


def make_client() -> OpenRouter:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")
    return OpenRouter(api_key=api_key)


def build_registry(api: MLRefresherAPI, client: OpenRouter, model: str) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in make_content_tools(api):
        registry.register(tool)
    for tool in make_code_tools(api):
        registry.register(tool)
    for tool in make_state_tools(api):
        registry.register(tool)
    for tool in make_presentation_tools(api):
        registry.register(tool)
    for tool in make_visualization_tools(api):
        registry.register(tool)
    for tool in make_assessment_tools(api, client, model):
        registry.register(tool)
    return registry


def on_text(text: str):
    print(text, end="", flush=True)


def on_tool_start(name: str, input: dict):
    print(f"\n[tool: {name}]", flush=True)


def on_tool_end(name: str, result: dict):
    if "error" in result:
        print(f"[error: {result['error']}]", flush=True)


async def run_session(mode: str, topic: str, model: str = DEFAULT_MODEL):
    client = make_client()
    api = MLRefresherAPI()
    registry = build_registry(api, client, model)

    agent = AgentLoop(
        client=client,
        model=model,
        registry=registry,
        on_text=on_text,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )

    orchestrator = SessionOrchestrator(mode=mode, topic=topic, agent=agent)

    print(f"Starting {mode} session on: {topic}")
    print(f"Model: {model}")
    print("Type 'quit' or 'q' to exit.\n")

    response = await orchestrator.start()
    print()

    while not orchestrator.is_complete:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if user_input.lower() in ("quit", "q", "exit"):
            print("Session ended.")
            break

        if not user_input:
            continue

        print()
        response = await orchestrator.handle_user_message(user_input)
        print()

    print(f"\nSession complete. Phase reached: {orchestrator.current_phase}")
