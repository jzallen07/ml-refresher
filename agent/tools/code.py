from __future__ import annotations

from cli.api import MLRefresherAPI
from agent.tools import Tool


def make_code_tools(api: MLRefresherAPI) -> list[Tool]:
    async def run_python(input: dict) -> dict:
        return api.run_code(
            code=input.get("code"),
            file=input.get("file"),
            timeout=input.get("timeout", 30),
        )

    async def get_code_example(input: dict) -> dict:
        results = api.get_code_examples(input["lesson_id"], name=input.get("name"))
        return {"examples": results}

    return [
        Tool(
            name="run_python",
            description="Execute Python code and return stdout/stderr. Use for demonstrations, experiments, and verifying concepts.",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "file": {"type": "string", "description": "Path to a Python file to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                },
            },
            execute=run_python,
        ),
        Tool(
            name="get_code_example",
            description="Get code examples from a lesson, optionally filtered by name.",
            input_schema={
                "type": "object",
                "properties": {
                    "lesson_id": {"type": "string", "description": "Lesson ID (e.g. 'pytorch/01')"},
                    "name": {"type": "string", "description": "Filter examples by name"},
                },
                "required": ["lesson_id"],
            },
            execute=get_code_example,
        ),
    ]
