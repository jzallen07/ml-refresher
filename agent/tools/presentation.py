from __future__ import annotations

from cli.api import MLRefresherAPI
from agent.tools import Tool


def make_presentation_tools(api: MLRefresherAPI) -> list[Tool]:
    async def render_diagram(input: dict) -> dict:
        result = api.render_diagram(
            diagram_type=input["diagram_type"],
            annotations=input.get("annotations"),
        )
        if result is None:
            available = [d["type"] for d in api.list_diagrams()]
            return {"error": f"Unknown diagram type. Available: {available}"}
        return result

    return [
        Tool(
            name="render_diagram",
            description="Render an ASCII architecture diagram with optional parameter annotations.",
            input_schema={
                "type": "object",
                "properties": {
                    "diagram_type": {
                        "type": "string",
                        "description": "Diagram type (e.g. 'transformer', 'attention')",
                    },
                    "annotations": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Key-value annotations (e.g. {\"d_model\": \"512\"})",
                    },
                },
                "required": ["diagram_type"],
            },
            execute=render_diagram,
        ),
    ]
