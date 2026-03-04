from __future__ import annotations

from cli.api import MLRefresherAPI
from agent.tools import Tool


def make_state_tools(api: MLRefresherAPI) -> list[Tool]:
    async def get_progress(input: dict) -> dict:
        return api.get_progress(topic=input.get("topic"))

    async def update_progress(input: dict) -> dict:
        return api.update_progress(
            topic=input["topic"],
            event_type=input["event_type"],
            score=input.get("score"),
            concepts_tested=input.get("concepts_tested"),
        )

    async def get_review_schedule(input: dict) -> dict:
        return api.get_review_schedule()

    return [
        Tool(
            name="get_progress",
            description="Get learning progress for a topic or overall. Returns level, scores, weak areas, and review status.",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic slug (omit for overall progress)"},
                },
            },
            execute=get_progress,
        ),
        Tool(
            name="update_progress",
            description="Record a learning event and update progress. Handles level promotion and FSRS card scheduling.",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic slug"},
                    "event_type": {
                        "type": "string",
                        "enum": ["quiz", "interview", "lesson", "practice"],
                        "description": "Type of learning event",
                    },
                    "score": {"type": "number", "description": "Score from 0.0 to 1.0"},
                    "concepts_tested": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concepts covered in this event",
                    },
                },
                "required": ["topic", "event_type"],
            },
            execute=update_progress,
        ),
        Tool(
            name="get_review_schedule",
            description="Get spaced-repetition cards that are due for review, sorted most overdue first.",
            input_schema={
                "type": "object",
                "properties": {},
            },
            execute=get_review_schedule,
        ),
    ]
