from __future__ import annotations

import asyncio

from cli.api import MLRefresherAPI
from agent.tools import Tool


def make_content_tools(api: MLRefresherAPI) -> list[Tool]:
    async def search_content(input: dict) -> dict:
        results = await asyncio.to_thread(
            api.search_content,
            query=input["query"],
            topic=input.get("topic"),
            source_type=input.get("source_type"),
            has_code=input.get("has_code"),
            limit=input.get("limit", 5),
        )
        return {"results": results}

    async def get_lesson(input: dict) -> dict:
        result = await asyncio.to_thread(api.get_lesson, input["lesson_id"])
        if result is None:
            return {"error": f"Lesson '{input['lesson_id']}' not found"}
        return result

    async def get_question(input: dict) -> dict:
        result = await asyncio.to_thread(
            api.get_question_with_rubric, input["question_id"]
        )
        if result is None:
            return {"error": f"Question '{input['question_id']}' not found"}
        return result

    return [
        Tool(
            name="search_content",
            description="Search ML learning content using semantic and keyword search. Returns relevant lessons, questions, and code examples.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "topic": {"type": "string", "description": "Filter by topic slug"},
                    "source_type": {
                        "type": "string",
                        "enum": ["lesson", "question", "code"],
                        "description": "Filter by content type",
                    },
                    "has_code": {"type": "boolean", "description": "Only return results with code"},
                    "limit": {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": ["query"],
            },
            execute=search_content,
        ),
        Tool(
            name="get_lesson",
            description="Get a full lesson by ID, including learning objectives, key concepts, and sections.",
            input_schema={
                "type": "object",
                "properties": {
                    "lesson_id": {"type": "string", "description": "Lesson ID (e.g. 'pytorch/01')"},
                },
                "required": ["lesson_id"],
            },
            execute=get_lesson,
        ),
        Tool(
            name="get_question",
            description="Get a question with its full rubric, including expected concepts and scoring criteria.",
            input_schema={
                "type": "object",
                "properties": {
                    "question_id": {"type": "string", "description": "Question ID"},
                },
                "required": ["question_id"],
            },
            execute=get_question,
        ),
    ]
