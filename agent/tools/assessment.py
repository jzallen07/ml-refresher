from __future__ import annotations

import json

from anthropic import AsyncAnthropic

from cli.api import MLRefresherAPI
from agent.tools import Tool
from agent.prompts import QUIZ_GENERATION_PROMPT, EVALUATION_PROMPT


def make_assessment_tools(
    api: MLRefresherAPI, client: AsyncAnthropic, model: str
) -> list[Tool]:

    async def get_interview_question(input: dict) -> dict:
        return api.next_question(
            topic=input["topic"],
            difficulty=input.get("difficulty"),
            exclude_ids=input.get("exclude_ids", []),
        )

    async def generate_quiz(input: dict) -> dict:
        topic = input["topic"]
        num_questions = input.get("num_questions", 3)
        difficulty = input.get("difficulty", "intermediate")

        content_results = api.search_content(query=topic, topic=topic, limit=5)
        context = "\n\n".join(
            r.get("text", r.get("content", "")) for r in content_results
        )

        prompt = QUIZ_GENERATION_PROMPT.format(
            topic=topic,
            num_questions=num_questions,
            difficulty=difficulty,
            context=context,
        )

        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        # Extract JSON from response
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            questions = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                parsed = json.loads(text[start:end])
                questions = parsed.get("questions", [parsed])
            except (ValueError, json.JSONDecodeError):
                return {"error": "Failed to parse quiz questions", "raw": text}

        return {"topic": topic, "difficulty": difficulty, "questions": questions}

    async def evaluate_answer(input: dict) -> dict:
        question_id = input.get("question_id")
        question_text = input.get("question_text", "")
        answer = input["answer"]
        rubric_concepts = input.get("rubric_concepts", [])

        # Load rubric if question_id provided
        rubric_text = ""
        if question_id:
            rubric = api.get_question_with_rubric(question_id)
            if rubric and "rubric" in rubric:
                rubric_data = rubric["rubric"]
                rubric_concepts = rubric_concepts or rubric_data.get("key_concepts", [])
                rubric_text = json.dumps(rubric_data, indent=2)
                question_text = question_text or rubric.get("question", "")

        prompt = EVALUATION_PROMPT.format(
            question=question_text,
            answer=answer,
            rubric=rubric_text,
            concepts=json.dumps(rubric_concepts),
        )

        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            result = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"error": "Failed to parse evaluation", "raw": text}

        # Validate
        score = result.get("score", 0)
        result["score"] = max(0, min(100, score))
        if rubric_concepts:
            for key in ("concepts_covered", "concepts_missed"):
                if key in result:
                    result[key] = [
                        c for c in result[key] if c in rubric_concepts
                    ]

        return result

    return [
        Tool(
            name="get_interview_question",
            description="Get an interview question for a topic, with rubric. Supports filtering by difficulty and excluding already-asked questions.",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic slug"},
                    "difficulty": {
                        "type": "string",
                        "enum": ["basic", "intermediate", "advanced"],
                        "description": "Difficulty level filter",
                    },
                    "exclude_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Question IDs to exclude",
                    },
                },
                "required": ["topic"],
            },
            execute=get_interview_question,
        ),
        Tool(
            name="generate_quiz",
            description="Generate a quiz on a topic using AI. Retrieves relevant content and creates questions with expected answers.",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to quiz on"},
                    "num_questions": {
                        "type": "integer",
                        "description": "Number of questions (default 3)",
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["basic", "intermediate", "advanced"],
                        "description": "Difficulty level (default intermediate)",
                    },
                },
                "required": ["topic"],
            },
            execute=generate_quiz,
        ),
        Tool(
            name="evaluate_answer",
            description="Evaluate a student's answer against a rubric using AI. Returns score, covered/missed concepts, and feedback.",
            input_schema={
                "type": "object",
                "properties": {
                    "question_id": {
                        "type": "string",
                        "description": "Question ID to load rubric from",
                    },
                    "question_text": {
                        "type": "string",
                        "description": "The question that was asked",
                    },
                    "answer": {"type": "string", "description": "The student's answer"},
                    "rubric_concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Expected concepts (overrides rubric if provided)",
                    },
                },
                "required": ["answer"],
            },
            execute=evaluate_answer,
        ),
    ]
