from __future__ import annotations

from dataclasses import dataclass

from agent.prompts import (
    WARMUP_PROMPT,
    INTRODUCE_PROMPT,
    EXPLORE_PROMPT,
    PRACTICE_PROMPT,
    WRAPUP_PROMPT,
    SETUP_PROMPT,
    QUESTION_PROMPT,
    FOLLOWUP_PROMPT,
    EVALUATE_PROMPT,
    DEBRIEF_PROMPT,
)


@dataclass
class PhaseConfig:
    system_prompt_template: str
    available_tools: list[str]
    forced_tool: str | None
    max_turns: int
    transition_condition: str  # "auto" | "user_ready" | "quiz_complete"


TEACHER_PHASES: dict[str, PhaseConfig] = {
    "warmup": PhaseConfig(
        system_prompt_template=WARMUP_PROMPT,
        available_tools=["get_review_schedule", "get_question", "evaluate_answer"],
        forced_tool="get_review_schedule",
        max_turns=4,
        transition_condition="auto",
    ),
    "introduce": PhaseConfig(
        system_prompt_template=INTRODUCE_PROMPT,
        available_tools=["search_content", "get_lesson", "render_diagram", "get_learning_path", "show_visualization", "render_heatmap", "render_function_plot", "render_tensor_shapes"],
        forced_tool="search_content",
        max_turns=3,
        transition_condition="auto",
    ),
    "explore": PhaseConfig(
        system_prompt_template=EXPLORE_PROMPT,
        available_tools=[
            "search_content",
            "run_python",
            "get_code_example",
            "render_diagram",
            "get_learning_path",
            "show_visualization",
            "render_heatmap",
            "render_function_plot",
            "render_tensor_shapes",
        ],
        forced_tool=None,
        max_turns=15,
        transition_condition="user_ready",
    ),
    "practice": PhaseConfig(
        system_prompt_template=PRACTICE_PROMPT,
        available_tools=[
            "generate_quiz",
            "evaluate_answer",
            "run_python",
            "search_content",
            "show_visualization",
        ],
        forced_tool="generate_quiz",
        max_turns=10,
        transition_condition="quiz_complete",
    ),
    "wrapup": PhaseConfig(
        system_prompt_template=WRAPUP_PROMPT,
        available_tools=["update_progress"],
        forced_tool="update_progress",
        max_turns=2,
        transition_condition="auto",
    ),
}

INTERVIEWER_PHASES: dict[str, PhaseConfig] = {
    "setup": PhaseConfig(
        system_prompt_template=SETUP_PROMPT,
        available_tools=["get_progress"],
        forced_tool="get_progress",
        max_turns=2,
        transition_condition="auto",
    ),
    "question": PhaseConfig(
        system_prompt_template=QUESTION_PROMPT,
        available_tools=["get_interview_question", "search_content"],
        forced_tool="get_interview_question",
        max_turns=2,
        transition_condition="auto",
    ),
    "followup": PhaseConfig(
        system_prompt_template=FOLLOWUP_PROMPT,
        available_tools=["search_content"],
        forced_tool=None,
        max_turns=4,
        transition_condition="user_ready",
    ),
    "evaluate": PhaseConfig(
        system_prompt_template=EVALUATE_PROMPT,
        available_tools=["evaluate_answer", "search_content"],
        forced_tool="evaluate_answer",
        max_turns=2,
        transition_condition="auto",
    ),
    "debrief": PhaseConfig(
        system_prompt_template=DEBRIEF_PROMPT,
        available_tools=["update_progress", "get_progress", "get_learning_path", "show_visualization", "render_heatmap"],
        forced_tool="update_progress",
        max_turns=3,
        transition_condition="auto",
    ),
}

TEACHER_PHASE_ORDER = ["warmup", "introduce", "explore", "practice", "wrapup"]
INTERVIEWER_PHASE_ORDER = ["setup", "question", "followup", "evaluate", "debrief"]
