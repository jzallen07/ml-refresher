from __future__ import annotations

from cli.services.lessons import list_lessons, get_lesson, get_lesson_code_examples
from cli.services.questions import (
    list_topics,
    get_questions,
    get_question,
)
from cli.services.executor import run_code
from cli.services.diagrams import (
    list_diagrams,
    render as render_diagram,
)


class MLRefresherAPI:
    """Programmatic interface to all ML Refresher operations.

    Returns plain dicts — no Click, no formatting.
    This is the interface agent tool shims will call.
    """

    # -- Content --

    def list_all_topics(self, category: str | None = None) -> list[dict]:
        results: list[dict] = []
        if category != "interview":
            results.extend(list_lessons())
        if category != "pytorch":
            results.extend(list_topics())
        return results

    def get_lesson(self, lesson_id: str) -> dict | None:
        return get_lesson(lesson_id)

    def get_questions(self, topic: str) -> list[dict] | None:
        return get_questions(topic)

    def get_question(self, q_id: str) -> dict | None:
        return get_question(q_id)

    # -- Code --

    def run_code(
        self,
        code: str | None = None,
        file: str | None = None,
        timeout: int = 30,
    ) -> dict:
        return run_code(code=code, file=file, timeout=timeout)

    def get_code_examples(
        self, lesson_id: str, name: str | None = None
    ) -> list[dict]:
        return get_lesson_code_examples(lesson_id, name=name)

    # -- Diagrams --

    def list_diagrams(self) -> list[dict]:
        return list_diagrams()

    def render_diagram(
        self,
        diagram_type: str,
        annotations: dict[str, str] | None = None,
    ) -> dict | None:
        return render_diagram(diagram_type, annotations)
