from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QUESTIONS_DIR = REPO_ROOT / "interview_questions"

QUESTION_HEADER_RE = re.compile(r"^## (Q\d+):?\s*(.+)$", re.MULTILINE)


def _parse_questions_readme(readme: Path) -> list[dict]:
    text = readme.read_text()
    questions: list[dict] = []

    # Find all ## Q{n}: headers and split content between them
    headers = list(QUESTION_HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        q_id = m.group(1)  # e.g. "Q2"
        q_title = m.group(2).strip()
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].strip()

        # Extract ### Answer section if present
        answer = ""
        if am := re.search(
            r"### Answer\n(.*?)(?=\n### |\n## |\n---|\Z)", body, re.DOTALL
        ):
            answer = am.group(1).strip()

        questions.append({
            "id": q_id,
            "title": q_title,
            "body": body,
            "answer": answer,
        })

    return questions


def list_topics() -> list[dict]:
    topics = []
    if not QUESTIONS_DIR.exists():
        return topics

    for d in sorted(QUESTIONS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        readme = d / "README.md"
        if not readme.exists():
            continue

        # Extract number and slug from dir name like "02_attention_mechanisms"
        parts = d.name.split("_", 1)
        num = parts[0]
        slug = parts[1] if len(parts) > 1 else d.name

        # Get topic title from first # heading
        text = readme.read_text()
        title = slug.replace("_", " ").title()
        if m := re.search(r"^# (.+)$", text, re.MULTILINE):
            title = m.group(1)

        # Count questions
        question_count = len(QUESTION_HEADER_RE.findall(text))

        topics.append({
            "id": slug,
            "number": int(num),
            "dir_name": d.name,
            "title": title,
            "question_count": question_count,
            "category": "interview",
        })

    return topics


def _find_topic_dir(topic: str) -> Path | None:
    """Find a topic directory by slug or partial match."""
    for d in sorted(QUESTIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        # Match by slug (everything after the number prefix)
        parts = d.name.split("_", 1)
        slug = parts[1] if len(parts) > 1 else d.name
        if slug == topic or d.name == topic:
            return d
    return None


def get_questions(topic: str) -> list[dict] | None:
    d = _find_topic_dir(topic)
    if not d:
        return None

    readme = d / "README.md"
    if not readme.exists():
        return None

    return _parse_questions_readme(readme)


def get_question(q_id: str) -> dict | None:
    """Find a question by its ID (e.g. 'Q2') across all topics."""
    q_id_upper = q_id.upper()
    if not q_id_upper.startswith("Q"):
        q_id_upper = f"Q{q_id_upper}"

    for d in sorted(QUESTIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        readme = d / "README.md"
        if not readme.exists():
            continue

        questions = _parse_questions_readme(readme)
        for q in questions:
            if q["id"].upper() == q_id_upper:
                # Add topic context
                parts = d.name.split("_", 1)
                q["topic"] = parts[1] if len(parts) > 1 else d.name
                q["topic_dir"] = d.name
                return q

    return None
