from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYTORCH_DIR = REPO_ROOT / "pytorch_refresher"

SECTION_RE = re.compile(
    r"^# =+\n# (SECTION \d+:.*)\n# =+",
    re.MULTILINE,
)


def _parse_readme(readme: Path) -> dict:
    text = readme.read_text()
    result: dict = {}

    # Title is the first # heading
    if m := re.search(r"^# (.+)$", text, re.MULTILINE):
        result["title"] = m.group(1)

    # Learning objectives: numbered items after "## Learning Objectives"
    if m := re.search(
        r"## Learning Objectives\n(.*?)(?=\n## |\Z)", text, re.DOTALL
    ):
        objectives = re.findall(r"^\d+\.\s+(.+)$", m.group(1), re.MULTILINE)
        result["learning_objectives"] = objectives

    # Key concepts: ### headings inside ## Key Concepts
    if m := re.search(
        r"## Key Concepts\n(.*?)(?=\n## |\Z)", text, re.DOTALL
    ):
        concepts = re.findall(r"^### (.+)$", m.group(1), re.MULTILINE)
        result["key_concepts"] = concepts

    return result


def _parse_lesson_py(lesson_file: Path) -> list[dict]:
    text = lesson_file.read_text()
    sections: list[dict] = []

    # Split on SECTION headers
    parts = SECTION_RE.split(text)
    # parts[0] is the preamble (imports, header), then alternating title/body
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append({"title": title, "code": body.strip()})

    return sections


def list_lessons() -> list[dict]:
    lessons = []
    if not PYTORCH_DIR.exists():
        return lessons

    for d in sorted(PYTORCH_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        readme = d / "README.md"
        if not readme.exists():
            continue

        # Extract number and slug from dir name like "01_tensors"
        parts = d.name.split("_", 1)
        num = parts[0]
        slug = parts[1] if len(parts) > 1 else d.name

        info = _parse_readme(readme)
        lessons.append({
            "id": f"pytorch/{num}",
            "number": int(num),
            "slug": slug,
            "dir_name": d.name,
            "title": info.get("title", slug),
            "category": "pytorch",
        })

    return lessons


def get_lesson(lesson_id: str) -> dict | None:
    # lesson_id like "pytorch/01"
    parts = lesson_id.split("/")
    if len(parts) != 2 or parts[0] != "pytorch":
        return None

    num = parts[1].zfill(2)

    # Find matching directory
    for d in sorted(PYTORCH_DIR.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith(f"{num}_"):
            readme = d / "README.md"
            lesson_py = d / "lesson.py"

            result: dict = {
                "id": lesson_id,
                "dir_name": d.name,
                "category": "pytorch",
            }

            if readme.exists():
                result.update(_parse_readme(readme))
                result["readme"] = readme.read_text()

            if lesson_py.exists():
                result["sections"] = _parse_lesson_py(lesson_py)
                result["code_file"] = str(lesson_py)

            return result

    return None


def get_lesson_code_examples(lesson_id: str, name: str | None = None) -> list[dict]:
    lesson = get_lesson(lesson_id)
    if not lesson or "sections" not in lesson:
        return []

    sections = lesson["sections"]
    if name:
        name_lower = name.lower().replace(" ", "_")
        sections = [
            s for s in sections
            if name_lower in s["title"].lower().replace(" ", "_")
        ]

    return sections
