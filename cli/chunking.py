from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from cli.services.questions import QUESTION_HEADER_RE

REPO_ROOT = Path(__file__).resolve().parent.parent
INTERVIEW_DIR = REPO_ROOT / "interview_questions"
PYTORCH_DIR = REPO_ROOT / "pytorch_refresher"

SUBSECTION_RE = re.compile(r"^### (.+)$", re.MULTILINE)
SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")


@dataclass
class Chunk:
    id: str
    text: str
    enriched_text: str = ""
    parent_id: str | None = None
    level: str = "child"  # "parent" or "child"
    source_type: str = ""  # interview_questions, pytorch_lesson, python_file
    file_path: str = ""
    category: str = ""
    question_id: str = ""
    question_text: str = ""
    section: str = ""
    has_code: bool = False
    content_type: str = "explanation"
    lesson_number: int = 0
    lesson_title: str = ""
    difficulty: str = ""
    function_name: str = ""
    metadata: dict = field(default_factory=dict)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _detect_content_type(text: str) -> str:
    if CODE_BLOCK_RE.search(text):
        return "code"
    if re.search(r"[=∑∏∫√]|\\frac|\\sum|\^[{T]", text):
        return "formula"
    if re.search(r"\|.*\|.*\|", text):
        return "comparison"
    if re.search(r"^\*\*Definition\*\*|^> ", text, re.MULTILINE):
        return "definition"
    return "explanation"


def _has_code(text: str) -> bool:
    return bool(CODE_BLOCK_RE.search(text))


def _extract_lesson_title(readme: Path) -> str:
    text = readme.read_text()
    if m := re.search(r"^# (.+)$", text, re.MULTILINE):
        return m.group(1)
    return readme.parent.name


def _extract_number(dir_name: str) -> int:
    parts = dir_name.split("_", 1)
    try:
        return int(parts[0])
    except ValueError:
        return 0


def _extract_slug(dir_name: str) -> str:
    parts = dir_name.split("_", 1)
    return parts[1] if len(parts) > 1 else dir_name


def chunk_interview_readme(path: Path) -> list[Chunk]:
    text = path.read_text()
    chunks: list[Chunk] = []

    dir_name = path.parent.name
    num = _extract_number(dir_name)
    category = _extract_slug(dir_name)

    headers = list(QUESTION_HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        q_id = m.group(1)  # e.g. "Q2"
        q_title = m.group(2).strip()
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        full_block = text[start:end].strip()

        parent_id = f"interview_{num:02d}_{q_id.lower()}"

        # Parent chunk: full question block
        parent = Chunk(
            id=parent_id,
            text=full_block,
            level="parent",
            source_type="interview_questions",
            file_path=str(path.relative_to(REPO_ROOT)),
            category=category,
            question_id=q_id,
            question_text=q_title,
            section="full",
            has_code=_has_code(full_block),
            content_type=_detect_content_type(full_block),
            lesson_number=num,
        )
        chunks.append(parent)

        # Child chunks: split on ### headers within this question block
        body_start = m.end() - start  # offset within full_block
        body = full_block[body_start:]
        subsections = list(SUBSECTION_RE.finditer(body))

        for j, sub in enumerate(subsections):
            sub_title = sub.group(1).strip()
            sub_start = sub.end()
            sub_end = subsections[j + 1].start() if j + 1 < len(subsections) else len(body)
            sub_text = body[sub_start:sub_end].strip()

            if not sub_text:
                continue

            # Include the header in the chunk text
            child_text = f"### {sub_title}\n\n{sub_text}"
            child_id = f"{parent_id}_{_slugify(sub_title)}"

            child = Chunk(
                id=child_id,
                text=child_text,
                level="child",
                parent_id=parent_id,
                source_type="interview_questions",
                file_path=str(path.relative_to(REPO_ROOT)),
                category=category,
                question_id=q_id,
                question_text=q_title,
                section=sub_title,
                has_code=_has_code(child_text),
                content_type=_detect_content_type(child_text),
                lesson_number=num,
            )
            chunks.append(child)

    return chunks


def chunk_pytorch_readme(path: Path) -> list[Chunk]:
    text = path.read_text()
    chunks: list[Chunk] = []

    dir_name = path.parent.name
    num = _extract_number(dir_name)
    lesson_title = _extract_lesson_title(path)

    sections = list(SECTION_RE.finditer(text))
    for i, m in enumerate(sections):
        sec_title = m.group(1).strip()
        start = m.start()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        full_section = text[start:end].strip()

        parent_id = f"pytorch_{num:02d}_{_slugify(sec_title)}"

        parent = Chunk(
            id=parent_id,
            text=full_section,
            level="parent",
            source_type="pytorch_lesson",
            file_path=str(path.relative_to(REPO_ROOT)),
            lesson_number=num,
            lesson_title=lesson_title,
            section=sec_title,
            has_code=_has_code(full_section),
            content_type=_detect_content_type(full_section),
        )
        chunks.append(parent)

        # Child chunks: ### sub-sections within this ## section
        body_start = m.end() - start
        body = full_section[body_start:]
        subsections = list(SUBSECTION_RE.finditer(body))

        for j, sub in enumerate(subsections):
            sub_title = sub.group(1).strip()
            sub_start = sub.end()
            sub_end = subsections[j + 1].start() if j + 1 < len(subsections) else len(body)
            sub_text = body[sub_start:sub_end].strip()

            if not sub_text:
                continue

            child_text = f"### {sub_title}\n\n{sub_text}"
            child_id = f"{parent_id}_{_slugify(sub_title)}"

            child = Chunk(
                id=child_id,
                text=child_text,
                level="child",
                parent_id=parent_id,
                source_type="pytorch_lesson",
                file_path=str(path.relative_to(REPO_ROOT)),
                lesson_number=num,
                lesson_title=lesson_title,
                section=sub_title,
                has_code=_has_code(child_text),
                content_type=_detect_content_type(child_text),
            )
            chunks.append(child)

    return chunks


def chunk_python_file(path: Path) -> list[Chunk]:
    text = path.read_text()
    chunks: list[Chunk] = []

    dir_name = path.parent.name
    num = _extract_number(dir_name)

    # Determine if this is pytorch or interview content
    rel = path.relative_to(REPO_ROOT)
    if str(rel).startswith("pytorch_refresher"):
        source_type = "python_file"
        lesson_title = _extract_lesson_title(path.parent / "README.md") if (path.parent / "README.md").exists() else dir_name
    else:
        source_type = "python_file"
        lesson_title = dir_name

    try:
        tree = ast.parse(text)
    except SyntaxError:
        # If AST parsing fails, treat the whole file as one chunk
        chunk_id = f"pyfile_{num:02d}_{_slugify(path.stem)}"
        chunks.append(Chunk(
            id=chunk_id,
            text=text,
            level="parent",
            source_type=source_type,
            file_path=str(rel),
            lesson_number=num,
            lesson_title=lesson_title,
            section="full_file",
            has_code=True,
            content_type="code",
        ))
        return chunks

    lines = text.splitlines(keepends=True)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno - 1
            end_line = node.end_lineno or node.lineno
            func_text = "".join(lines[start_line:end_line])
            func_name = node.name

            chunk_id = f"pyfile_{num:02d}_{_slugify(func_name)}"

            chunks.append(Chunk(
                id=chunk_id,
                text=func_text,
                level="child",
                source_type=source_type,
                file_path=str(rel),
                lesson_number=num,
                lesson_title=lesson_title,
                function_name=func_name,
                section=func_name,
                has_code=True,
                content_type="code",
            ))

    # If no functions/classes found, chunk by SECTION comments (lesson.py pattern)
    if not chunks:
        section_re = re.compile(
            r"^# =+\n# (SECTION \d+:.*)\n# =+",
            re.MULTILINE,
        )
        parts = section_re.split(text)
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                title = parts[i].strip()
                body = parts[i + 1] if i + 1 < len(parts) else ""
                section_slug = _slugify(title)
                chunk_id = f"pyfile_{num:02d}_{section_slug}"

                chunks.append(Chunk(
                    id=chunk_id,
                    text=body.strip(),
                    level="child",
                    source_type=source_type,
                    file_path=str(rel),
                    lesson_number=num,
                    lesson_title=lesson_title,
                    section=title,
                    has_code=True,
                    content_type="code",
                ))
        else:
            # Single chunk for entire file
            chunk_id = f"pyfile_{num:02d}_{_slugify(path.stem)}"
            chunks.append(Chunk(
                id=chunk_id,
                text=text,
                level="parent",
                source_type=source_type,
                file_path=str(rel),
                lesson_number=num,
                lesson_title=lesson_title,
                section="full_file",
                has_code=True,
                content_type="code",
            ))

    return chunks


def add_context_prefix(text: str, metadata: dict) -> str:
    source_type = metadata.get("source_type", "")
    if source_type == "interview_questions":
        prefix = (
            f"From LLM Interview Questions, "
            f"Category: {metadata.get('category', '').replace('_', ' ').title()}, "
            f"{metadata.get('question_id', '')}: {metadata.get('question_text', '')}. "
            f"Section: {metadata.get('section', 'Overview')}."
        )
    elif source_type == "pytorch_lesson":
        prefix = (
            f"From PyTorch Refresher, "
            f"Lesson {metadata.get('lesson_number', '')}: {metadata.get('lesson_title', '')}. "
            f"Section: {metadata.get('section', '')}."
        )
    elif source_type == "python_file":
        prefix = (
            f"Code from {metadata.get('file_path', '')}, "
            f"function: {metadata.get('function_name', '') or 'module-level'}."
        )
    else:
        prefix = ""

    return f"{prefix}\n\n{text}" if prefix else text


def chunk_all_content() -> list[Chunk]:
    all_chunks: list[Chunk] = []

    # Interview questions
    if INTERVIEW_DIR.exists():
        for d in sorted(INTERVIEW_DIR.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            readme = d / "README.md"
            if readme.exists():
                all_chunks.extend(chunk_interview_readme(readme))

    # PyTorch lessons
    if PYTORCH_DIR.exists():
        for d in sorted(PYTORCH_DIR.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            readme = d / "README.md"
            if readme.exists():
                all_chunks.extend(chunk_pytorch_readme(readme))
            # Python files
            for py_file in sorted(d.glob("*.py")):
                if py_file.name == "__init__.py":
                    continue
                all_chunks.extend(chunk_python_file(py_file))

    # Enrich all chunks with context prefix
    for chunk in all_chunks:
        meta = {
            "source_type": chunk.source_type,
            "category": chunk.category,
            "question_id": chunk.question_id,
            "question_text": chunk.question_text,
            "section": chunk.section,
            "lesson_number": chunk.lesson_number,
            "lesson_title": chunk.lesson_title,
            "file_path": chunk.file_path,
            "function_name": chunk.function_name,
        }
        chunk.enriched_text = add_context_prefix(chunk.text, meta)

    return all_chunks
