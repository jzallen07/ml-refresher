#!/usr/bin/env python3
"""Evaluate LLM models via OpenRouter for ML Refresher TUI learning and interview modes."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openrouter import OpenRouter

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set in environment or .env")
    sys.exit(1)

client = OpenRouter(api_key=API_KEY)

MODELS = [
    "minimax/minimax-m2.5",
    "moonshotai/kimi-k2.5",
    "moonshotai/kimi-k2-0905",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.6",
    "z-ai/glm-5",
]

# Load content
ROOT = Path(__file__).resolve().parent.parent
LESSON_PATH = ROOT / "pytorch_refresher" / "05_tensor_gradients" / "README.md"
INTERVIEW_PATH = ROOT / "interview_questions" / "02_attention_mechanisms" / "README.md"

lesson_content = LESSON_PATH.read_text()
interview_content = INTERVIEW_PATH.read_text()

OUTPUT_FILE = Path(__file__).resolve().parent / "eval_results.txt"


def chat(model: str, messages: list[dict]) -> str:
    """Send a chat completion request and return the response text."""
    try:
        resp = client.chat.send(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {e}"


def run_learning_session(model: str) -> list[dict]:
    """Run a 2-exchange learning session about PyTorch gradients."""
    system_msg = (
        "You are an ML tutor. The student is learning about PyTorch gradients.\n\n"
        "Here is the lesson content for context:\n\n" + lesson_content
    )
    messages = [{"role": "system", "content": system_msg}]

    # Turn 1
    user1 = "Can you explain what requires_grad=True does and why we need it? Give me a simple example."
    messages.append({"role": "user", "content": user1})
    reply1 = chat(model, messages)
    messages.append({"role": "assistant", "content": reply1})

    # Turn 2
    user2 = (
        "If I do `z = x**2 + 3*y` and call `z.backward()`, what would `x.grad` be "
        "if x=2? Walk me through it."
    )
    messages.append({"role": "user", "content": user2})
    reply2 = chat(model, messages)
    messages.append({"role": "assistant", "content": reply2})

    return [
        ("User", user1),
        ("Assistant", reply1),
        ("User", user2),
        ("Assistant", reply2),
    ]


def run_interview_session(model: str) -> list[dict]:
    """Run a 2-exchange interview coaching session about attention mechanisms."""
    system_msg = (
        "You are an ML interview coach. Help the student practice answering "
        "interview questions. Ask the question, then evaluate their answer."
    )
    messages = [{"role": "system", "content": system_msg}]

    # Turn 1
    user1 = "Ask me: How does the attention mechanism function in transformer models?"
    messages.append({"role": "user", "content": user1})
    reply1 = chat(model, messages)
    messages.append({"role": "assistant", "content": reply1})

    # Turn 2
    user2 = (
        "The attention mechanism uses Query, Key, and Value vectors. You compute "
        "QK^T, scale by sqrt(d_k), apply softmax to get weights, then multiply by V. "
        "This lets the model focus on relevant parts of the input for each token."
    )
    messages.append({"role": "user", "content": user2})
    reply2 = chat(model, messages)
    messages.append({"role": "assistant", "content": reply2})

    return [
        ("User", user1),
        ("Assistant", reply1),
        ("User", user2),
        ("Assistant", reply2),
    ]


def format_transcript(exchanges: list[tuple[str, str]]) -> str:
    """Format a list of (role, content) exchanges into readable text."""
    lines = []
    for role, content in exchanges:
        lines.append(f"  [{role}]")
        for line in content.split("\n"):
            lines.append(f"    {line}")
        lines.append("")
    return "\n".join(lines)


def main():
    results = []

    for model in MODELS:
        header = f"{'=' * 70}\nMODEL: {model}\n{'=' * 70}"
        print(header)
        results.append(header)

        # Learning session
        print(f"\n--- Learning Session ---")
        results.append("\n--- Learning Session ---")
        learning = run_learning_session(model)
        transcript = format_transcript(learning)
        print(transcript)
        results.append(transcript)

        # Interview session
        print(f"\n--- Interview Session ---")
        results.append("\n--- Interview Session ---")
        interview = run_interview_session(model)
        transcript = format_transcript(interview)
        print(transcript)
        results.append(transcript)

        print()
        results.append("")

    # Save to file
    OUTPUT_FILE.write_text("\n".join(results))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
