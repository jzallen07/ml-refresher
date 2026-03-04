from __future__ import annotations

import asyncio
import sys


def main():
    from agent.harness import run_session, DEFAULT_MODEL

    if len(sys.argv) < 3:
        print(f"Usage: python -m agent <teacher|interviewer> <topic> [model]")
        print(f"  Default model: {DEFAULT_MODEL}")
        sys.exit(1)

    mode = sys.argv[1]
    topic = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL

    asyncio.run(run_session(mode, topic, model))


main()
