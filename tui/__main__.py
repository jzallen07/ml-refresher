import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tui.app import MLRefresherApp


def main():
    if len(sys.argv) < 3:
        # No mode/topic — launch with welcome screen
        model = sys.argv[1] if len(sys.argv) > 1 else None
        MLRefresherApp(model=model).run()
        return

    mode = sys.argv[1]
    topic = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else None

    if mode not in ("teacher", "interviewer"):
        print(f"Unknown mode: {mode}. Use 'teacher' or 'interviewer'.")
        sys.exit(1)

    MLRefresherApp(mode=mode, topic=topic, model=model).run()


main()
