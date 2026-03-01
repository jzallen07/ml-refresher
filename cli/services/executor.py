from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TIMEOUT = 30


def run_code(
    code: str | None = None,
    file: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    if not code and not file:
        return {"success": False, "error": "No code or file provided"}

    tmp_path = None
    try:
        if code:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir=REPO_ROOT
            )
            tmp.write(code)
            tmp.close()
            target = tmp.name
            tmp_path = tmp.name
        else:
            target = str(Path(file).resolve())
            if not Path(target).exists():
                return {"success": False, "error": f"File not found: {file}"}

        start = time.time()
        result = subprocess.run(
            ["uv", "run", "python", target],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        elapsed = time.time() - start

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "elapsed_seconds": round(elapsed, 2),
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Execution timed out after {timeout}s",
            "exit_code": -1,
        }
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
