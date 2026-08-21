"""Execute one generated solver function and checkpoint any returned candidate."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    request_path, response_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        spec = importlib.util.spec_from_file_location("coopa_generated_solver", request["file"])
        if spec is None or spec.loader is None:
            raise ImportError("could not load generated solver")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = getattr(module, request["object"])
        result = target(*request.get("args", []), **request.get("kwargs", {}))
        candidate_dir = request_path.parent / "candidate_solutions"
        candidate_dir.mkdir(exist_ok=True)
        atomic_json(candidate_dir / f"candidate-{time.time_ns()}.json", result)
        atomic_json(request_path.parent / "candidate_solution.json", result)
        payload = {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    atomic_json(response_path, payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
