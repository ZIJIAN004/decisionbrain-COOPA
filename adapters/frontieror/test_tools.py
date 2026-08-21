"""Check the bounded see_file and list_dir against realistically shaped files.

Run with:  python -m adapters.frontieror.test_tools

Unbounded reading is the one capability gap the adaptation had to close: an
instance whose median size is 206 KB cannot be returned whole, and a JSON file
written without newlines cannot be paged through either, so the tool has to say
so instead of silently handing back a fragment. These checks are about that
contract, not about the agents.

smolagents only provides the Tool base class here, so a stub stands in when it
is not installed; the reading logic under test is the same either way.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

try:  # pragma: no cover - depends on the environment, not on the code
    import smolagents.tools  # noqa: F401
except ImportError:
    stub = types.ModuleType("smolagents")
    tools_stub = types.ModuleType("smolagents.tools")

    class _Tool:  # minimal stand-in for smolagents.tools.Tool
        def __init__(self, *args, **kwargs):
            pass

    tools_stub.Tool = _Tool
    stub.tools = tools_stub
    sys.modules.setdefault("smolagents", stub)
    sys.modules.setdefault("smolagents.tools", tools_stub)

from general_tools.file_editing.file_editing_tools import (  # noqa: E402
    MAX_OUTPUT_CHARS,
    MAX_READ_BYTES,
    TRUNCATION_NOTICE,
    ListDir,
    LoadObjectFromPythonFile,
    SeeFile,
)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="coopa_tools_"))

    # A single-line JSON of roughly the median FrontierOR instance size.
    (work / "instance.json").write_text(
        json.dumps({"demands": list(range(20000)), "Q": 120}), encoding="utf-8"
    )
    (work / "notes.md").write_text(
        "\n".join(f"line {i}" for i in range(1, 501)), encoding="utf-8"
    )
    (work / "huge.json").write_text("x" * (MAX_READ_BYTES + 10), encoding="utf-8")
    (work / "small.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    see = SeeFile(str(work))
    listing = ListDir(str(work))

    print("see_file:")
    out = see.forward("small.txt")
    check("small file returns numbered lines", out == "1:alpha\n2:beta\n3:gamma\n", repr(out))

    out = see.forward("nope.txt")
    check("missing file reports clearly", "does not exist" in out, repr(out))

    out = see.forward("../escape.txt")
    check("traversal refused", "not allowed" in out, repr(out))

    out = see.forward("huge.json")
    check(
        "oversize whole-file read refused",
        "larger than the" in out and "line range" in out,
        repr(out)[:120],
    )

    out = see.forward("huge.json", start_line=1, line_count=1)
    check(
        "oversize file still readable by range",
        len(out) <= MAX_OUTPUT_CHARS and out.startswith("1:"),
    )

    out = see.forward("instance.json")
    check("single-line JSON is capped", len(out) <= MAX_OUTPUT_CHARS, f"len={len(out)}")
    check("single-line JSON marks truncation", TRUNCATION_NOTICE.strip() in out)
    check(
        "truncation keeps head and tail",
        out.startswith('1:{"demands"') and out.rstrip().endswith("}"),
        repr(out[-40:]),
    )

    out = see.forward("notes.md", start_line=10, line_count=3)
    check("line range is exact", out == "10:line 10\n11:line 11\n12:line 12\n", repr(out))

    out = see.forward("notes.md", start_line=9999)
    check("range past end explains itself", "fewer than" in out, repr(out))

    out = see.forward("notes.md", max_chars=50)
    check("max_chars lowers the cap", len(out) <= 50, f"len={len(out)}")

    out = see.forward("notes.md", max_chars=10 ** 9)
    check("max_chars cannot raise the cap", len(out) <= MAX_OUTPUT_CHARS, f"len={len(out)}")

    out = see.forward("instance.json", start_line=1, line_count=1, max_chars=200)
    check("tight cap on a huge line still returns", len(out) <= 200)

    print("list_dir:")
    out = listing.forward(".")
    check("sizes are shown", "instance.json  (" in out, repr(out))
    check("oversize file is flagged", "too large to view in full" in out, repr(out))
    check(
        "all files listed",
        all(name in out for name in ("small.txt", "notes.md", "huge.json")),
    )

    print("bounded solver execution:")
    (work / "solver.py").write_text(
        "def solve_problem():\n    return {'objective': 7, 'x': [1, 0]}\n",
        encoding="utf-8",
    )
    os.environ["ADAPTER_FRONTIEROR_MODE"] = "1"
    os.environ["ADAPTER_SOLVER_TIMEOUT"] = "5"
    result = LoadObjectFromPythonFile(str(work)).forward("solver.py", "solve_problem")()
    check("solver result is returned", result["objective"] == 7)
    check(
        "solver result is checkpointed atomically",
        json.loads((work / "candidate_solution.json").read_text(encoding="utf-8")) == result,
    )
    os.environ.pop("ADAPTER_FRONTIEROR_MODE", None)
    os.environ.pop("ADAPTER_SOLVER_TIMEOUT", None)

    print()
    if failures:
        print(f"{len(failures)} failing: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
