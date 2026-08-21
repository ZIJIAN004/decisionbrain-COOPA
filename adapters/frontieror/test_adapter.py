"""Check the parts of the adaptation that need no model and no network.

Run with:  python -m adapters.frontieror.test_adapter

What is checked here is the shape of what reaches the agents: that the instance
is staged where their tools can reach it and nowhere else, that the statement
survives into the manager's prompt, and that the retargeted optimizer prompts no
longer tell an agent to write the data out by hand. Whether a model then solves
the problem is not something a test can answer.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


def build_suite(root: Path) -> Path:
    """A miniature FrontierOR layout: statement in one tree, instance in another.

    The reference solution is placed beside the instance exactly as the real
    suite places it, so staging can be checked to leave it behind.
    """
    problem_dir = root / "problems" / "demo2026" / "input"
    problem_dir.mkdir(parents=True)
    (problem_dir / "problem.md").write_text(
        "# Demo\n\nA plant ships to customers subject to capacity.\n", encoding="utf-8"
    )

    instance_dir = root / "instances" / "demo2026" / "instance"
    instance_dir.mkdir(parents=True)
    (instance_dir / "large_instance_0.json").write_text(
        json.dumps({"Q": 120, "demands": [3, 4, 5]}), encoding="utf-8"
    )
    solution_dir = root / "instances" / "demo2026" / "gurobi_solution"
    solution_dir.mkdir(parents=True)
    (solution_dir / "solution.json").write_text('{"objective": 42}', encoding="utf-8")

    index = root / "index.json"
    index.write_text(
        json.dumps(
            {
                "cases": {
                    "demo2026": {
                        "instance_index": 0,
                        "instance_bytes": 34,
                        "formulation_type": "MILP",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return index


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="coopa_adapter_"))
    index = build_suite(root)

    os.environ["FRONTIEROR_INDEX"] = str(index)
    os.environ["FRONTIEROR_INSTANCE_ROOT"] = str(root / "instances")
    os.environ["FRONTIEROR_PROBLEM_ROOT"] = str(root / "problems")
    os.environ["ADAPTER_RUNS_ROOT"] = str(root / "runs")
    os.environ["ADAPTER_WORKSPACE_ROOT"] = str(root / "runs" / "workspaces")

    from . import config

    importlib.reload(config)

    print("staging:")
    workspace = config.stage_workspace("demo2026")
    staged = sorted(p.name for p in workspace.iterdir())
    check("instance is staged under the working directory", staged == ["instance.json"], str(staged))
    check(
        "staged instance matches the source",
        json.loads((workspace / "instance.json").read_text(encoding="utf-8"))["Q"] == 120,
    )
    check(
        "the reference solution is not staged",
        not (workspace / "gurobi_solution").exists() and "solution.json" not in staged,
    )
    check(
        "staging is idempotent",
        sorted(p.name for p in config.stage_workspace("demo2026").iterdir()) == ["instance.json"],
    )

    print("prompt:")
    question = config.build_question("demo2026")
    check("statement is carried verbatim", "A plant ships to customers subject to capacity." in question)
    check("pointer names the staged file", "instance.json" in question)
    check("pointer asks for a reference, not a number", "instance.json:<field>" in question)
    check(
        "no real path leaks into the prompt",
        str(root) not in question and "FrontierOR" not in question,
    )

    from apps.operations_research.formulation_utils import format_formulation_prompt

    from .run_one import build_prompt

    formulation = types.SimpleNamespace(
        parameters=[
            types.SimpleNamespace(
                name="capacity",
                data_type="str",
                description="Vehicle capacity",
                value="instance.json:Q",
                units=None,
            )
        ],
        variables=[],
        objective=types.SimpleNamespace(
            sense="minimize", description="cost", expression="sum(c*x)", variables_involved=["x"]
        ),
        constraints=[],
    )
    prompt = build_prompt(question, formulation, format_formulation_prompt)
    check("manager prompt keeps the statement", "A plant ships to customers" in prompt)
    check("manager prompt keeps the formulation", "## PARAMETERS:" in prompt)
    check("a file reference reaches the manager as a reference", "instance.json:Q" in prompt)
    check(
        "the manager is no longer told to write a parameters file",
        "saving parameters to JSON" not in prompt,
    )

    print("model ids:")
    check("a bare name is qualified", config.litellm_model_id("deepseek-v4-flash") == "openai/deepseek-v4-flash")
    check("a qualified name is left alone", config.litellm_model_id("gemini/x") == "gemini/x")

    print("retargeted optimizer prompts:")
    prompts_dir = config.REPO_ROOT / "apps" / "operations_research" / "or_agents" / "prompts"
    for path in sorted(prompts_dir.glob("*optimizer.yaml")):
        text = path.read_text(encoding="utf-8")
        check(f"{path.name}: no parameters.json survives", "parameters.json" not in text)
        check(f"{path.name}: points at the staged instance", "instance.json" in text)
        check(
            f"{path.name}: forbids retyping the data",
            "MUST NOT retype" in text,
        )
        check(
            f"{path.name}: first step is inspection",
            "(1) Inspect instance.json," in text,
        )

    print()
    if failures:
        print(f"{len(failures)} failing: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
