"""Fixed post-processing stage that converts a recovered candidate to FrontierOR JSON."""

from __future__ import annotations

import argparse
import json
import time

from . import config

FORMATTER_PROMPT = """
Convert the recovered candidate JSON files into solution.json matching solution_schema.json exactly.
Read problem.md and instance.json when field meanings or identifiers are needed. Preserve every
value from the selected candidate. If candidate_solutions contains multiple checkpoints, use the
problem's objective direction and recorded objective values to select the best complete one. Do not
optimize, rerun a solver, invent missing decisions, or claim feasibility. Write only solution.json,
then return a short status message.
""".strip()


def _validate(instance: object, schema: dict) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - deployment precondition
        raise RuntimeError("jsonschema is required for formatter validation") from exc
    jsonschema.validate(instance=instance, schema=schema)


def format_candidate(paper_id: str, model_name: str) -> dict:
    workspace = config.WORKSPACE_ROOT / paper_id
    candidate = workspace / "candidate_solution.json"
    started = time.time()
    if not candidate.is_file():
        return {"outcome": "no_candidate", "wall_s": 0.0}

    required = [workspace / "problem.md", workspace / "solution_schema.json"]
    if any(not path.is_file() for path in required):
        return {
            "outcome": "formatter_failed",
            "error": "formatter inputs were not prepared by the supervisor",
            "wall_s": round(time.time() - started, 1),
        }

    config.ensure_import_path()
    config.configure_llm_env(force=True)
    from apps.operations_research.model_utils import build_model
    from general_tools.file_editing.file_editing_tools import (
        CreateFileWithContent,
        ListDir,
        SeeFile,
    )
    from src.agents import CodeAgent

    agent = CodeAgent(
        tools=[ListDir(str(workspace)), SeeFile(str(workspace)), CreateFileWithContent(str(workspace))],
        managed_agents=[],
        additional_authorized_imports=["json"],
        model=build_model(config.litellm_model_id(model_name)),
        max_steps=10,
        name="frontieror_solution_formatter",
        description="Convert an existing candidate to the required JSON schema without solving.",
        stream_outputs=False,
    )
    try:
        agent.run(FORMATTER_PROMPT, reset=True)
        solution_path = workspace / "solution.json"
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
        schema = json.loads((workspace / "solution_schema.json").read_text(encoding="utf-8"))
        _validate(solution, schema)
        return {
            "outcome": "completed",
            "solution": str(solution_path),
            "wall_s": round(time.time() - started, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "outcome": "formatter_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": round(time.time() - started, 1),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(format_candidate(args.problem, args.model), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
