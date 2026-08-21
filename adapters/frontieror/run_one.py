"""Run COOPA on one FrontierOR task.

The method is untouched: the formulation step, the manager's routing and the
four optimizer agents run exactly as they do upstream, with the iterative
confidence-based selection the paper describes turned on. Three things differ,
all of them about how the problem reaches the agents:

1. The statement is passed on to the manager as well as to the formulation step.
   Upstream discards it (formulation_utils.format_formulation_prompt renders the
   formulation only), which costs nothing when the statement is a few hundred
   characters carrying its own numbers, and costs everything when the statement
   is the only place the problem's semantics exist.

2. The instance file is staged in the working directory, so the agents reach the
   data the way their own tools reach anything: by reading a file they are
   confined to. Their prompts have been retargeted to load it instead of
   retyping the numbers out of the prompt.

3. Scoring is left to FrontierOR's checker. Upstream reads the first number out
   of the agent's final answer and compares it to a gold value with a fixed
   tolerance; that number is recorded here, but so is the whole answer, and the
   verdict is not taken from it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from . import config

# The same expression upstream uses (run_exp_with_kb_full_multiprocess.py), kept
# so the recorded number is the number COOPA itself would have been scored on.
FIRST_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

STATEMENT_HEADER = (
    "## PROBLEM STATEMENT (verbatim, this is the problem to solve):\n\n"
)
STATEMENT_FOOTER = (
    "\n\n---\n\nA structured formulation of the statement above follows. Pass BOTH the "
    "statement and the formulation on to the optimizer agent you delegate to: the "
    "formulation is a summary, and the statement is the source.\n"
)


def build_prompt(question: str, formulation, format_formulation_prompt) -> str:
    execution_contract = (
        f"\n\nEvery individual solver execution has a hard limit of "
        f"{config.SOLVER_TIMEOUT_SECONDS} seconds. Set the solver's native time limit to "
        "that value and return the best incumbent when the limit is reached."
    )
    return (
        STATEMENT_HEADER
        + question
        + STATEMENT_FOOTER
        + format_formulation_prompt(formulation)
        + execution_contract
    )


def extract_formulation_phase(question: str, model_id: str) -> tuple[object, dict]:
    """Phase 0, with the paper's iterative confidence-based selection enabled.

    Returns the selected formulation and a record of how it was chosen, so a run
    can be inspected afterwards without re-running anything.
    """
    from apps.operations_research.or_agents.formulation import (
        create_instructor_client,
        extract_formulation,
    )
    from apps.operations_research.or_agents.iterative_formulation import (
        extract_formulation_with_refinement,
    )

    if not config.USE_ITERATIVE_REFINEMENT:
        client = create_instructor_client(model_name=model_id, timeout=90.0)
        formulation = extract_formulation(problem_text=question, client=client, model=model_id)
        return formulation, {"iterative_refinement": False}

    formulation, evaluation, selected, history = extract_formulation_with_refinement(
        problem_text=question,
        max_iterations=config.MAX_REFINEMENT_ITERATIONS,
        formulation_model=model_id,
        evaluation_model=model_id,
        verbose=False,
        return_history=True,
    )
    return formulation, {
        "iterative_refinement": True,
        "max_iterations": config.MAX_REFINEMENT_ITERATIONS,
        "selected_iteration": selected,
        "selected_evaluation": evaluation.model_dump(),
        "candidates": [
            {
                "iteration": entry["iteration"],
                "overall_confidence": entry["overall_confidence"],
                "min_confidence": entry["min_confidence"],
            }
            for entry in history
        ],
    }


def run(paper_id: str, case: dict, model_name: str, log_path: Path | None = None) -> dict:
    os.environ["ADAPTER_FRONTIEROR_MODE"] = "1"
    os.environ.setdefault("ADAPTER_SOLVER_TIMEOUT", str(config.SOLVER_TIMEOUT_SECONDS))
    config.ensure_import_path()
    config.configure_llm_env()

    from apps.operations_research.formulation_utils import format_formulation_prompt
    from apps.operations_research.run import create_manager_agent

    # Importing that module pulls in the four optimizer agents, each of which
    # calls load_dotenv(override=True) and so replaces the environment set
    # above with whatever a stray .env holds. Settling it afterwards is what
    # keeps the endpoint ours and the search key inert.
    config.configure_llm_env(force=True)

    model_id = config.litellm_model_id(model_name)
    workspace = (
        config.WORKSPACE_ROOT / paper_id
        if os.environ.get("ADAPTER_WORKSPACE_PRESTAGED") == "1"
        else config.stage_workspace(paper_id)
    )
    question = config.build_question(paper_id)

    started = time.time()
    formulation, selection = extract_formulation_phase(question, model_id)
    formulation_s = round(time.time() - started, 1)

    # Upstream leaves the formulation in the working directory, where the agents
    # can read it back; that is kept.
    (workspace / "formulation.json").write_text(
        json.dumps(formulation.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    prompt = build_prompt(question, formulation, format_formulation_prompt)
    manager = create_manager_agent(
        model_id=model_id, working_directory=str(workspace), allow_web=False
    )

    solve_started = time.time()
    error = None
    try:
        answer = manager.run(prompt, reset=True)
    except Exception as exc:  # noqa: BLE001 - a failed task is a result, not a crash
        answer = None
        error = f"{type(exc).__name__}: {exc}"

    match = FIRST_NUMBER.search(str(answer)) if answer is not None else None

    record = {
        "paper_id": paper_id,
        "model": model_id,
        "objective": float(match.group()) if match else None,
        "answer": str(answer)[:8000] if answer is not None else None,
        "error": error,
        "formulation_s": formulation_s,
        "solve_s": round(time.time() - solve_started, 1),
        "wall_s": round(time.time() - started, 1),
        "instance_bytes": case["instance_bytes"],
        "formulation_type": case["formulation_type"],
        "selection": selection,
        "web_search": "disabled",
        "workspace": str(workspace),
    }
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "result": record,
                    "prompt": prompt,
                    "formulation": formulation.model_dump(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, help="paper_id from the suite index")
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="defaults to a new single-<timestamp>/logs/<problem>.json under the runs root",
    )
    args = parser.parse_args()

    config.configure_llm_env()
    problems = config.check_preconditions()
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2

    cases = config.load_cases()
    if args.problem not in cases:
        parser.error(f"unknown problem {args.problem!r}; not in {config.INDEX_JSON}")

    log_path = args.log or (config.new_run_dir("single") / "logs" / f"{args.problem}.json")
    print(f"log: {log_path}", file=sys.stderr, flush=True)

    result = run(args.problem, cases[args.problem], args.model, log_path)
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
