"""Run COOPA over the FrontierOR cases.

Each task runs as its own process under a memory cap and a wall clock, so a task
that exhausts either is killed on its own without touching the host or the other
tasks, and its cost is measured separately.

Everything a run produces goes under runs/<tag>-<timestamp>/:

    report.jsonl          one line per task: result plus resource usage
    logs/<paper_id>.json  the task's own record, its prompt and its formulation
    logs/<paper_id>.log   whatever the task process wrote to stdout and stderr

The agents' own working directories, holding the staged instance and whatever
code they wrote, stay under the workspace root so they can be inspected after a
run without being mixed into the report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

from . import config


def _run_task(paper_id: str, model: str, mem_gb: int, run_dir: Path) -> dict:
    """Spawn wrapper.py so this task's resource usage is measured on its own."""
    task_log = run_dir / "logs" / f"{paper_id}.json"
    argv = [
        sys.executable, "-m", "adapters.frontieror.wrapper",
        "--mem-gb", str(mem_gb),
        "--timeout", str(config.TASK_TIMEOUT_SECONDS),
        "--cwd", str(config.REPO_ROOT),
        "--log", str(run_dir / "logs" / f"{paper_id}.log"),
        "--",
        sys.executable, "-m", "adapters.frontieror.run_one",
        "--problem", paper_id,
        "--model", model,
        "--log", str(task_log),
    ]
    proc = subprocess.run(
        argv, cwd=str(config.REPO_ROOT), capture_output=True, text=True, check=False
    )

    try:
        usage = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {
            "paper_id": paper_id,
            "outcome": "wrapper_failed",
            "returncode": proc.returncode,
            "stderr": proc.stderr[-2000:],
        }

    record = {"paper_id": paper_id, "model": model, **usage}
    # The objective is read from the task's own record rather than parsed out of
    # its stdout, so a task that died mid-run simply has no result field.
    if task_log.is_file():
        try:
            record["result"] = json.loads(task_log.read_text(encoding="utf-8"))["result"]
        except (ValueError, KeyError):
            record["result"] = None
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=config.JOBS)
    parser.add_argument("--budget-gb", type=int, default=config.TOTAL_BUDGET_GB)
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--only", nargs="*", help="restrict to these paper_ids")
    parser.add_argument("--run-dir", type=Path, default=None, help="defaults to a new runs/ folder")
    args = parser.parse_args()

    config.configure_llm_env()
    problems = config.check_preconditions()
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2

    # One knob. The per-task cap is the budget divided by how many run at once,
    # so the run as a whole cannot exceed the budget however tasks are scheduled.
    mem_gb = max(1, args.budget_gb // args.jobs)

    cases = config.load_cases()
    if args.only:
        cases = {k: v for k, v in cases.items() if k in set(args.only)}
        if not cases:
            parser.error("no matching paper_ids in the suite index")

    # Smallest first: the largest instances are the likeliest to hit the cap, and
    # finishing the cheap cases first makes a partial run useful.
    ordered = sorted(cases.items(), key=lambda kv: kv[1]["instance_bytes"])

    run_dir = args.run_dir or config.new_run_dir()
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.jsonl"

    refinement = (
        f"on x{config.MAX_REFINEMENT_ITERATIONS}" if config.USE_ITERATIVE_REFINEMENT else "off"
    )
    print(f"{len(ordered)} cases | jobs={args.jobs} | {mem_gb} GB per task "
          f"| {config.TASK_TIMEOUT_SECONDS}s wall | model={args.model} "
          f"| iterative refinement {refinement}", flush=True)
    print(f"run dir: {run_dir}", flush=True)

    started = time.time()
    with report_path.open("w", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(_run_task, pid, args.model, mem_gb, run_dir): pid
                for pid, _ in ordered
            }
            for future in concurrent.futures.as_completed(futures):
                paper_id = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001
                    record = {"paper_id": paper_id, "outcome": "scheduler_failed",
                              "error": f"{type(exc).__name__}: {exc}"}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                result = record.get("result") or {}
                print(
                    f"  {paper_id:<20} {record.get('outcome', '?'):<16} "
                    f"obj={result.get('objective')} "
                    f"phase0={result.get('formulation_s')}s "
                    f"peak={record.get('peak_rss_gb')}GB "
                    f"{record.get('wall_s')}s",
                    flush=True,
                )

    print(f"\ndone in {time.time() - started:.0f}s -> {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
