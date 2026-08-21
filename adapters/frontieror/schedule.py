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
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config
from .sandbox import build_command


def _run_task(
    paper_id: str, model: str, slice_name: str, cpu_cores: int, run_dir: Path
) -> dict:
    """Spawn wrapper.py so this task's resource usage is measured on its own."""
    workspace = config.stage_workspace(paper_id)
    case = config.load_cases()[paper_id]
    python_env = Path(sys.executable).resolve().parent.parent
    task_log = run_dir / "logs" / f"{paper_id}.json"
    argv = [
        sys.executable, "-m", "adapters.frontieror.wrapper",
        "--slice", slice_name,
        "--cpu-cores", str(cpu_cores),
        "--timeout", str(config.TASK_TIMEOUT_SECONDS),
        "--cwd", str(config.REPO_ROOT),
        "--log", str(run_dir / "logs" / f"{paper_id}.log"),
        "--sandbox-workspace", str(workspace),
        "--sandbox-python-env", str(python_env),
        "--sandbox-index", str(config.INDEX_JSON),
        "--sandbox-instance", str(config.instance_path(paper_id, case["instance_index"])),
        "--sandbox-problem", str(config.problem_md_path(paper_id)),
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
        usage = {
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

    # Formatting is a mandatory adapter stage after the 7200-second COOPA budget.
    formatter_started = time.time()
    try:
        shutil.copy2(config.problem_md_path(paper_id), workspace / "problem.md")
        shutil.copy2(config.solution_schema_path(paper_id), workspace / "solution_schema.json")
        formatter_command = build_command(
            [
                sys.executable,
                "-m",
                "adapters.frontieror.formatter",
                "--problem",
                paper_id,
                "--model",
                model,
            ],
            repo=config.REPO_ROOT,
            python_env=python_env,
            workspace=workspace,
            index=config.INDEX_JSON,
            instance=config.instance_path(paper_id, case["instance_index"]),
            problem=config.problem_md_path(paper_id),
        )
        formatter = subprocess.run(
            ["systemd-run", "--user", "--scope", "-q", "--slice", slice_name,
             *formatter_command],
            cwd=str(config.REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=config.FORMATTER_TIMEOUT_SECONDS,
            check=False,
        )
        record["formatter"] = json.loads(formatter.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        record["formatter"] = {"outcome": "formatter_timeout"}
    except Exception as exc:  # noqa: BLE001 - formatting failure is a task result
        record["formatter"] = {
            "outcome": "formatter_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    record["formatter_wall_s"] = round(time.time() - formatter_started, 1)
    record["total_wall_s"] = round(float(record.get("wall_s") or 0) + record["formatter_wall_s"], 1)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=config.JOBS)
    parser.add_argument(
        "--memory-gb", type=int, default=config.TOTAL_MEMORY_GB,
        help="aggregate systemd slice MemoryMax for all concurrent tasks (default: 100)",
    )
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--only", nargs="*", help="restrict to these paper_ids")
    parser.add_argument("--run-dir", type=Path, default=None, help="defaults to a new runs/ folder")
    args = parser.parse_args()

    if not 1 <= args.jobs <= config.TOTAL_CPU_CORES:
        parser.error(f"--jobs must be between 1 and {config.TOTAL_CPU_CORES}")

    config.configure_llm_env()
    problems = config.check_preconditions()
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2

    if args.memory_gb < 1:
        parser.error("--memory-gb must be at least 1")
    cpu_cores = config.TOTAL_CPU_CORES // args.jobs

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
    slice_name = "coopa-frontieror.slice"
    slice_unit = Path(__file__).with_name(slice_name).resolve()
    subprocess.run(
        ["systemctl", "--user", "link", "--runtime", str(slice_unit)], check=False
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "start", slice_name], check=True)
    subprocess.run(
        ["systemctl", "--user", "set-property", "--runtime", slice_name,
         f"MemoryMax={args.memory_gb}G", "MemorySwapMax=0"],
        check=True,
    )

    refinement = (
        f"on x{config.MAX_REFINEMENT_ITERATIONS}" if config.USE_ITERATIVE_REFINEMENT else "off"
    )
    print(f"{len(ordered)} cases | jobs={args.jobs} | {args.memory_gb} GB aggregate "
          f"| {cpu_cores} CPU cores per task (system quota) "
          f"| {config.TASK_TIMEOUT_SECONDS}s agent wall | {config.SOLVER_TIMEOUT_SECONDS}s solver "
          f"| model={args.model} "
          f"| iterative refinement {refinement}", flush=True)
    print(f"run dir: {run_dir}", flush=True)

    started = time.time()
    with report_path.open("w", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(_run_task, pid, args.model, slice_name, cpu_cores, run_dir): pid
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
