"""Run one command under a memory cap and a wall clock, then report what it cost.

Invoked as a subprocess, one per task. Running it in its own process is what
makes the accounting per-task: getrusage(RUSAGE_CHILDREN) reports a high-water
mark across *all* children of the calling process, so a single scheduler process
supervising several tasks at once could only report their maximum, not each
task's own peak.

Prints one JSON object on stdout. Everything the supervised command writes goes
to the log file, not here.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

from .sandbox import build_command

# Each task gets its own CPU-limited scope beneath the shared FrontierOR slice.
# MemoryMax is enforced by that parent slice, so all concurrent Agent, solver,
# and Formatter descendants contribute to one aggregate limit.
SCOPE_CMD = ["systemd-run", "--user", "--scope", "-q"]


def run_capped(
    command: list[str], slice_name: str, cpu_cores: int, timeout_s: int, cwd: Path, log_path: Path,
    sandbox: dict[str, Path] | None = None,
) -> dict:
    if sandbox:
        command = build_command(command, repo=cwd, **sandbox)
    argv = SCOPE_CMD + [
        "--slice", slice_name,
        "-p", f"CPUQuota={cpu_cores * 100}%",
        *command,
    ]
    started = time.time()
    timed_out = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            argv, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            returncode = proc.wait()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    return {
        "returncode": returncode,
        "outcome": classify(returncode, timed_out),
        "peak_rss_gb": round(usage.ru_maxrss / 1e6, 3),
        "cpu_user_s": round(usage.ru_utime, 1),
        "cpu_sys_s": round(usage.ru_stime, 1),
        "wall_s": round(time.time() - started, 1),
        "memory_slice": slice_name,
        "cpu_quota_cores": cpu_cores,
        "timeout_s": timeout_s,
        "log": str(log_path),
    }


def classify(returncode: int, timed_out: bool) -> str:
    """Separate resource limits from the baseline's own failures, so a task
    killed by our budget is never scored as COOPA getting it wrong."""
    if timed_out:
        return "task_timeout"
    if returncode == -9:
        return "memory_exceeded"
    if returncode == 0:
        return "completed"
    return "coopa_failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", required=True)
    parser.add_argument("--cpu-cores", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--sandbox-workspace", type=Path)
    parser.add_argument("--sandbox-python-env", type=Path)
    parser.add_argument("--sandbox-index", type=Path)
    parser.add_argument("--sandbox-instance", type=Path)
    parser.add_argument("--sandbox-problem", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")

    sandbox = None
    values = [args.sandbox_workspace, args.sandbox_python_env, args.sandbox_index,
              args.sandbox_instance, args.sandbox_problem]
    if any(values):
        if not all(values):
            parser.error("all sandbox paths are required together")
        sandbox = {
            "workspace": args.sandbox_workspace,
            "python_env": args.sandbox_python_env,
            "index": args.sandbox_index,
            "instance": args.sandbox_instance,
            "problem": args.sandbox_problem,
        }
    record = run_capped(
        command, args.slice, args.cpu_cores, args.timeout, args.cwd, args.log, sandbox
    )
    json.dump(record, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
