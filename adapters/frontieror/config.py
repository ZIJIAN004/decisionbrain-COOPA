"""Paths, staging, model wiring and the prompt block for the FrontierOR adaptation.

COOPA reads a problem as one string with the numbers written into the sentences,
and its optimizer agents are scripted to retype those numbers into a parameters
file before modelling. FrontierOR supplies the numbers as a separate file whose
median size is 206 KB, so neither step survives contact with it: the statement
cannot carry the data, and no agent can retype it.

What this module changes is therefore only where the data lives and what the
agents are told about it. The formulation step, the manager's routing and the
four optimizer agents are the method being measured and are left alone.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

INDEX_JSON = Path(
    os.environ.get(
        "FRONTIEROR_INDEX",
        "/home/bhz/Decision Brain/benchmarks/frontieror-large-all/index.json",
    )
)
INSTANCE_ROOT = Path(os.environ.get("FRONTIEROR_INSTANCE_ROOT", "/home/bhz/FrontierOR_all"))
PROBLEM_ROOT = Path(
    os.environ.get("FRONTIEROR_PROBLEM_ROOT", "/home/bhz/Decision Brain/benchmarks/frontieror")
)
# Run output lives beside the baseline repositories rather than inside this one,
# so every baseline's results sit together under one parent and nothing a run
# produces is mixed into the checkout.
RUNS_ROOT = Path(os.environ.get("ADAPTER_RUNS_ROOT", "/home/bhz/baselines/coopa-runs"))
WORKSPACE_ROOT = Path(os.environ.get("ADAPTER_WORKSPACE_ROOT", RUNS_ROOT / "workspaces"))

TOTAL_BUDGET_GB = int(os.environ.get("ADAPTER_TOTAL_BUDGET_GB", "100"))
JOBS = int(os.environ.get("ADAPTER_JOBS", "4"))
TASK_TIMEOUT_SECONDS = int(os.environ.get("ADAPTER_TASK_TIMEOUT", "7200"))

# The paper's headline component: several candidate formulations, scored per
# component, selected by the highest minimum score. The released code leaves it
# off, so running the method as published means turning it on here.
USE_ITERATIVE_REFINEMENT = os.environ.get("ADAPTER_ITERATIVE_REFINEMENT", "1") != "0"
MAX_REFINEMENT_ITERATIONS = int(os.environ.get("ADAPTER_REFINEMENT_ITERATIONS", "3"))

INSTANCE_FILENAME = "instance.json"
DEFAULT_MODEL = os.environ.get("LLM_CHAT_MODEL", "deepseek-v4-flash")


def ensure_import_path() -> None:
    """Make the upstream packages importable without installing them.

    apps/, src/ and general_tools/ are all imported as top-level packages from
    the repository root, which is already sys.path[0] under `python -m` from
    there; this covers the case where it is not.
    """
    text = str(REPO_ROOT)
    if text not in sys.path:
        sys.path.insert(0, text)


def litellm_model_id(model: str) -> str:
    """Qualify a bare model name with the provider LiteLLM should route it to.

    build_model and create_instructor_client both hand the name straight to
    LiteLLM, which needs a provider prefix for anything it does not recognise by
    name. An id that already carries one is left as it is.
    """
    if "/" in model:
        return model
    return f"{os.environ.get('ADAPTER_LLM_PROVIDER', 'openai')}/{model}"


def configure_llm_env() -> None:
    """Point LiteLLM at the endpoint this host already uses.

    DecisionBrain's .env stores a full chat-completions URL in LLM_MODEL_URL and
    the key in LLM_API_KEY; LiteLLM's openai provider reads OPENAI_BASE_URL and
    OPENAI_API_KEY and appends the path itself, so the suffix is trimmed rather
    than duplicated. Nothing here ever prints a key.
    """
    url = os.environ.get("ADAPTER_BASE_URL") or os.environ.get("LLM_MODEL_URL")
    if url:
        base = url.removesuffix("/chat/completions").rstrip("/")
        # LiteLLM has read both names across versions, so both are set rather
        # than betting on which one the installed version looks at.
        for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
            os.environ.setdefault(name, base)
    key = os.environ.get("ADAPTER_API_KEY") or os.environ.get("LLM_API_KEY")
    if key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key


def check_preconditions() -> list[str]:
    """Name what is missing before a run starts rather than mid-task.

    The web-browsing agent is built for every optimizer agent, and its Google
    search tool wants SERPER_API_KEY at construction time, so a missing key does
    not fail the search: it fails the whole task before any modelling happens.
    """
    problems = []
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ADAPTER_API_KEY")):
        problems.append("no LLM key: set LLM_API_KEY (or OPENAI_API_KEY)")
    if not os.environ.get("SERPER_API_KEY"):
        problems.append(
            "SERPER_API_KEY is unset: create_web_browsing_agent builds a Serper-backed "
            "search tool for every optimizer agent, so every task fails at construction"
        )
    if not INDEX_JSON.is_file():
        problems.append(f"suite index not found: {INDEX_JSON}")
    return problems


def new_run_dir(tag: str = "frontieror") -> Path:
    """RUNS_ROOT/<tag>-<UTC timestamp>/ with report.jsonl and logs/ inside it."""
    import time

    run_dir = RUNS_ROOT / f"{tag}-{time.strftime('%Y%m%d-%H%M%SZ', time.gmtime())}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def load_cases() -> dict:
    with INDEX_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def instance_path(paper_id: str, instance_index: int) -> Path:
    return INSTANCE_ROOT / paper_id / "instance" / f"large_instance_{instance_index}.json"


def problem_md_path(paper_id: str) -> Path:
    return PROBLEM_ROOT / paper_id / "input" / "problem.md"


def stage_workspace(paper_id: str) -> Path:
    """Build the working directory: the instance and nothing else.

    This is also the directory the agents write solve.py into and the directory
    their file tools are confined to, so the instance has to sit at its root:
    _safe_path refuses absolute paths, and the generated code resolves its own
    location with os.path.dirname(__file__).

    The reference solution lives in a sibling directory of the real instance and
    the hidden checker lives beside the problem statement, so the agent is given
    a copy and is never told where the copy came from.
    """
    workspace = WORKSPACE_ROOT / paper_id
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    case = load_cases()[paper_id]
    # shutil.copy2, matching DecisionBrain's runner (benchmark/runner.py:698).
    shutil.copy2(instance_path(paper_id, case["instance_index"]), workspace / INSTANCE_FILENAME)
    return workspace


# The only text this adaptation authors. It exists because of how the data is
# supplied, not because of anything about the problems: COOPA's formulation step
# is told to enumerate every numeric fact it can see, and here there are none to
# see, so it has to be told where they are instead.
POINTER_BLOCK = f"""

---

The numeric data for this problem is not in the statement above. It is in the
file `{INSTANCE_FILENAME}` in the working directory, and it is far too large to
be reproduced by hand.

When you record a parameter whose value comes from that file, do not invent a
number: give its value as a reference of the form `{INSTANCE_FILENAME}:<field>`
naming where it will be found, and describe what the field means. Whoever writes
the solver code will inspect the file and load it at run time.
"""


def build_question(paper_id: str) -> str:
    """The statement as written, plus the pointer.

    COOPA discards the statement after the formulation step, which is harmless
    for a problem whose statement is a few hundred characters of prose carrying
    its own numbers. A FrontierOR statement is the only place the semantics
    exist, so it is passed on to the manager as well; DecisionBrain's agent
    reads the same file through its tools, so this restores parity rather than
    adding to it.
    """
    return problem_md_path(paper_id).read_text(encoding="utf-8").rstrip() + POINTER_BLOCK
