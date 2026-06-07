import json
from pathlib import Path
from smolagents import LiteLLMModel, tool
from src.agents import CodeAgent
from smolagents.monitoring import LogLevel
from dotenv import load_dotenv
load_dotenv()

import os
import base64
import random
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import re
import warnings
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings.*",
    category=UserWarning,
)
import litellm
litellm.suppress_debug_info = True

import argparse
import tempfile
import shutil
import multiprocessing
from multiprocessing import Pool, Lock

from datetime import datetime
from .run import create_manager_agent

# Import formulation extraction tools
from .or_agents.formulation import (
    create_instructor_client,
    extract_formulation,
)

# Import iterative formulation refinement
from .or_agents.iterative_formulation import (
    extract_formulation_with_refinement,
)

# Shared prompt builder (also used by the interactive Gradio app in run.py)
from .formulation_utils import format_formulation_prompt

def strip_ansi_codes(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

class CleanOutputFile:
    """File wrapper that strips ANSI codes before writing."""
    def __init__(self, file):
        self.file = file

    def write(self, text):
        clean_text = strip_ansi_codes(str(text))
        return self.file.write(clean_text)

    def flush(self):
        return self.file.flush()

    def __getattr__(self, name):
        return getattr(self.file, name)

def get_current_timestamp():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def normalize_dataset_item(item):
    """
    Normalize dataset item keys to handle different dataset formats.

    Supports:
    - BWOR format: {"question", "answer", "index"}
    - industryor/other formats: {"en_question", "en_answer", "id"}

    Returns a normalized dict with keys: "question", "answer", "id"
    """
    normalized = {}

    # Handle question key
    if "en_question" in item:
        normalized["question"] = item["en_question"]
    elif "question" in item:
        normalized["question"] = item["question"]
    else:
        raise ValueError("Item missing both 'en_question' and 'question' keys")

    # Handle answer key
    if "en_answer" in item:
        normalized["answer"] = item["en_answer"]
    elif "answer" in item:
        normalized["answer"] = item["answer"]
    else:
        raise ValueError("Item missing both 'en_answer' and 'answer' keys")

    # Handle id/index key
    if "id" in item:
        normalized["id"] = item["id"]
    elif "index" in item:
        normalized["id"] = item["index"]
    else:
        raise ValueError("Item missing both 'id' and 'index' keys")

    return normalized

def process_single_problem(args_tuple):
    """
    Worker function to process a single problem.
    This function will be called by each multiprocessing worker.

    Args:
        args_tuple: Tuple containing all necessary parameters

    Returns:
        dict: Result dictionary for this problem
    """
    (item, model_id, log_dir, dataset_name, output_path,
     use_iterative_refinement, max_refinement_iterations, working_directory) = args_tuple

    # Normalize dataset item keys to handle different formats (BWOR vs industryor)
    normalized_item = normalize_dataset_item(item)
    question = normalized_item["question"]
    gold_answer = normalized_item["answer"]
    idx = normalized_item["id"]

    # Create a unique working directory for this process
    problem_working_directory = Path(working_directory) / f"problem_{idx}"
    problem_working_directory.mkdir(parents=True, exist_ok=True)

    # Create manager agent for this worker
    manager_agent = create_manager_agent(
        model_id=model_id,
        working_directory=str(problem_working_directory),
    )

    # Create log file for this question
    model_name = model_id.replace('/', '-').replace('.', '_')
    log_file = log_dir / f"{dataset_name}_{model_name}_question_{idx}_log.txt"

    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Open log file and capture all output including formulation extraction
        with open(log_file, 'w', encoding='utf-8') as f_log:
            # Write header
            f_log.write(f"=== Dataset: {dataset_name} | Model: {model_id} | Question {idx} ===\n\n")

            # Extract a structured formulation from the raw problem text
            formulation_confidence_data = None
            prompt = None

            f_log.write(f"=== PHASE 0: FORMULATION EXTRACTION ===\n\n")
            f_log.write(f"Original Problem:\n{question}\n\n")
            f_log.write(f"{'='*80}\n\n")

            # Wrap file with ANSI code stripper
            clean_log = CleanOutputFile(f_log)

            # Redirect stdout/stderr to capture formulation extraction output
            sys.stdout = clean_log
            sys.stderr = clean_log

            try:
                if use_iterative_refinement:
                    # Use iterative refinement with confidence evaluation
                    print(f"Extracting formulation with iterative refinement for problem {idx}...")
                    formulation, evaluation, num_iterations = extract_formulation_with_refinement(
                        problem_text=question,
                        max_iterations=max_refinement_iterations,
                        formulation_model=model_id,
                        evaluation_model=model_id,
                        verbose=True
                    )
                    formulation_confidence_data = {
                        "evaluation": evaluation.model_dump(),
                        "num_iterations": num_iterations
                    }
                    print(f"\nFormulation refined in {num_iterations} iteration(s) for problem {idx}")
                    print(f"Final confidence: {evaluation.overall_confidence}/100")
                else:
                    # Use simple extraction without refinement
                    formulation_client = create_instructor_client(model_name=model_id, timeout=90.0)
                    print(f"Extracting formulation for problem {idx}...")
                    formulation = extract_formulation(
                        problem_text=question,
                        client=formulation_client,
                        model=model_id
                    )
                    print(f"Formulation extracted successfully for problem {idx}")

                # Format the formulation into a structured prompt
                prompt = format_formulation_prompt(formulation)

                # Save formulation schema and evaluation results to working directory
                try:
                    formulation_file = problem_working_directory / "formulation.json"
                    with open(formulation_file, 'w', encoding='utf-8') as f:
                        json.dump(formulation.model_dump(), f, indent=2)
                    print(f"Formulation saved to {formulation_file}")

                    # Save schema if evaluation data is available
                    if formulation_confidence_data is not None:
                        evaluation_file = problem_working_directory / "formulation_evaluation.json"
                        with open(evaluation_file, 'w', encoding='utf-8') as f:
                            json.dump(formulation_confidence_data, f, indent=2)
                        print(f"Evaluation results saved to {evaluation_file}")

                except Exception as schema_error:
                    print(f"Warning: Failed to save formulation files: {schema_error}")

            except Exception as e:
                # Do NOT silently fall back to raw text — surface the failure loudly.
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                f_log.write(f"\n=== FORMULATION EXTRACTION FAILED for problem {idx}: {e} ===\n")
                raise

            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            f_log.write(f"\n{'='*80}\n\n")

            f_log.write(f"=== PHASE 1: PROBLEM SOLVING ===\n\n")
            f_log.write(f"Prompt:\n{prompt}\n\n")
            f_log.write(f"{'='*80}\n\n")

            # Wrap file with ANSI code stripper
            clean_log = CleanOutputFile(f_log)

            # Redirect stdout/stderr to clean file wrapper
            sys.stdout = clean_log
            sys.stderr = clean_log

            try:
                agent_response = manager_agent.run(prompt, reset=True)
                # Try to extract a number from the response
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(agent_response))
                if match:
                    predicted = float(match.group())
                    correct = abs(predicted - float(gold_answer)) < 0.1
                else:
                    predicted = None
                    correct = False
            except Exception as e:
                agent_response = str(e)
                predicted = None
                correct = False

            # Restore stdout/stderr before writing summary
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # Write Phase 1 summary to log file
            f_log.write(f"\n{'='*80}\n")
            f_log.write(f"Phase 1 Final Response: {agent_response}\n")
            f_log.write(f"\nGold Answer: {gold_answer}\n")
            f_log.write(f"Predicted Answer: {predicted}\n")
            f_log.write(f"Correct: {correct}\n")
    finally:
        # Always restore stdout/stderr even if there's an error
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    result = {
        "index": idx,
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted,
        "agent_response": str(agent_response),
        "correct": correct,
    }

    # Add formulation confidence data if available
    if formulation_confidence_data is not None:
        result["formulation_confidence"] = formulation_confidence_data

    print(f"Problem {idx}: Correct={correct} | Gold={gold_answer} | Predicted={predicted}")

    return result


def run_experiment(
    dataset_path,
    cur_date_time,
    model_id="gpt-4.1",
    working_directory="working_directory",
    output_path="experiment_results.jsonl",
    num_processes=None,
    use_iterative_refinement=False,
    max_refinement_iterations=3
):
    """
    Run experiments on the full dataset.

    The pipeline extracts a structured formulation for each problem, then delegates
    it to the manager agent which solves it via the optimizer agents.
    """
    print(f"Running experiment on FULL dataset")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(working_directory).mkdir(parents=True, exist_ok=True)
    print(f"Using working directory: {Path(working_directory).resolve()}")

    if "industryor" in dataset_path:
        dataset_name = "industryor"
    elif "BWOR" in dataset_path:
        dataset_name = "BWOR"
    elif "complexlp" in dataset_path:
        dataset_name = "complexlp"

    # Per-problem logs go in a logs/ subfolder next to the results JSONL
    log_dir = Path(output_path).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load all problems from the dataset (from the first question onward)
    problems_to_process = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            problems_to_process.append(item)

    print(f"Processing {len(problems_to_process)} problems")

    # Prepare arguments for each problem
    args_list = [
        (item, model_id, log_dir, dataset_name, output_path,
         use_iterative_refinement, max_refinement_iterations, working_directory)
        for item in problems_to_process
    ]

    # Determine number of processes
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(f"Using {num_processes} parallel processes")

    # Process problems in parallel using multiprocessing
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for better performance (order doesn't matter)
        # Results will be written to file as they complete
        for result in pool.imap_unordered(process_single_problem, args_list):
            # Write results incrementally as they complete
            with open(output_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(result, default=str) + "\n")

    print(f"Experiment finished. Results saved to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch experiments with manager agent on the selected dataset.",
    )
    parser.add_argument("--dataset", type=str, default="industryor")
    parser.add_argument("--model_id", type=str, default="o4-mini")
    parser.add_argument("--working_directory", type=str, default=None, help="Path to permanent working directory (default: <output folder>/working_directory)")
    parser.add_argument("--output", type=str)
    parser.add_argument("--num_processes", type=int, default=1,
                       help="Number of parallel processes to use (default: 1)")
    parser.add_argument("--use_iterative_refinement", action="store_true", default=False,
                       help="Use iterative refinement with confidence evaluation for formulation extraction")
    parser.add_argument("--max_refinement_iterations", type=int, default=3,
                       help="Maximum number of refinement iterations (default: 3)")
    args = parser.parse_args()

    cur_date_time = get_current_timestamp()

    if args.output is None:
        args.output = Path(f"results/{args.dataset}_{args.model_id.replace('/', '-')}/experiment_results_{cur_date_time}.jsonl").resolve()
    if args.working_directory is None:
        # Co-locate per-problem working dirs with the results JSONL:
        # results/{dataset}_{model}/working_directory/problem_{idx}/
        args.working_directory = Path(args.output).parent / "working_directory"

    run_experiment(
        dataset_path=f"apps/operations_research/datasets/{args.dataset}/{args.dataset}.jsonl",
        cur_date_time=cur_date_time,
        model_id=args.model_id,
        working_directory=args.working_directory,
        output_path=args.output,
        num_processes=args.num_processes,
        use_iterative_refinement=args.use_iterative_refinement,
        max_refinement_iterations=args.max_refinement_iterations,
    )
