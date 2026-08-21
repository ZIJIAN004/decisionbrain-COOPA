from smolagents import LiteLLMModel, GradioUI, ToolCallingAgent
from src.agents import CodeAgent
from smolagents.monitoring import LogLevel
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import argparse
import tempfile
import yaml
import importlib
# import optimizer agents
from .or_agents.mathematical_optimizer_agent import create_mathematical_optimizer_agent
from .or_agents.combinatorial_optimizer_agent import create_combinatorial_optimizer_agent
from .or_agents.metaheuristic_optimizer_agent import create_metaheuristic_optimizer_agent
from .or_agents.general_optimizer_agent import create_general_optimizer_agent

from .or_agents.web_browsing_agent import create_web_browsing_agent
# import tools available to the manager agent
from general_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeFile,
    ModifyFile,
    DeleteFileOrFolder,
    CreateFileWithContent,
)
# import model utilities
from .model_utils import build_model

def create_manager_agent(model_id="gpt-4.1", working_directory=None, allow_web=True):
    """
    Create a manager agent for operations research problems.

    The manager delegates each problem to the appropriate optimizer agent
    (mathematical / combinatorial / metaheuristic / general) and solves it directly,
    without any knowledge-base retrieval or curation.
    """
    # Define the working directory
    if working_directory is None:
        # Use a temporary directory if not specified
        working_directory = tempfile.mkdtemp()
    else:
        # Create the working directory if it doesn't exist
        Path(working_directory).mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {working_directory}")

    web_browsing_agent = None
    if allow_web:
        downloads_folder = str(Path(working_directory) / "downloads")
        web_browsing_agent = create_web_browsing_agent(
            model_id=model_id, downloads_folder=downloads_folder
        )

    managed_agents_for_optimizers = [web_browsing_agent] if web_browsing_agent else []

    # Create the mathematical optimizer agent
    mathematical_optimizer_agent = create_mathematical_optimizer_agent(
        model_id=model_id,
        managed_agents=managed_agents_for_optimizers,
        working_directory=working_directory,
        verbosity_level=LogLevel.DEBUG,
        is_curation=True,
    )
    # Create the combinatorial optimizer agent
    combinatorial_optimizer_agent = create_combinatorial_optimizer_agent(
        model_id=model_id,
        managed_agents=managed_agents_for_optimizers,
        working_directory=working_directory,
        verbosity_level=LogLevel.DEBUG,
        is_curation=True,
    )
    # Create the metaheuristic optimizer agent
    metaheuristic_optimizer_agent = create_metaheuristic_optimizer_agent(
        model_id=model_id,
        managed_agents=managed_agents_for_optimizers,
        working_directory=working_directory,
        verbosity_level=LogLevel.DEBUG,
        is_curation=True,
    )
    # Create the general optimizer agent
    general_optimizer_agent = create_general_optimizer_agent(
        model_id=model_id,
        managed_agents=managed_agents_for_optimizers,
        working_directory=working_directory,
        verbosity_level=LogLevel.DEBUG,
        is_curation=True,
    )

    # Load the manager prompt template
    manager_prompt_template = yaml.safe_load(
                importlib.resources.files("apps.operations_research.or_agents.prompts").joinpath("manager.yaml").read_text(encoding="utf-8")
            )

    manager_managed_agents = [
        general_optimizer_agent,
        mathematical_optimizer_agent,
        combinatorial_optimizer_agent,
        metaheuristic_optimizer_agent,
    ]
    if web_browsing_agent is not None:
        manager_managed_agents.insert(0, web_browsing_agent)

    # Create the manager agent
    manager_agent = CodeAgent(
        tools=[
            ListDir(working_directory),
            SeeFile(working_directory),
            ModifyFile(working_directory),
            DeleteFileOrFolder(working_directory),
            CreateFileWithContent(working_directory),
            ],
        managed_agents=manager_managed_agents,
        prompt_templates=manager_prompt_template,
        additional_authorized_imports=['numpy', 'numpy.*', 'random', 'random.*', 'math', 'math.*', 'json'],
        model=build_model(model_id),
        name="or_agent",
        description="An agent that can solve operations research problems.",
        verbosity_level=LogLevel.DEBUG,
        stream_outputs=False
    )
    return manager_agent


def launch_gradio_app(manager_agent, model_id, use_iterative_refinement=False,
                      max_refinement_iterations=3, share=True):
    """
    Launch a Gradio web app that runs the FULL pipeline on a raw problem typed by
    the user: Phase 0 formulation extraction -> structured prompt -> manager agent
    delegates to an optimizer agent. This mirrors what the batch experiment runner
    does, so typing the raw problem here reproduces the runner's behaviour.
    """
    import json
    import gradio as gr
    from .formulation_utils import build_solver_prompt

    def format_report(formulation, candidates):
        if not candidates:
            # Simple (non-refined) path: only a single formulation, no scores.
            return (
                "### Formulation\n"
                "_Per-candidate confidence scoring is only available with "
                "`--use_iterative_refinement`._\n\n"
                "```json\n" + json.dumps(formulation.model_dump(), indent=2) + "\n```"
            )
        # Iterative refinement: show every candidate and its confidence scores,
        # marking the max-min selected one.
        lines = ["### Candidate formulations & confidence (max-min selection)\n",
                 "| Candidate | Min | Overall | Params | Vars | Obj | Constr |",
                 "|---|---|---|---|---|---|---|"]
        for c in candidates:
            mark = " ✅" if c["selected"] else ""
            lines.append(
                f"| M{c['iteration']}{mark} | {c['min']} | {c['overall']} | "
                f"{c['parameters']} | {c['variables']} | {c['objective']} | {c['constraints']} |"
            )
        lines.append("")
        for c in candidates:
            tag = " — SELECTED" if c["selected"] else ""
            lines.append(
                f"<details><summary><b>Candidate M{c['iteration']}{tag}</b> "
                f"(min {c['min']} / overall {c['overall']})</summary>\n"
            )
            lines.append("```json\n" + json.dumps(c["formulation"], indent=2) + "\n```")
            lines.append("</details>\n")
        return "\n".join(lines)

    def solve(question):
        if not question or not question.strip():
            return "Please enter a problem first.", ""
        # Phase 0: extract a structured formulation from the raw problem text
        prompt, formulation, candidates = build_solver_prompt(
            question,
            model_id,
            use_iterative_refinement=use_iterative_refinement,
            max_refinement_iterations=max_refinement_iterations,
        )
        report = format_report(formulation, candidates)
        # Phase 1: the manager delegates to an optimizer agent and returns the answer
        answer = manager_agent.run(prompt, reset=True)
        return report, str(answer)

    with gr.Blocks(title="COOPA — OR Solver") as demo:
        gr.Markdown(
            "# COOPA Operations Research Solver\n"
            "Enter an optimization problem in natural language. It is automatically "
            "**formulated** into a structured model and then **solved** by the most "
            "suitable optimizer agent."
        )
        question = gr.Textbox(
            label="Problem (natural language)",
            lines=8,
            placeholder="Paste your operations-research problem here...",
        )
        solve_btn = gr.Button("Solve", variant="primary")
        report_box = gr.Markdown(label="Formulation & confidence")
        answer_box = gr.Textbox(label="Answer", lines=10)
        solve_btn.click(solve, inputs=question, outputs=[report_box, answer_box])

    demo.launch(share=share)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the Operations Research Agent")
    argparser.add_argument(
        "--model_id",
        type=str,
        default="gpt-4.1-nano",
        help="The ID of the model to use for the agent.",
    )
    argparser.add_argument(
        "--working_directory",
        type=str,
        default=None,
        help="The directory where the agent will store its working files.",
    )
    argparser.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=["gradio", "cli"],
        help="The mode to run the agent in. 'gradio' for web interface, 'cli' for command line interface.",
    )
    argparser.add_argument(
        "--use_iterative_refinement",
        action="store_true",
        default=False,
        help="In gradio mode, refine the extracted formulation over several iterations.",
    )
    argparser.add_argument(
        "--max_refinement_iterations",
        type=int,
        default=3,
        help="Maximum number of formulation refinement iterations (default: 3).",
    )
    args = argparser.parse_args()
    # Ensure the base temp_files directory exists
    base_temp_dir = "apps/operations_research/temp_files"
    Path(base_temp_dir).mkdir(parents=True, exist_ok=True)
    if args.working_directory is None:
        args.working_directory = tempfile.mkdtemp(dir=base_temp_dir, prefix="working_directory_")

    # Create the agent
    manager_agent = create_manager_agent(
        model_id=args.model_id,
        working_directory=args.working_directory
        )
    
    if args.mode == "cli":
        # Run the agent in CLI mode
        while True:
            try:
                manager_agent.run("Based on the conversation so far, talk with the user.", reset=False)
                print("Agent finished running. Waiting for next command...")
                print("Press Ctrl+C to exit.")
            except KeyboardInterrupt:
                print("Exiting...")
                break
    else:
        # Run the agent in Gradio mode — full pipeline (formulate -> solve)
        print("Launching Gradio UI...")
        launch_gradio_app(
            manager_agent,
            model_id=args.model_id,
            use_iterative_refinement=args.use_iterative_refinement,
            max_refinement_iterations=args.max_refinement_iterations,
        )
