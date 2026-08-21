"""Shared utilities for formulation extraction and prompt building."""


def format_formulation_prompt(formulation) -> str:
    """
    Convert an OptimizationFormulation object into a structured prompt for the
    manager agent (parameters, decision variables, objective, constraints +
    delegation instructions).
    """
    prompt_parts = []

    prompt_parts.append("Delegate the following operations research problem to the correct optimizer agent:\n")

    # Parameters section
    if formulation.parameters:
        prompt_parts.append("\n## PARAMETERS:")
        for param in formulation.parameters:
            param_str = f"- {param.name} ({param.data_type}): {param.description}"
            if param.value is not None:
                param_str += f" = {param.value}"
            if param.units:
                param_str += f" [{param.units}]"
            prompt_parts.append(param_str)

    # Variables section
    if formulation.variables:
        prompt_parts.append("\n## DECISION VARIABLES:")
        for var in formulation.variables:
            var_str = f"- {var.name} ({var.data_type}): {var.description}"
            var_str += f" | Domain: {var.domain}"
            prompt_parts.append(var_str)

    # Objective section
    prompt_parts.append("\n## OBJECTIVE:")
    prompt_parts.append(f"- Sense: {formulation.objective.sense.upper()}")
    prompt_parts.append(f"- Description: {formulation.objective.description}")
    prompt_parts.append(f"- Expression: {formulation.objective.expression}")
    prompt_parts.append(f"- Variables involved: {', '.join(formulation.objective.variables_involved)}")

    # Constraints section
    if formulation.constraints:
        prompt_parts.append("\n## CONSTRAINTS:")
        for i, constraint in enumerate(formulation.constraints, 1):
            prompt_parts.append(f"\n{i}. {constraint.name} ({constraint.sense}):")
            prompt_parts.append(f"   Expression: {constraint.expression}")
            prompt_parts.append(f"   Variables: {', '.join(constraint.variables_involved)}")

    prompt_parts.append("\n\n## CRITICAL INSTRUCTIONS:")
    prompt_parts.append("- You are the MANAGER. You MUST NOT solve this problem yourself. Do NOT write solver code, do NOT perform calculations, and do NOT reason about the solution.")
    prompt_parts.append("- Your ONLY job is to delegate the COMPLETE problem above to the appropriate optimizer agent (mathematical_optimizer_agent, combinatorial_optimizer_agent, metaheuristic_optimizer_agent, or general_optimizer_agent) in your FIRST Code block.")
    prompt_parts.append("- The optimizer agent will handle everything: inspecting the provided instance.json in its working directory, building the solver, executing it, and returning the result.")
    prompt_parts.append("- Do NOT call final_answer() in the same response where you call an optimizer agent. You MUST wait for the system to return the optimizer's REAL result first, then call final_answer() in a SEPARATE response.")
    prompt_parts.append("- Your code block MUST start with EXACTLY ```py (three backticks followed by py). Do NOT omit the backticks. If you write just 'py' without backticks, the code will NOT execute and the delegation will FAIL.")
    prompt_parts.append("- AFTER writing ```<end_code>, STOP IMMEDIATELY. Do NOT output any more text. Do NOT write 'Successfully executed', do NOT guess results, do NOT write the next Thought/Code block. Any text after ```<end_code> means you are hallucinating and your answer will be WRONG.")

    return "\n".join(prompt_parts)


def build_solver_prompt(question, model_id, use_iterative_refinement=False,
                        max_refinement_iterations=3, verbose=False):
    """
    Run formulation extraction on a raw problem and return the structured prompt
    that should be handed to the manager agent.

    This is the same Phase-0 pipeline the batch experiment runner uses, so an
    interactive caller can take a raw problem and reproduce the runner's behaviour.

    Returns:
        (prompt, formulation, candidates) where:
        - prompt: the manager prompt string,
        - formulation: the selected OptimizationFormulation object,
        - candidates: with iterative refinement, a list of per-iteration candidates
          (each a dict with iteration, selected flag, confidence scores, and the
          formulation dump); None for the simple (non-refined) path.
    """
    from .or_agents.formulation import create_instructor_client, extract_formulation
    from .or_agents.iterative_formulation import extract_formulation_with_refinement

    candidates = None
    if use_iterative_refinement:
        formulation, _evaluation, selected_iteration, history = extract_formulation_with_refinement(
            problem_text=question,
            max_iterations=max_refinement_iterations,
            formulation_model=model_id,
            evaluation_model=model_id,
            verbose=verbose,
            return_history=True,
        )
        candidates = []
        for entry in history:
            ev = entry["evaluation"]
            candidates.append({
                "iteration": entry["iteration"],
                "selected": entry["iteration"] == selected_iteration,
                "overall": entry["overall_confidence"],
                "min": entry["min_confidence"],
                "parameters": ev.parameters.confidence,
                "variables": ev.decision_variables.confidence,
                "objective": ev.objective.confidence,
                "constraints": ev.constraints.confidence,
                "formulation": entry["formulation"].model_dump(),
            })
    else:
        client = create_instructor_client(model_name=model_id, timeout=90.0)
        formulation = extract_formulation(problem_text=question, client=client, model=model_id)

    return format_formulation_prompt(formulation), formulation, candidates


def wrap_formulation_text(formulation_text: str) -> str:
    """
    Wrap raw formulation text (## PARAMETERS through ## CONSTRAINTS)
    with the delegation preamble and CRITICAL INSTRUCTIONS.

    This produces the same prompt as format_formulation_prompt() in
    run_exp_with_kb_full_multiprocess.py but from raw text instead
    of an OptimizationFormulation object.

    Args:
        formulation_text: Raw formulation text extracted from logs

    Returns:
        Full prompt string ready for the manager agent
    """
    parts = []

    parts.append("Delegate the following operations research problem to the correct optimizer agent:\n")
    parts.append(formulation_text.strip())

    parts.append("\n\n## CRITICAL INSTRUCTIONS:")
    parts.append("- You are the MANAGER. You MUST NOT solve this problem yourself. Do NOT write solver code, do NOT perform calculations, and do NOT reason about the solution.")
    parts.append("- Your ONLY job is to delegate the COMPLETE problem above to the appropriate optimizer agent (mathematical_optimizer_agent, combinatorial_optimizer_agent, metaheuristic_optimizer_agent, or general_optimizer_agent) in your FIRST Code block.")
    parts.append("- The optimizer agent will handle everything: saving parameters to JSON via create_file_with_content(), building the solver, executing it, and returning the result.")
    parts.append("- Do NOT call final_answer() in the same response where you call an optimizer agent. You MUST wait for the system to return the optimizer's REAL result first, then call final_answer() in a SEPARATE response.")
    parts.append("- Your code block MUST start with EXACTLY ```py (three backticks followed by py). Do NOT omit the backticks. If you write just 'py' without backticks, the code will NOT execute and the delegation will FAIL.")
    parts.append("- AFTER writing ```<end_code>, STOP IMMEDIATELY. Do NOT output any more text. Do NOT write 'Successfully executed', do NOT guess results, do NOT write the next Thought/Code block. Any text after ```<end_code> means you are hallucinating and your answer will be WRONG.")

    return "\n".join(parts)
