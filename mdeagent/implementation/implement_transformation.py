from pathlib import Path
from typing import Callable

from langchain.chat_models import BaseChatModel

from mdeagent.evaluation import (
    EvaluationPipe,
    EvaluationResult,
    EvaluationRun,
)
from mdeagent.evaluation.filter import IsErrorFilter, IsExecutionRunFilter, IsReportCandidateFilter
from mdeagent.implementation.generator import (
    ImplementationTransformationSpec,
    TransformationClassTemplateResolver,
)
from mdeagent.implementation.state import ImplementationState

PROMPT_TEMPLATE_WITH_PLAN = """
You are a Java transformation code generator for EMF-based model transformations.
Generate a concrete implementation of the AgentTransformationForEMF interface based on the task specification and the provided template.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return a valid structured result matching the required Java class structure.
The implementation must use the EMF interface methods and the Java generic types for source, target, and decisions.
"""


def _format_evaluation_results(results: list[EvaluationResult]) -> str:
    """
    Format a list of EvaluationResult objects into a human-readable text.
    
    Args:
        results: List of evaluation results to format.
        
    Returns:
        A formatted string representation of the evaluation results.
    """
    if not results:
        return "No evaluation results available."
    
    formatted_lines = []
    for i, result in enumerate(results, start=1):
        success_status = "SUCCESS" if result.metadata.get("success", True) else "FAILURE"
        formatted_lines.append(f"{i}. [{success_status}] {result.content}")
        
        # Add metadata details if present
        metadata = result.metadata
        if "file" in metadata:
            formatted_lines.append(f"   File: {metadata['file']}")
        if "line" in metadata:
            line_info = f"Line: {metadata['line']}"
            if "column" in metadata:
                line_info += f", Column: {metadata['column']}"
            formatted_lines.append(f"   {line_info}")
    
    return "\n".join(formatted_lines)


def _filter_execution_results(
    latest_evaluation_runs: dict[str, EvaluationRun],
) -> list[EvaluationResult]:
    """
    Filter evaluation results from execution runs (JavaCompilation and FileExistence).
    
    Uses EvaluationPipe with IsErrorFilter to get error results and
    IsReportCandidateFilter to get report-worthy results.
    Both filters are applied separately and combined (OR logic).
    
    Args:
        latest_evaluation_runs: Dictionary of evaluation runs.
        
    Returns:
        A list of filtered evaluation results.
    """
    # Filter runs by category "execution" using IsExecutionRunFilter
    execution_pipe = EvaluationPipe() | IsExecutionRunFilter
    execution_runs = execution_pipe.filter_results(list(latest_evaluation_runs.values()))
    
    # Collect all results from execution runs
    all_execution_results: list[EvaluationResult] = []
    for run in execution_runs:
        all_execution_results.extend(run.results)
    
    if not all_execution_results:
        return []
    
    # Use EvaluationPipe with IsErrorFilter to get error results
    error_pipe = EvaluationPipe() | IsErrorFilter
    error_results = error_pipe.filter_results(all_execution_results)
    
    # Use EvaluationPipe with IsReportCandidateFilter to get report-worthy results
    report_pipe = EvaluationPipe() | IsReportCandidateFilter
    report_results = report_pipe.filter_results(all_execution_results)
    
    # Combine both lists, avoiding duplicates (OR logic)
    combined_results = list({id(result): result for result in error_results + report_results}.values())
    
    return combined_results


def create_input_prompt(
    task_specification: str,
    transformation_plan: str,
    template: str,
    evaluation_results_text: str = "No evaluation results available.",
) -> str:
    return PROMPT_TEMPLATE_WITH_PLAN.format(
        task_specification=task_specification,
        transformation_plan=transformation_plan,
        template=template,
        evaluation_results_text=evaluation_results_text,
    )


def create_implement_transformation_node(
    llm: BaseChatModel,
    optional_plan_factory: Callable,
    template_path: Path = Path.cwd() / "templates",
):
    """
    Creates the implement_transformation node for the implementation graph.

    This node uses a structured LLM approach to generate the transformation class.
    It reads the transformation plan and includes it in the prompt sent to the LLM.

    Args:
        llm: The base chat model to use for generation.
        optional_plan_factory: A factory function to create a transformation plan if none exists.
        template_path: The path to the templates directory.

    Returns:
        A node function that generates the transformation class and updates the state.
    """
    structured_llm = llm.with_structured_output(ImplementationTransformationSpec)
    resolver = TransformationClassTemplateResolver(template_path=template_path)

    async def implement_transformation(state: ImplementationState) -> ImplementationState:
        transformation_class_path = state.get("transformation_class_path")
        if transformation_class_path is None:
            raise ValueError(
                "Transformation class path is required to write the generated code."
            )
        
        # 1. Read the transformation plan from the state or create one
        transformation_plan = state.get("transformation_md") or optional_plan_factory()

        # 2. Filter evaluation results from JavaCompilation and FileExistence runs
        latest_evaluation_runs = state.get("latest_evaluation_runs", {})
        filtered_results = _filter_execution_results(latest_evaluation_runs)
        evaluation_results_text = _format_evaluation_results(filtered_results)

        # 3. Build the prompt for the LLM based on the transformation plan, task specification,
        #    and evaluation results
        task_specification = state.get("task_specification")
        raw_template = resolver.get_raw_template()
        input_prompt = create_input_prompt(
            task_specification=task_specification,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            evaluation_results_text=evaluation_results_text,
        )

        # 3. Invoke the structured LLM to generate the transformation class.
        #    The class *name* is no longer asked from the LLM: it is determined
        #    in the ``prepare_workspace`` node and reaches this node encoded in
        #    the ``transformation_class_path`` state field.
        response: ImplementationTransformationSpec = await structured_llm.ainvoke(input=input_prompt)

        # 4. Render the template with the generated specification. The class
        #    name is derived from the transformation class path (set by
        #    ``prepare_workspace``) so that the file name and the declared class
        #    always stay consistent.
        transformation_class_name = transformation_class_path.stem
        rendered_code = resolver.render_template(
            response, class_name=transformation_class_name
        )

        # 5. Write the generated code to a file
        transformation_class_path.touch(exist_ok=True)
        transformation_class_path.write_text(rendered_code, encoding="utf-8")

        # 6. Retrieve the written files from the state and add the new one
        written_java_files = state.get("written_java_files", []) + [transformation_class_path]

        # NOTE: The iteration counter is *not* advanced here. Unlike the
        # preparation subgraph (which has a single work node), the
        # implementation graph may run several work nodes per cycle
        # (``implement_transformation`` + ``implement_bx_tool``), and the
        # ``integration_error`` branch even routes back to ``implement_bx_tool``
        # without re-running ``implement_transformation``. Incrementing in a
        # work node would therefore either double-count or skip the increment
        # entirely. The counter is advanced once per cycle in the
        # ``evaluate_implementation`` node instead (see ``agent.py``).
        return {
            "transformation_md": transformation_plan,
            "written_java_files": written_java_files,
            "task_specification": task_specification,
            "transformation_implementation": rendered_code,
        }

    return implement_transformation
