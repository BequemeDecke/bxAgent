import asyncio
from pathlib import Path
from typing import Callable

from langchain.chat_models import BaseChatModel

from mdeagent.evaluation import (
    EvaluationPipe,
    EvaluationResult,
    EvaluationRun,
)
from mdeagent.evaluation.filter import (
    IsErrorFilter,
    IsExecutionRunFilter,
    IsReportCandidateFilter,
)
from mdeagent.implementation.transformation.generator import (
    BackwardMethodBody,
    FallbackParser,
    ForwardMethodBody,
    ImplementationTransformationSpec,
    SynchMethodBody,
    TransformationClassMetadata,
    TransformationClassTemplateResolver,
    TransformationFieldsAndConstructor,
    ainvoke_and_parse,
)
from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.types import TransformationClassGenerator

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


# Specialized prompts for piecewise generation
METADATA_PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Extract the metadata (package names and type names) for the transformation class.

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

Return the package name, source type, target type, decision type, and the transformation package where AgentTransformationForEMF is declared.
"""

FIELDS_AND_CONSTRUCTOR_PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Define the fields and constructor for the transformation class.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN METADATA ---
Package: {package_name}
Source Type: {source_type}
Target Type: {target_type}
Decision Type: {decision_type}
--- END METADATA ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return the field declarations and constructor definition for the transformation class.
Fields should be a list of objects with 'type' and 'name'.
The constructor should have 'parameters' (string) and 'assignments' (list of {{target, value}}).
If no fields or constructor are needed, return empty/null values.
"""

FORWARD_BODY_PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Implement the forward transformation method body.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN METADATA ---
Package: {package_name}
Source Type: {source_type}
Target Type: {target_type}
Decision Type: {decision_type}
--- END METADATA ---

--- BEGIN FIELDS ---
{fields_info}
--- END FIELDS ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return only the Java code for the forward method body (no method signature, just the body content).
The forward method transforms from {source_type} to {target_type}.
"""

BACKWARD_BODY_PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Implement the backward transformation method body.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN METADATA ---
Package: {package_name}
Source Type: {source_type}
Target Type: {target_type}
Decision Type: {decision_type}
--- END METADATA ---

--- BEGIN FIELDS ---
{fields_info}
--- END FIELDS ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return only the Java code for the backward method body (no method signature, just the body content).
The backward method transforms from {target_type} to {source_type}.
"""

SYNCH_BODY_PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Implement the synchronization method body.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

--- BEGIN METADATA ---
Package: {package_name}
Source Type: {source_type}
Target Type: {target_type}
Decision Type: {decision_type}
--- END METADATA ---

--- BEGIN FIELDS ---
{fields_info}
--- END FIELDS ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---

Return only the Java code for the synch method body (no method signature, just the body content).
The synch method handles incremental updates between {source_type} and {target_type}.
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
        success_status = (
            "SUCCESS" if result.metadata.get("success", True) else "FAILURE"
        )
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
    execution_runs = execution_pipe.filter_results(
        list(latest_evaluation_runs.values())
    )

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
    combined_results = list(
        {id(result): result for result in error_results + report_results}.values()
    )

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


def create_metadata_prompt(
    task_specification: str,
    transformation_plan: str,
    template: str,
    evaluation_results_text: str = "No evaluation results available.",
) -> str:
    """Create prompt for generating transformation metadata (package and type names)."""
    return METADATA_PROMPT_TEMPLATE.format(
        task_specification=task_specification,
        transformation_plan=transformation_plan,
        template=template,
        evaluation_results_text=evaluation_results_text,
    )


def create_fields_and_constructor_prompt(
    task_specification: str,
    transformation_plan: str,
    template: str,
    metadata: TransformationClassMetadata,
    evaluation_results_text: str = "No evaluation results available.",
) -> str:
    """Create prompt for generating fields and constructor."""
    return FIELDS_AND_CONSTRUCTOR_PROMPT_TEMPLATE.format(
        task_specification=task_specification,
        transformation_plan=transformation_plan,
        package_name=metadata.package_name,
        source_type=metadata.source_type,
        target_type=metadata.target_type,
        decision_type=metadata.decision_type,
        template=template,
        evaluation_results_text=evaluation_results_text,
    )


def create_forward_body_prompt(
    task_specification: str,
    transformation_plan: str,
    template: str,
    metadata: TransformationClassMetadata,
    fields_info: str,
    evaluation_results_text: str = "No evaluation results available.",
) -> str:
    """Create prompt for generating forward method body."""
    return FORWARD_BODY_PROMPT_TEMPLATE.format(
        task_specification=task_specification,
        transformation_plan=transformation_plan,
        package_name=metadata.package_name,
        source_type=metadata.source_type,
        target_type=metadata.target_type,
        decision_type=metadata.decision_type,
        fields_info=fields_info,
        template=template,
        evaluation_results_text=evaluation_results_text,
    )


def create_backward_body_prompt(
    task_specification: str,
    transformation_plan: str,
    template: str,
    metadata: TransformationClassMetadata,
    fields_info: str,
    evaluation_results_text: str = "No evaluation results available.",
) -> str:
    """Create prompt for generating backward method body."""
    return BACKWARD_BODY_PROMPT_TEMPLATE.format(
        task_specification=task_specification,
        transformation_plan=transformation_plan,
        package_name=metadata.package_name,
        source_type=metadata.source_type,
        target_type=metadata.target_type,
        decision_type=metadata.decision_type,
        fields_info=fields_info,
        template=template,
        evaluation_results_text=evaluation_results_text,
    )


def create_synch_body_prompt(
    task_specification: str,
    transformation_plan: str,
    template: str,
    metadata: TransformationClassMetadata,
    fields_info: str,
    evaluation_results_text: str = "No evaluation results available.",
) -> str:
    """Create prompt for generating synch method body."""
    return SYNCH_BODY_PROMPT_TEMPLATE.format(
        task_specification=task_specification,
        transformation_plan=transformation_plan,
        package_name=metadata.package_name,
        source_type=metadata.source_type,
        target_type=metadata.target_type,
        decision_type=metadata.decision_type,
        fields_info=fields_info,
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

    This node uses a **piecewise structured LLM approach** to generate the
    transformation class. Instead of generating everything in one large call,
    it breaks down the generation into smaller steps:

    1. First, generate metadata (package and type names) - this is needed as context
    2. Then, in parallel, generate:
       - Fields and constructor
       - Forward method body
       - Backward method body
       - Synch method body
    3. Finally, combine all parts and render the template

    This approach reduces timeout risk by using smaller, focused prompts and
    processing independent parts in parallel.

    Args:
        llm: The base chat model to use for generation.
        optional_plan_factory: A factory function to create a transformation plan if none exists.
        template_path: The path to the templates directory.

    Returns:
        A node function that generates the transformation class and updates the state.
    """
    # Create parser for structured responses
    parser = FallbackParser()

    async def invoke_and_parse(prompt: str, model_class) -> object:
        """Invoke LLM and parse response using the shared parser infrastructure."""
        return await ainvoke_and_parse(llm, prompt, model_class, parser)

    resolver = TransformationClassTemplateResolver(template_path=template_path)

    async def implement_transformation(
        state: ImplementationState,
    ) -> ImplementationState:
        transformation_class_path = state.get("transformation_class_path")
        if transformation_class_path is None:
            raise ValueError(
                "Transformation class path is required to write the generated code."
            )
        transformation_package_path = state.get("transformation_package_path")

        # 1. Read the transformation plan from the state or create one
        transformation_plan = state.get("transformation_md") or optional_plan_factory()

        # 2. Filter evaluation results from JavaCompilation and FileExistence runs
        latest_evaluation_runs = state.get("latest_evaluation_runs", {})
        filtered_results = _filter_execution_results(latest_evaluation_runs)
        evaluation_results_text = _format_evaluation_results(filtered_results)

        # 3. Build the base inputs
        task_specification = state.get("task_specification")
        raw_template = resolver.get_raw_template()

        # STEP 1: Generate metadata first (needed as context for other parts)
        metadata_prompt = create_metadata_prompt(
            task_specification=task_specification,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            evaluation_results_text=evaluation_results_text,
        )
        metadata_response: TransformationClassMetadata = await invoke_and_parse(
            metadata_prompt, TransformationClassMetadata
        )

        # Prepare fields info string for context in method body generation
        fields_result_dict = metadata_response.model_dump()
        fields_list = fields_result_dict.get("fields", []) or []
        fields_info = "Fields: " + ", ".join(
            f"{f.get('type', 'Object')} {f.get('name', 'field')}" for f in fields_list
        )

        # STEP 2: Generate independent parts in PARALLEL
        # Create all prompts for parallel generation
        fields_prompt = create_fields_and_constructor_prompt(
            task_specification=task_specification,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata_response,
            evaluation_results_text=evaluation_results_text,
        )

        forward_prompt = create_forward_body_prompt(
            task_specification=task_specification,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata_response,
            fields_info=fields_info,
            evaluation_results_text=evaluation_results_text,
        )

        backward_prompt = create_backward_body_prompt(
            task_specification=task_specification,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata_response,
            fields_info=fields_info,
            evaluation_results_text=evaluation_results_text,
        )

        synch_prompt = create_synch_body_prompt(
            task_specification=task_specification,
            transformation_plan=str(transformation_plan),
            template=raw_template,
            metadata=metadata_response,
            fields_info=fields_info,
            evaluation_results_text=evaluation_results_text,
        )

        # Execute all four calls in parallel
        (
            fields_result,
            forward_result,
            backward_result,
            synch_result,
        ) = await asyncio.gather(
            invoke_and_parse(fields_prompt, TransformationFieldsAndConstructor),
            invoke_and_parse(forward_prompt, ForwardMethodBody),
            invoke_and_parse(backward_prompt, BackwardMethodBody),
            invoke_and_parse(synch_prompt, SynchMethodBody),
        )

        # STEP 3: Combine all parts into the final spec
        combined_spec = ImplementationTransformationSpec(
            package_name=transformation_package_path,
            source_type=metadata_response.source_type,
            target_type=metadata_response.target_type,
            decision_type=metadata_response.decision_type,
            transformation_package=metadata_response.transformation_package,
            fields=fields_result.fields or [],
            constructor=fields_result.constructor,
            forward_body=forward_result.forward_body,
            backward_body=backward_result.backward_body,
            synch_body=synch_result.synch_body,
            transform_source_to_target_body=None,  # Will default to calling forward
            transform_target_to_source_body=None,  # Will default to calling backward
        )

        # STEP 4: Render the template with the generated specification
        transformation_class_name = transformation_class_path.stem
        rendered_code = resolver.render_template(
            combined_spec, class_name=transformation_class_name
        )

        # STEP 5: Write the generated code to a file
        transformation_class_path.touch(exist_ok=True)
        transformation_class_path.write_text(rendered_code, encoding="utf-8")

        # STEP 6: Retrieve the written files from the state and add the new one
        written_java_files = state.get("written_java_files", []) + [
            transformation_class_path
        ]

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
