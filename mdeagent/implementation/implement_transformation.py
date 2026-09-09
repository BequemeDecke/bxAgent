from pathlib import Path
from typing import Callable

from langchain.chat_models import BaseChatModel

from mdeagent.implementation.generator import (
    TransformationClassSpec,
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

Return a valid structured result matching the required Java class structure.
The implementation must use the EMF interface methods and the Java generic types for source, target, and decisions.
"""


def create_input_prompt(
    task_specification: str, transformation_plan: str, template: str
) -> str:
    return PROMPT_TEMPLATE_WITH_PLAN.format(
        task_specification=task_specification,
        transformation_plan=transformation_plan,
        template=template,
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
        workspace: The workspace path where files will be written.
        optional_plan_factory: A factory function to create a transformation plan if none exists.
        template_path: The path to the templates directory.

    Returns:
        A node function that generates the transformation class and updates the state.
    """
    structured_llm = llm.with_structured_output(TransformationClassSpec)
    resolver = TransformationClassTemplateResolver(template_path=template_path)

    async def implement_transformation(state: ImplementationState) -> ImplementationState:
        transformation_class_path = state.get("transformation_class_path")
        if transformation_class_path is None:
            raise ValueError(
                "Transformation class path is required to write the generated code."
            )
        
        # 1. Read the transformation plan from the state or create one
        transformation_plan = state.get("transformation_md") or optional_plan_factory()

        # 2. Build the prompt for the LLM based on the transformation plan and task specification
        task_specification = state.get("task_specification")
        raw_template = resolver.get_raw_template()
        input_prompt = create_input_prompt(
            task_specification=task_specification,
            transformation_plan=str(transformation_plan),
            template=raw_template,
        )

        # 3. Invoke the structured LLM to generate the transformation class
        response: TransformationClassSpec = await structured_llm.ainvoke(input=input_prompt)

        # 4. Render the template with the generated specification
        rendered_code = resolver.render_template(response)

        # 5. Write the generated code to a file
        transformation_class_path.touch(exist_ok=True)
        transformation_class_path.write_text(rendered_code, encoding="utf-8")

        # 6. Retrieve the written files from the state and add the new one
        written_java_files = state.get("written_java_files", []) + [transformation_class_path]

        return {
            "transformation_md": transformation_plan,
            "written_java_files": written_java_files,
            "task_specification": task_specification,
            "transformation_implementation": rendered_code,
        }

    return implement_transformation
