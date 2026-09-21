from pathlib import Path

from langchain.chat_models import BaseChatModel

from mdeagent.implementation.bxtool.bxtool import BxToolForEMF, BxToolTemplateResolver
from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.transformation.template.generator import (
    FallbackParser,
    ainvoke_and_parse,
)

PROMPT_TEMPLATE = """
You are a coding assistant for implementing a bx tool for EMF model transformations. 
The provided transformation class is already implemented.
Your task is to implement an adapter with the BxToolForEMF interface structure that integrates the transformation logic into a bx tool, so the transformation can be tested with benchmarx.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN BX TOOL TEMPLATE ---
{template}
--- END BX TOOL TEMPLATE ---

--- BEGIN TRANSFORMATION IMPLEMENTATION ---
{transformation_implementation}
--- END TRANSFORMATION IMPLEMENTATION ---
"""


def create_input_prompt(
    task_specification: str, template: str, transformation_implementation: str
) -> str:
    return PROMPT_TEMPLATE.format(
        task_specification=task_specification,
        template=template,
        transformation_implementation=transformation_implementation,
    )


def create_implement_bx_tool_node(
    llm: BaseChatModel, workspace: Path, benchmarx_path: Path
):
    """
    Creates the implement_bx_tool node for the implementation graph.

    This node uses the CodeGenerator infrastructure to generate a bx tool
    adapter that integrates the transformation with BenchmarX.

    Args:
        llm: The base chat model to use for generation.
        workspace: Path to the workspace directory.
        benchmarx_path: Path to the BenchmarX installation.

    Returns:
        A node function that generates the bx tool and updates the state.
    """
    # Create parser for structured responses
    parser = FallbackParser()
    resolver = BxToolTemplateResolver()

    async def implement_bx_tool(state: ImplementationState) -> ImplementationState:
        # 1. Collect information and construct the prompt
        task_specification = state["task_specification"]

        # Get the transformation implementation code from transformation_class
        transformation_class = state["transformation_class"]
        transformation_implementation = transformation_class.get("code")
        if transformation_implementation is None:
            raise ValueError(
                "Transformation implementation code is required for bx tool generation. "
                "Make sure implement_transformation runs before implement_bx_tool and "
                "stores the code in transformation_class['code']."
            )

        bxtool_path = state["bxtool_path"]
        if bxtool_path is None:
            raise ValueError(
                "BxTool path is required for implement_bx_tool. "
                "This node should only be called when BenchmarX integration is enabled."
            )
        raw_template = resolver.get_raw_template()
        input_prompt = create_input_prompt(
            task_specification=task_specification,
            template=raw_template,
            transformation_implementation=transformation_implementation,
        )

        # 2. Invoke the llm to get the bx tool implementation using shared parser
        response: BxToolForEMF = await ainvoke_and_parse(
            llm, input_prompt, BxToolForEMF, parser
        )
        bx_tool = resolver.render_template(response)

        # 3. Write the implementation to the workspace file
        bxtool_path.touch(exist_ok=True)
        bxtool_path.write_text(bx_tool)

        # NOTE: The iteration counter is advanced in the ``evaluate_implementation``
        # node (see ``agent.py``), not in the work nodes. See
        # ``implement_transformation`` for the rationale.
        return {
            "written_files": state.get("written_files", []) + [bxtool_path],
        }

    return implement_bx_tool
