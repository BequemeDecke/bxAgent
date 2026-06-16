from pathlib import Path

from langchain.chat_models import BaseChatModel

from bxagent.comprehension.bxtool import BxToolForEMF, BxToolTemplateResolver

from .state import CodingAgentState

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


def create_implement_bx_tool_node(llm: BaseChatModel, workspace: Path):
    structured_llm = llm.with_structured_output(BxToolForEMF)
    resolver = BxToolTemplateResolver()

    def implement_bx_tool(state: CodingAgentState) -> CodingAgentState:
        # 1. Collect information and construct the prompt
        task_specification = state["task_specification"]
        transformation_implementation = state["transformation_implementation"]
        raw_template = resolver.get_raw_template()
        input_prompt = create_input_prompt(
            task_specification=task_specification,
            template=raw_template,
            transformation_implementation=transformation_implementation,
        )

        # 2. Invoke the llm to get the bx tool implementation
        response: BxToolForEMF = structured_llm.invoke(input=input_prompt)
        bx_tool = resolver.render_template(response)

        # 3. Write the implementation to the workspace file
        file_path = workspace / (
            response.transformation_implementation.class_name + ".java"
        )
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
            file_path.touch()
        file_path.write_text(bx_tool)

        return {"written_java_files": state.get("written_java_files", []) + [file_path]}

    return implement_bx_tool
