from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage
from pydantic import BaseModel

from .state import CodingAgentState

PROMPT_TEMPLATE = """
"""


class ImplementBxToolOutput(BaseModel):
    pass


def create_input_prompt(task_specification: str, transformation_md: str) -> str:
    return PROMPT_TEMPLATE


def create_implement_bx_tool_node(llm: BaseChatModel):
    structured_llm = llm.with_structured_output(ImplementBxToolOutput)

    def implement_bx_tool(state: CodingAgentState) -> CodingAgentState:
        transformation_md = state["transformation_md"]
        task_specification = state["task_specification"]
        input_prompt = create_input_prompt(task_specification, transformation_md)
        response: ImplementBxToolOutput = structured_llm.invoke(
            input=HumanMessage(content=input_prompt)
        )

        return {}

    return implement_bx_tool
