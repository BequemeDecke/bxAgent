from langchain.chat_models import BaseChatModel
from typing import Optional

from bxagent.tools import transformation
from .state import CodingAgentState


def get_transformation_plan(transformation_md: Optional[str]) -> str:
    transformation_md = transformation_md or transformation._read_transformation_plan()
    if transformation_md.strip() == "":
        raise ValueError("The TRANSFORMATION.md file is empty. Please provide the necessary information for the transformation.")
    return transformation_md


def create_implement_transformation_node(llm: BaseChatModel):
    def implement_transformation(agent_state: CodingAgentState) -> CodingAgentState:
        # 1. Read the transformation plan from the TRANSFORMATION.md file
        transformation_md = get_transformation_plan(agent_state["transformation_md"])

        return {"transformation_md": transformation_md}

    return implement_transformation
