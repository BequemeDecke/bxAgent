from pathlib import Path
from typing import Literal

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents import AgentState
from langchain.chat_models import BaseChatModel
from langchain.tools import ToolRuntime, tool

from mdeagent.comprehension.plan import SerializedTransformationPlan, TransformationPlan
from mdeagent.models import build_coding_model

SYSTEM_PROMPT = """
You are a coding agent in a model driven development environment that helps writing java code for model to model transformations.
You are given the interfaces of the source and target models, and the transformation rules, and you are expected to write the code for the transformation between them.

A transformation plan is also provided, which describes the steps to be taken in order to perform the transformation. You should follow the plan and write the code accordingly.
"""


class CodingAgentState(AgentState):
    """
    Expands the AgentState (messages) to include the transformation plan and the list of written java files.
    """

    transformation_plan: SerializedTransformationPlan
    written_java_files: list[Path]


Section = Literal[
    "source_model_implementation",
    "target_model_implementation",
    "transformation_direction",
    "implementation_steps",
    "difficulties",
    "source_model_package",
    "target_model_package",
    "source_model_name",
    "target_model_name",
]


@tool
def read_transformation_plan(runtime: ToolRuntime, section: Section) -> str:
    """Tool to read the transformation plan from the runtime state. The transformation plan is stored in the runtime state as a serialized object, and this tool deserializes it and returns it as a TransformationPlan object.

    Args:
        runtime (ToolRuntime): The runtime of the agent, which contains the state where the transformation plan is stored.
        section (Section): The section of the transformation plan to read. Can be one of "source_model_implementation", "target_model_implementation", "transformation_direction", "implementation_steps", or "difficulties".

    Raises:
        ValueError: If the transformation plan is not found in the runtime state.

    Returns:
        str: The requested section of the transformation plan as a string.
    """
    serialized_tp: SerializedTransformationPlan = runtime.state.get(
        "transformation_plan"
    )
    if serialized_tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp: TransformationPlan = TransformationPlan.from_dict(serialized_tp)
    return tp.data[section]


def build_coding_agent(workspace: Path, model: BaseChatModel | None = None):
    """Creates a coding agent that can write code for model to model transformations based on the provided transformation plan.
    The agent is created with a custom set of tools that allow it to read and write the transformation class files, as well as to read the transformation plan.

    Args:
        workspace (Path): The root path of the workspace where the agent will operate.
        model (BaseChatModel | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if model is None:
        model = build_coding_model()

    return create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_path=workspace, virtual_mode=True),
        state_schema=CodingAgentState,
        tools=[read_transformation_plan],
    )
