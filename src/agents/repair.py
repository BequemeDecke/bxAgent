from typing import Optional

from deepagents import create_deep_agent
from deepagents.backends import BackendProtocol
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.models import build_coding_model
from src.tools.audit import check_file_existence, check_compilation
from src.output.repair import RepairOutputSchema

REPAIR_SYSTEM_PROMPT = """
You are a subagent resp onsible for repairing code. 
You will be given a code snippet and an error message. 
Your task is to analyze the error message, understand the issue in the code, and provide a corrected version of the code snippet that resolves the error. 
Make sure to explain the changes you made to fix the issue.

Do not make any changes to the code snippet that are not necessary to fix the error!
"""


def build_repair_agent(
    system_prompt: str = REPAIR_SYSTEM_PROMPT,
    backend: Optional[BackendProtocol] = None,
):
    """Builds the RepairAgent using the chat model."""
    model = build_coding_model()

    return create_deep_agent(
        model=model,
        system_prompt=SystemMessage(system_prompt),
        checkpointer=InMemorySaver(),
        tools=[check_file_existence, check_compilation], # Exploration and editing tools are added by deepagents
        backend=backend,
        # response_format=RepairOutputSchema
    )
