from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent

from src.models import build_coding_model

REPAIR_SYSTEM_PROMPT = """

"""


def build_repair_agent(
    system_prompt: str = REPAIR_SYSTEM_PROMPT,
):
    """Builds the RepairAgent using the chat model."""
    model = build_coding_model()

    return create_deep_agent(
        model=model,
        system_prompt=SystemMessage(system_prompt),
        checkpointer=InMemorySaver(),
    )
