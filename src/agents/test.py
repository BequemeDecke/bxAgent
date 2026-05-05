from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent

from src.models import build_coding_model

TEST_SYSTEM_PROMPT = """

"""


def build_test_agent(
    system_prompt: str = TEST_SYSTEM_PROMPT,
):
    """Builds the TestAgent using the chat model."""
    model = build_coding_model()

    return create_deep_agent(
        model=model,
        system_prompt=SystemMessage(system_prompt),
        checkpointer=InMemorySaver(),
    )
