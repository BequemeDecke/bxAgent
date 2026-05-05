from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend, CompositeBackend, FilesystemBackend
from pathlib import Path

from src.models import build_coding_model

SYNTHESIS_SYSTEM_PROMPT = """

"""


def build_synthesis_agent(
    system_prompt: str = SYNTHESIS_SYSTEM_PROMPT,
):
    """Builds the SynthesisAgent using the chat model."""
    model = build_coding_model()

    return create_deep_agent(
        model=model,
        system_prompt=SystemMessage(system_prompt),
        checkpointer=InMemorySaver(),
    )
