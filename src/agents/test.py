from langchain.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from deepagents.backends import BackendProtocol
from typing import Optional

from src.models import build_coding_model
from src.tools.testing import check_file_existence
from src.tools.transformation import read_transformation_plan

TEST_SYSTEM_PROMPT = """
You are a subagent responsible for testing the generated code. You have access to the following tools:
- read_transformation_plan: Use this tool the read the transformation plan and extract the files which should have been created
- check_file_existence(file_path: Path): Checks if a file exists at the given file path. Returns True if the file exists, False otherwise.

Your task is to use these tools to validate the generated code. You should check if the expected files are created and if they exist in the correct location. You can use the `check_file_existence` tool to check for the existence of files.
"""


def build_test_agent(
    system_prompt: str = TEST_SYSTEM_PROMPT,
    backend: Optional[BackendProtocol] = None
):
    """Builds the TestAgent using the chat model.
    This agent is more deterministic. The LLM is only used to extract the file paths of the generated code.
    For more control, we need Langgraph instead of Langchain, as we need to extract the file paths from the LLM output and use them to save the generated code in the correct location.
    """
    model = build_coding_model()

    return create_deep_agent(
        model=model,
        system_prompt=SystemMessage(content=system_prompt),
        tools=[read_transformation_plan, check_file_existence],
        backend=backend,
        checkpointer=InMemorySaver()
    )
