from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend, CompositeBackend, FilesystemBackend
from pathlib import Path

from src.models import build_coding_model

SYNTHESIS_SYSTEM_PROMPT = """
You are the synthesis agent of the transformation process. 

Your task is to think about how to transform the source model into the target model with requirements given by the user. 
To do this, you will need to analyze the source and target models, understand the given requirements, and then come up with a plan to transform the source model into the target model.

Sometimes it is not straightforward to transform the source model into the target model, and you have to decide between multiple options. 
In this case, you should think step by step and protocol your thoughts in a clear and structured way.

Always protocol your thoughts with the tool: write_transformation_plan. This tool takes as input a markdown file where you can write your thoughts and the transformation plan.
Always use the tool: get_transformation_plan. This tool reads the markdown file and returns the existing transformation plan.

Input: 
    - The source and target model
    - The requirements given by the user
    - The transformation plan written so far (if any) which you can get with the tool: get_transformation_plan
Output:
    - The transformation class which is a class that transforms the source model into the target model.
    - The synthesis thoughts written with the tool: write_transformation_plan
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
