from langchain.messages import SystemMessage
from langchain.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from deepagents.backends import BackendProtocol
from deepagents import create_deep_agent

from bxagent.models import build_base_model
from bxagent.tools import transformation_plan_tools
from .middleware import SynthesisAgentStateMiddleware

SYNTHESIS_SYSTEM_PROMPT = """
You are the synthesis agent of the transformation process. 

Your task is to think about how to transform the source model into the target model with requirements given by the user and implement the transformation afterwards. 
To do this, you will need to analyze the source and target models, understand the given requirements, and then come up with a plan to transform the source model into the target model. Then you will implement the transformation by writing the transformation class in the transformation plan with the filesystem tools.

Sometimes it is not straightforward to transform the source model into the target model, and you have to decide between multiple configurations. 
In this case, you should think step by step and protocol your thoughts in a clear and structured way.

Always protocol your thoughts with the tools: 
- update_transformation_plan: This tool allows you to update the transformation plan with new thoughts or changes. You should use this tool whenever you want to update the transformation plan with new thoughts or changes.
- read_transformation_plan: This tool reads the markdown file and returns the existing transformation plan.

Your final output should be the transformation class which is a class that transforms the source model into the target model.
For this, write the transformation class in the transformation plan with the filesystem tools. Note that the programming language of the transformation class should be Java.

In summary, this is your workflow:
Input: 
    - The source and target model
    - The requirements given by the user
    - The transformation plan written so far (if any) which you can get with the tool: read_transformation_plan
Task:
    - Analyze the source and target models and understand the given requirements.
    - Come up with a plan to transform the source model into the target model. Be detailed and think step by step. If there are multiple options, think about the pros and cons of each option and decide which one to choose.
    - Protocol your thoughts in a clear and structured way with the tools: update_transformation
    - After the transformation plan is complete, write the transformation class in the transformation plan with the filesystem tools.
Output:
    - The transformation class which is a class that transforms the source model into the target model.
    - The synthesis thoughts written with the tool: write_transformation_plan
    - A list with the created files of your transformation class which you write to the `transformation.md` file.
"""


def build_synthesis_agent(
    system_prompt: str = SYNTHESIS_SYSTEM_PROMPT,
    model: BaseChatModel | None = None,
    backend: BackendProtocol | None = None,
):
    """Builds the SynthesisAgent using the chat model."""
    if model is None:
        model = build_base_model()

    return create_deep_agent(
        model=model,
        system_prompt=SystemMessage(system_prompt),
        middleware=[SynthesisAgentStateMiddleware()],
        checkpointer=InMemorySaver(),
        backend=backend,
        tools=[*transformation_plan_tools]
    )
