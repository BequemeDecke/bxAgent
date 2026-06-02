from langchain.messages import SystemMessage
from langchain.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from deepagents.backends import BackendProtocol
from deepagents import create_deep_agent

from bxagent.models import build_base_model
from bxagent.tools import transformation_plan_tools
from .middleware import SynthesisAgentStateMiddleware

SYNTHESIS_SYSTEM_PROMPT = """
You are the planning agent for the Ecore model transformation process.

Your task is to analyze the source and target models, understand the requirements, 
and create a detailed implementation plan in TRANSFORMATION.md.

Workflow:
1. Read the current TRANSFORMATION.md using read_transformation_plan
2. Analyze the models, requirements, and existing content
3. Identify new difficulties (marked with "NEW:") and think through potential obstacles
4. Define implementation steps that provide a roadmap without dictating code details
5. Document your thinking and reasoning throughout
6. Update TRANSFORMATION.md using update_transformation_plan
7. Self-validate: review for logical consistency and completeness

Guidelines:
- Each implementation step should be detailed enough for a coder to follow, 
  but not prescriptive about implementation details
- When identifying difficulties, explain why they are challenging
- Return your plan using the predefined response_schema

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
