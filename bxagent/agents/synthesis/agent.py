from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from bxagent.models import build_base_model
from bxagent.tools import transformation_plan_tools
from bxagent.tools.transformation import _read_transformation_plan
from .output import SynthesisResponseFormat

SYNTHESIS_SYSTEM_PROMPT = """
You are the planning agent for the Ecore model transformation process.

Your task is to analyze the source and target models, understand the requirements, 
and create a detailed implementation plan in TRANSFORMATION.md.

Workflow:
1. Read the current TRANSFORMATION.md using read_transformation_plan. If it is empty, start writing it from scratch using the input of the user.
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

Transformation Plan:
{transformation_plan}
"""


def build_synthesis_agent(
    system_prompt: str = SYNTHESIS_SYSTEM_PROMPT,
    model: BaseChatModel | None = None,
):
    """Builds the SynthesisAgent using the chat model."""
    if model is None:
        model = build_base_model()
        
    transformation_plan = _read_transformation_plan()
    system_prompt = system_prompt.format(transformation_plan=transformation_plan)

    return create_agent(
        model=model,
        system_prompt=SystemMessage(system_prompt),
        checkpointer=InMemorySaver(),
        response_format=SynthesisResponseFormat,
        tools=[*transformation_plan_tools]
    )
