from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from bxagent.models import build_base_model
from bxagent.comprehension import (
    TransformationPlan,
    transformation_plan_tools,
)
from .output import SynthesisResponseFormat
from .state import SynthesisAgentState

SYNTHESIS_SYSTEM_PROMPT = """
You are the planning agent for the Ecore model transformation process.

Your task is to analyze the source and target models, understand the requirements, 
and create a detailed implementation plan in TRANSFORMATION.md (= the current transformation plan).

Workflow:
1. The transformation plan is underneath. If it is empty, start writing it from scratch using the input of the user.
2. Analyze the models, requirements, and existing content
3. Identify new difficulties and think through potential obstacles
4. Define implementation steps that provide a roadmap without dictating code details
5. Document your thinking and reasoning throughout
6. Update the transformation plan using the available tools, which also keep track of the transformation plan history
7. Self-validate: review for logical consistency and completeness

Guidelines:
- Each implementation step should be detailed enough for a coder to follow, 
  but not prescriptive about implementation details
- When identifying difficulties, explain why they are challenging
- Return your plan using the predefined response_schema

--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---
"""


def build_synthesis_agent(
    transformation_plan: TransformationPlan,
    system_prompt: str = SYNTHESIS_SYSTEM_PROMPT,
    model: BaseChatModel | None = None,
):
    """Builds the SynthesisAgent using the chat model."""
    if model is None:
        model = build_base_model()

    system_prompt = system_prompt.format(transformation_plan=str(transformation_plan))

    return create_agent(
        model=model,
        state_schema=SynthesisAgentState,
        system_prompt=SystemMessage(system_prompt),
        # checkpointer=InMemorySaver(
        #     serde=JsonPlusSerializer(
        #         pickle_fallback=True,
        #         allowed_json_modules=[TransformationPlan],
        #         allowed_msgpack_modules=[TransformationPlan],
        #     )
        # ),
        response_format=SynthesisResponseFormat,
        tools=[*transformation_plan_tools],
    )
