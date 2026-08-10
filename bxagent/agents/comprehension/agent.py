from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage

from mdagent.models import build_base_model
from mdagent.comprehension import (
    TransformationPlan,
)
from mdagent.tools.comprehension import transformation_plan_tools
from .state import ComprehensionAgentState

COMPREHENSION_SYSTEM_PROMPT = """
You are the planning agent for the Ecore model transformation process.

Your task is to analyze the source and target models, understand the requirements, 
and create a detailed implementation plan in TRANSFORMATION.md (= the current transformation plan).

Workflow:
1. The transformation plan is given by the user input. If it is empty, start writing it from scratch using the input of the user.
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
"""


def build_comprehension_agent(
    system_prompt: str = COMPREHENSION_SYSTEM_PROMPT,
    model: BaseChatModel | None = None,
):
    """Builds the ComprehensionAgent using the chat model."""
    if model is None:
        model = build_base_model()

    return create_agent(
        model=model,
        state_schema=ComprehensionAgentState,
        system_prompt=SystemMessage(system_prompt),
        # checkpointer=InMemorySaver(
        #     serde=JsonPlusSerializer(
        #         pickle_fallback=True,
        #         allowed_json_modules=[TransformationPlan],
        #         allowed_msgpack_modules=[TransformationPlan],
        #     )
        # ),
        tools=[*transformation_plan_tools],
    )
