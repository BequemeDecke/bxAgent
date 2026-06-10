from langchain.chat_models import BaseChatModel
from typing import Optional, Callable

from bxagent.tools.transformation.plan import TransformationPlan
from .state import CodingAgentState

PROMPT_TEMPLATE = """
Implement the specific task required for the transformation based on the transformation plan provided below.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

Guidelines:
- Follow the implementation steps outlined in `# 4. Implementation Steps` of the transformation plan. 
- If there are any difficulties mentioned in `# 3. Identified Difficulties`, make sure to address them in your implementation. 

--- BEGIN TRANSFORMATION PLAN ---
{transformation_specification}
--- END TRANSFORMATION PLAN ---
"""


def create_input_prompt(
    task_specification: str, transformation_plan: TransformationPlan
) -> str:
    transformation_specification = str(transformation_plan)
    return PROMPT_TEMPLATE.format(
        task_specification=task_specification,
        transformation_specification=transformation_specification,
    )


def create_implement_transformation_node(
    optional_plan_factory: Callable[[], TransformationPlan],
):
    def implement_transformation(agent_state: CodingAgentState) -> CodingAgentState:
        # 1. Read the transformation plan from the TRANSFORMATION.md file
        transformation_plan = (
            agent_state["transformation_md"] or optional_plan_factory()
        )

        # 2. Build the prompt for the coding agent based on the transformation plan and task specification
        task_specification = agent_state["task_specification"]
        input_prompt = create_input_prompt(task_specification, transformation_plan)

        return {"transformation_md": transformation_plan}

    return implement_transformation
