from langchain.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from typing import Callable

from bxagent.comprehension.plan import TransformationPlan
from .state import ImplementationState

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
    coding_agent: CompiledStateGraph,
    optional_plan_factory: Callable[[], TransformationPlan],
):
    def implement_transformation(agent_state: ImplementationState) -> ImplementationState:
        # 1. Read the transformation plan from the TRANSFORMATION.md file
        transformation_plan = (
            agent_state.get("transformation_md") or optional_plan_factory()
        )

        # 2. Build the prompt for the coding agent based on the transformation plan and task specification
        task_specification = agent_state.get("task_specification")
        input_prompt = create_input_prompt(task_specification, transformation_plan)

        # 3. Invoke the coding deep agent with the created prompt
        written_java_files = agent_state.get("written_java_files", []) # If there are already some files in previous iterations
        response = coding_agent.invoke(
            input={
                "messages": [HumanMessage(content=input_prompt)],
                "written_java_files": written_java_files,
            },
            version="v2",
        )

        # 4. Retrieve the state of the agent to get the written_java_files
        written_java_files = response.value["data"]["written_java_files"]

        return {
            "transformation_md": transformation_plan,
            "written_java_files": written_java_files,
            "task_specification": task_specification,
        }

    return implement_transformation
