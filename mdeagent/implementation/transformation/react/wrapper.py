from langchain.agents import AgentState
from langchain.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph, RunnableConfig

from mdeagent.comprehension.plan import SerializedTransformationPlan, TransformationPlan
from mdeagent.evaluation.types import EvaluationRun
from mdeagent.evaluation.utils import format_evaluation_results
from mdeagent.implementation.types import (
    TransformationClass,
    TransformationClassGenerator,
)


class TransformationClassAgentState(AgentState):
    """
    Expands the AgentState (messages) to include the transformation plan, the list of written files, the specific task, the transformation class, and the evaluation results.
    """

    written_files: list[str]
    specific_task: str | None
    transformation_plan: SerializedTransformationPlan
    transformation_class: TransformationClass
    evaluation_results: dict[str, EvaluationRun]


INPUT_PROMPT_TEMPLATE = """
--- BEGIN VARIABLES ---
- Package of the source model: {source_model_package}
- Package of the target model: {target_model_package}
- Package of the transformation implementation: {transformation_package}
- Transformation class name: {transformation_class_name}
- Transformation class path: {transformation_class_path}
--- END VARIABLES ---

Use the read_transformation_plan tool to read the transformation plan and follow the steps described in it to implement the transformation class.

You specific task is: {specific_task}

--- BEGIN EVALUATION RESULTS ---
{evaluation_results_text}
--- END EVALUATION RESULTS ---
"""


def create_input_prompt(
    transformation_plan: TransformationPlan,
    transformation_class: TransformationClass,
    specific_task: str | None = None,
    evaluation_results: dict[str, str] | None = None,
) -> TransformationClassAgentState:
    """
    Creates an input prompt for the coding agent based on the provided transformation plan, transformation class, specific task, and evaluation results.

    Args:
        transformation_plan (TransformationPlan): The transformation plan that describes the steps to be taken in order to perform the transformation.
        transformation_class (TransformationClass): The transformation class that contains the source and target model interfaces and the transformation rules.
        specific_task (str | None, optional): A specific task that the agent should focus on. Defaults to None.
        evaluation_results (dict[str, str] | None, optional): Evaluation results from previous runs. Defaults to None.

    Returns:
        TransformationClassAgentState: The input prompt for the coding agent.
    """
    content = INPUT_PROMPT_TEMPLATE.format(
        source_model_package=transformation_plan.data["source_model_package"],
        target_model_package=transformation_plan.data["target_model_package"],
        transformation_package=transformation_class["package"],
        transformation_class_name=transformation_class["name"],
        transformation_class_path=transformation_class["path"],
        specific_task=specific_task or "No specific task provided.",
        evaluation_results_text=format_evaluation_results(evaluation_results)
        if evaluation_results
        else "No evaluation results provided.",
    )
    message = HumanMessage(content=content)
    return message


class TransformationClassAgentWrapper(TransformationClassGenerator):
    graph: CompiledStateGraph[TransformationClassAgentState]
    config: RunnableConfig

    def __init__(self, graph: CompiledStateGraph[TransformationClassAgentState]):
        super().__init__()
        self.graph = graph
        self.config = {
            "configurable": {"thread_id": "transformation_class_agent"},
        }

    async def synthesize_transformation_class(
        self,
        transformation_plan: TransformationPlan,
        transformation_class: TransformationClass,
        specific_task: str | None = None,
        evaluation_results: dict[str, EvaluationRun] | None = None,
    ):
        input = TransformationClassAgentState(
            messages=[
                create_input_prompt(
                    transformation_plan,
                    transformation_class,
                    specific_task=specific_task,
                    evaluation_results=evaluation_results or {},
                )
            ],
            written_files=[],
            specific_task=specific_task,
            transformation_plan=transformation_plan.to_dict(),
            transformation_class=transformation_class,
            evaluation_results=evaluation_results or {},
        )
        output = await self.graph.ainvoke(input, config=self.config, version="v2")
        return output.value["written_files"]
