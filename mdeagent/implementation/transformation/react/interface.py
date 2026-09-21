from langgraph.graph.state import CompiledStateGraph, RunnableConfig

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationRun
from mdeagent.implementation.types import (
    TransformationClass,
    TransformationClassGenerator,
)


class TransformationClassAgentState:
    written_java_files: list[str]
    specific_task: str | None
    transformation_plan: str
    transformation_class: TransformationClass
    evaluation_results: dict[str, EvaluationRun]


class TransformationClassAgent(TransformationClassGenerator):
    graph: CompiledStateGraph[TransformationClassAgentState]
    config: RunnableConfig

    def __init__(self, graph: CompiledStateGraph[TransformationClassAgentState]):
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
            written_java_files=[],
            specific_task=specific_task,
            transformation_plan=transformation_plan,
            transformation_class=transformation_class,
            evaluation_results=evaluation_results or {},
        )
        output = await self.graph.ainvoke(input, config=self.config, version="v2")
        return output.value["written_java_files"]
