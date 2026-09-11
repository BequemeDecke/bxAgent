from langchain.agents import AgentState

from mdeagent.comprehension import SerializedTransformationPlan


class ComprehensionAgent(AgentState):
    transformation_plan: SerializedTransformationPlan
