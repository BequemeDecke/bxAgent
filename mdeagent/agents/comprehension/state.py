from langchain.agents import AgentState

from mdeagent.comprehension import SerializedTransformationPlan


class ComprehensionAgentState(AgentState):
    transformation_plan: SerializedTransformationPlan
