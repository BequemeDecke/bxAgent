from langchain.agents import AgentState
from mdeagent.comprehension import TransformationPlan


class ComprehensionAgentState(AgentState):
    transformation_plan: TransformationPlan
