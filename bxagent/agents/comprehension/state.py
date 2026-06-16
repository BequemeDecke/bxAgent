from langchain.agents import AgentState
from bxagent.comprehension import TransformationPlan


class ComprehensionAgentState(AgentState):
    transformation_plan: TransformationPlan
