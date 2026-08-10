from langchain.agents import AgentState
from mdagent.comprehension import TransformationPlan


class ComprehensionAgentState(AgentState):
    transformation_plan: TransformationPlan
