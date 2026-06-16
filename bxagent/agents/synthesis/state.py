from langchain.agents import AgentState
from bxagent.comprehension import TransformationPlan


class SynthesisAgentState(AgentState):
    transformation_plan: TransformationPlan
