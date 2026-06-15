from langchain.agents import AgentState
from bxagent.tools.transformation import TransformationPlan


class SynthesisAgentState(AgentState):
    transformation_plan: TransformationPlan
