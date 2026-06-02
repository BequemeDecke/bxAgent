from langgraph.graph import StateGraph

from bxagent.models import build_base_model

def build_workflow_agent() -> StateGraph:
    llm = build_base_model()
    
    pass