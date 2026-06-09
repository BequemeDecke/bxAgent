from langchain.chat_models import BaseChatModel

from .state import CodingAgentState

def create_implement_transformation_node(llm: BaseChatModel):
    def implement_transformation(agent_state: CodingAgentState) -> CodingAgentState:
        pass

    return implement_transformation
