from typing import Literal

from langchain.chat_models import BaseChatModel
from pydantic import BaseModel

from bxagent.config import Config
from .state import CodingAgentState

config = Config.get_instance()

WORKFLOW_MAX_ITERATIONS = config.WORKFLOW_APPROACH.WORKFLOW_MAX_ITERATIONS


class EvaluationRoute(BaseModel):
    decision: Literal[
        "implementation_error", "integration_error", "implementation_success"
    ] = None


def create_input_prompt_for_evaluation() -> str:
    pass


def create_evaluate_transformation_implementation(llm: BaseChatModel):
    structured_llm = llm.with_structured_output(EvaluationRoute)

    def evaluate_transformation_implementation(
        agent_state: CodingAgentState, max_iterations: int = WORKFLOW_MAX_ITERATIONS
    ) -> Literal["implementation_error", "integration_error", "implementation_success"]:
        results = agent_state.get("latest_audit_results", [])

    return evaluate_transformation_implementation
