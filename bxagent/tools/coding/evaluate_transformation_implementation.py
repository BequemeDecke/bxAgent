from typing import Literal

from langchain.chat_models import BaseChatModel
from pydantic import BaseModel

from bxagent.config import Config
from .state import CodingAgentState

config = Config.get_instance()

WORKFLOW_MAX_ITERATIONS = config.WORKFLOW_APPROACH.WORKFLOW_MAX_ITERATIONS


EvaluationDecision = Literal[
    "implementation_error",
    "integration_error",
    "implementation_success",
    "max_iteration_reached",
]


class EvaluationRoute(BaseModel):
    decision: EvaluationDecision = None


def create_input_prompt_for_evaluation() -> str:
    pass


def create_evaluate_transformation_implementation(llm: BaseChatModel):
    structured_llm = llm.with_structured_output(EvaluationRoute)

    def evaluate_transformation_implementation(
        agent_state: CodingAgentState, max_iterations: int = WORKFLOW_MAX_ITERATIONS
    ) -> EvaluationDecision:
        iteration = agent_state.get("implementation_iteration", 0)
        if iteration >= max_iterations:
            return "max_iteration_reached"

        if any(
            audit_run.errors
            for audit_run in agent_state.get("latest_audit_results", {}).values()
        ):
            return "implementation_error"

    return evaluate_transformation_implementation
