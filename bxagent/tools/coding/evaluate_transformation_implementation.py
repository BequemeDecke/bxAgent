from typing import Literal

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


def create_input_prompt_for_evaluation() -> str:
    pass


def create_evaluate_transformation_implementation():
    def evaluate_transformation_implementation(
        agent_state: CodingAgentState, max_iterations: int = WORKFLOW_MAX_ITERATIONS
    ) -> EvaluationDecision:
        iteration = agent_state.get("implementation_iteration", 0)
        if iteration >= max_iterations:
            return "max_iteration_reached"

        # If there is an error in the bxtool_file it's an integration error, otherwise it's an implementation error
        latest_results = agent_state.get("latest_audit_results", {})
        integration_results = latest_results.get("integration_compilation")
        if integration_results is None or len(integration_results.errors) > 0:
            return "integration_error"

        # If there are any errors in the other audit results, it's an implementation error
        if any(
            len(results.errors) > 0
            for key, results in latest_results.items()
            if key != "integration_compilation"
        ):
            return "implementation_error"

        # If there are no errors, we consider the implementation successful
        return "implementation_success"

    return evaluate_transformation_implementation
