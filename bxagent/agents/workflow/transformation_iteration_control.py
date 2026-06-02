from typing import Literal

from .state import WorkflowState
from bxagent.config import Config

config = Config.get_instance()

WORKFLOW_MAX_ITERATIONS = config.WORKFLOW_APPROACH.WORKFLOW_MAX_ITERATIONS

def check_transformation_iteration(state: WorkflowState, max_iterations: int = WORKFLOW_MAX_ITERATIONS) -> Literal["stop", "continue", "error"]:
    """
    Gate function to check if the transformation needs another iteration or not.
    """
    if state['iteration'] >= max_iterations:
        return "stop"
    
    return "continue"
    