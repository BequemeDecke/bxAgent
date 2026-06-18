from bxagent.agents.workflow.state import WorkflowState
from typing import Dict, Any


def map_workflow_to_commands(state: WorkflowState) -> Dict[str, Any]:
    return {"commands": state.get("required_commands", [])}
