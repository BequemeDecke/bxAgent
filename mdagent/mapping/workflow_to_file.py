from mdagent.agents.workflow.state import WorkflowState
from typing import Dict, Any


def map_workflow_to_file(state: WorkflowState) -> Dict[str, Any]:
    return {"files": state.get("written_files", [])}
