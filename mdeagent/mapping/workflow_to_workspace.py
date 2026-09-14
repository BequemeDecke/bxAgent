from mdeagent.state import MDEAgentState
from typing import Dict, Any


def map_workflow_to_workspace(state: MDEAgentState) -> Dict[str, Any]:
    return {
        "workspace_path": state["workspace_path"],
        "artifact_id": state["artifact_id"],
        "package_path": state["transformation_package_path"],
    }
