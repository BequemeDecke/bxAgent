from typing import Any

from mdeagent.state import MDEAgentState


def mde_to_workspace(state: MDEAgentState) -> dict[str, Any]:
    return {
        "workspace_path": state["workspace_path"],
        "artifact_id": state["artifact_id"],
        "package_path": state["transformation_package_path"],
    }
