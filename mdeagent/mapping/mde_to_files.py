from typing import Any

from mdeagent.state import MDEAgentState


def mde_to_files(state: MDEAgentState) -> dict[str, Any]:
    return {"files": state.get("written_files", [])}
