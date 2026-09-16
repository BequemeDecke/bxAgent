from typing import Any

from mdeagent.state import MDEAgentState


def mde_to_tools(state: MDEAgentState) -> dict[str, Any]:
    return {"tools": state.get("required_tools", [])}
