from typing import Any

from mdeagent.implementation.state import ImplementationState


def implementation_to_java_files(state: ImplementationState) -> dict[str, Any]:
    return {"files": state.get("written_java_files", [])}
