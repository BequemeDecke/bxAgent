from bxagent.implementation.state import ImplementationState
from typing import Dict, Any


def map_coding_to_file(state: ImplementationState) -> Dict[str, Any]:
    return {"files": state.get("written_java_files", [])}
