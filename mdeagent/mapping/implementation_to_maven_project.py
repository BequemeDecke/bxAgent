from typing import Any

from mdeagent.evaluation.types import StateToEvaluationMapper
from mdeagent.implementation.state import ImplementationState


def implementation_to_maven_project(state: ImplementationState) -> dict[str, Any]:
    return {"project_path": state.get("maven_project_path")}
