from typing import Any

from mdeagent.state import MDEAgentState


def map_workflow_to_maven_project(state: MDEAgentState) -> dict[str, Any]:
    """
    Map the workflow state to the schema required for Java compilation evaluation.

    Extracts the Maven project path from the state and returns it as 'project_path'.

    Args:
        state: The MDEAgentState containing the workflow state.

    Returns:
        A dictionary with 'project_path' key for JavaCompilationEvaluationConfig.

    Raises:
        KeyError: If 'maven_project_path' is not set in the state.
    """
    return {"project_path": state["maven_project_path"]}
