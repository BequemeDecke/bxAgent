from typing import Any

from mdeagent.implementation.state import ImplementationState


def implementation_to_java_files(state: ImplementationState) -> dict[str, Any]:
    """Map implementation state to evaluation parameters for Java-related evaluations.
    
    For file_existence evaluation: returns the list of written Java files.
    For java_compilation evaluation: returns both files and the Maven project path.
    
    Args:
        state: The ImplementationState containing workflow state.
        
    Returns:
        A dictionary with 'files' and optionally 'project_path' keys.
    """
    result = {"files": state.get("written_java_files", [])}
    
    # Add project_path for java_compilation evaluation if available
    maven_project_path = state.get("maven_project_path")
    if maven_project_path is not None:
        result["project_path"] = maven_project_path
    
    return result
