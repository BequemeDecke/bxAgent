from pathlib import Path

from mdeagent.preparation.pom import format_java_files

from .state import ImplementationState


def create_format_code_node(workspace: Path):
    """
    Creates a node that formats all Java files in the workspace using Maven Spotless plugin.
    
    Args:
        workspace: The workspace path where the Maven project is located.
        
    Returns:
        A node function that formats Java files and updates the state.
    """

    def format_code(state: ImplementationState) -> ImplementationState:
        """
        Format all Java files in the workspace using Maven Spotless plugin.
        
        Args:
            state: The current implementation state.
            
        Returns:
            The updated state (formatting is done in-place).
        """
        # Run spotless:apply to format all Java files
        format_java_files(workspace)
        
        # Return the state unchanged (files are formatted in-place)
        return state

    return format_code
