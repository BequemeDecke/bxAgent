from pathlib import Path

from mdeagent.preparation.maven import MavenProject

from .state import ImplementationState


def create_format_code_node(workspace: Path):
    """
    Creates a node that formats all Java files in the workspace using Maven Spotless plugin.
    
    Args:
        workspace: The workspace path where the Maven project is located.
        
    Returns:
        A node function that formats Java files and updates the state.
    """

    async def format_code(state: ImplementationState) -> ImplementationState:
        """
        Format all Java files in the workspace using Maven Spotless plugin.
        
        Args:
            state: The current implementation state.
            
        Returns:
            The updated state (formatting is done in-place).
        """
        # Run spotless:apply to format all Java files
        maven_project = MavenProject.load(state["maven_project_path"])
        maven_project.format() # TODO: Return an error if formatting fails
        
        # Return the state unchanged (files are formatted in-place)
        return state

    return format_code
