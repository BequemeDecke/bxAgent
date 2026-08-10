from typing import Dict, List

from langchain.tools import BaseTool, tool
from bxagent.evaluation import ValidationExecutor, Validation, ValidationRun

def create_validation_tools(validations: Dict[str, Validation]) -> List[BaseTool]:
    """
    Factory function to create an instance of the ValidationTool.
    This function can be extended to accept parameters for customization.
    """
    executor = ValidationExecutor(validations)
    
    @tool("validation_tool", return_direct=True)
    async def validation_tool() -> List[ValidationRun]:
        """Executes all validations and returns their latest results.

        Returns:
            List[ValidationRun]: A list of the latest ValidationRun for each validation.
        """
        return await executor.execute_all()
    
    @tool("validation_tool_specific", return_direct=True)
    async def validation_tool_specific(validation_id: str) -> ValidationRun:
        """Executes a specific validation and returns its latest result.

        Args:
            validation_id (str): The ID of the validation to execute.

        Returns:
            ValidationRun: The latest ValidationRun for the specified validation.
        """
        return await executor.execute_specific(validation_id)
    
    @tool("validation_tool_latest_results", return_direct=True)
    def validation_tool_latest_results() -> List[ValidationRun]:
        """Returns the latest results of all validations without executing them.

        Returns:
            List[ValidationRun]: A list of the latest ValidationRun for each validation.
        """
        return executor.get_latest_results()

    return [validation_tool, validation_tool_specific, validation_tool_latest_results]