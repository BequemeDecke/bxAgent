from typing import Dict, List

from langchain.tools import BaseTool, tool
from .executor import ValidationExecutor, Audit, ValidationRun

def create_audit_tools(audits: Dict[str, Audit]) -> List[BaseTool]:
    """
    Factory function to create an instance of the AuditTool.
    This function can be extended to accept parameters for customization.
    """
    executor = ValidationExecutor(audits)
    
    @tool("audit_tool", return_direct=True)
    async def audit_tool() -> List[ValidationRun]:
        """Executes all audits and returns their latest results.

        Returns:
            List[ValidationRun]: A list of the latest ValidationRun for each audit.
        """
        return await executor.execute_all()
    
    @tool("audit_tool_specific", return_direct=True)
    async def audit_tool_specific(audit_id: str) -> ValidationRun:
        """Executes a specific audit and returns its latest result.

        Args:
            audit_id (str): The ID of the audit to execute.

        Returns:
            ValidationRun: The latest ValidationRun for the specified audit.
        """
        return await executor.execute_specific(audit_id)
    
    @tool("audit_tool_latest_results", return_direct=True)
    def audit_tool_latest_results() -> List[ValidationRun]:
        """Returns the latest results of all audits without executing them.

        Returns:
            List[ValidationRun]: A list of the latest ValidationRun for each audit.
        """
        return executor.get_latest_results()

    return [audit_tool, audit_tool_specific, audit_tool_latest_results]