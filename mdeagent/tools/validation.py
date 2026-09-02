from typing import Dict, List

from langchain.tools import BaseTool, tool
from mdeagent.evaluation import EvaluationExecutor, Evaluation, EvaluationRun

def create_evaluation_tools(evaluations: Dict[str, Evaluation]) -> List[BaseTool]:
    """
    Factory function to create an instance of the EvaluationTool.
    This function can be extended to accept parameters for customization.
    """
    executor = EvaluationExecutor(evaluations)
    
    @tool("evaluation_tool", return_direct=True)
    async def evaluation_tool() -> List[EvaluationRun]:
        """Executes all evaluations and returns their latest results.

        Returns:
            List[EvaluationRun]: A list of the latest EvaluationRun for each evaluation.
        """
        return await executor.execute_all()
    
    @tool("evaluation_tool_specific", return_direct=True)
    async def evaluation_tool_specific(evaluation_id: str) -> EvaluationRun:
        """Executes a specific evaluation and returns its latest result.

        Args:
            evaluation_id (str): The ID of the evaluation to execute.

        Returns:
            EvaluationRun: The latest EvaluationRun for the specified evaluation.
        """
        return await executor.execute_specific(evaluation_id)
    
    @tool("evaluation_tool_latest_results", return_direct=True)
    def evaluation_tool_latest_results() -> List[EvaluationRun]:
        """Returns the latest results of all evaluations without executing them.

        Returns:
            List[EvaluationRun]: A list of the latest EvaluationRun for each evaluation.
        """
        return executor.get_latest_results()

    return [evaluation_tool, evaluation_tool_specific, evaluation_tool_latest_results]