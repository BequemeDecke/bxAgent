import shutil

from pydantic import BaseModel

from mdeagent.evaluation.types import Evaluation, EvaluationError, EvaluationResult


class ToolInstalledSchema(BaseModel):
    tools: list[str]


class ToolInstalledEvaluation(Evaluation):
    async def setup(self):
        pass

    async def run(
        self, **kwargs
    ) -> tuple[list[EvaluationResult], list[EvaluationError]]:
        """
        This evaluation checks whether the specified tools are installed on the machine.
        """

        tools = kwargs.get("tools", [])

        results = []
        errors = []

        for tool in tools:
            try:
                if shutil.which(tool) is None:
                    results.append(
                        EvaluationResult(
                            content=f"Tool '{tool}' is not installed on the system.",
                            metadata={"success": False, "include_in_report": False},
                        )
                    )
                else:
                    results.append(
                        EvaluationResult(
                            content=f"Tool '{tool}' is installed on the system.",
                            metadata={"success": True, "include_in_report": False},
                        )
                    )
            except Exception as e:
                errors.append(
                    EvaluationError(
                        message=f"An error occurred while checking tool '{tool}': {str(e)}",
                        type=type(e).__name__,
                        details={"tool": tool},
                    )
                )
        return results, errors
