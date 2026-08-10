import shutil
from typing import List, Tuple
from pydantic import BaseModel

from ..types import Evaluation, EvaluationResult, EvaluationError


class CommandInstalledEvaluationConfig(BaseModel):
    commands: List[str]


class CommandInstalledEvaluation(Evaluation):
    async def setup(self):
        pass

    async def run(
        self, **kwargs
    ) -> Tuple[List[EvaluationResult], List[EvaluationError]]:
        """
        This evaluation checks whether the specified commands are installed on the machine.
        """

        commands = kwargs.get("commands", [])

        results = []
        errors = []

        for command in commands:
            try:
                if shutil.which(command) is None:
                    results.append(
                        EvaluationResult(
                            content=f"Command '{command}' is not installed on the system.",
                            metadata={"success": False, "include_in_report": False},
                        )
                    )
                else:
                    results.append(
                        EvaluationResult(
                            content=f"Command '{command}' is installed on the system.",
                            metadata={"success": True, "include_in_report": False},
                        )
                    )
            except Exception as e:
                errors.append(
                    EvaluationError(
                        message=f"An error occurred while checking command '{command}': {str(e)}",
                        type=type(e).__name__,
                        details={"command": command},
                    )
                )
        return results, errors
