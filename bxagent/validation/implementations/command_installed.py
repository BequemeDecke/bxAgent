import shutil
from typing import List, Tuple
from pydantic import BaseModel

from ..types import Validation, ValidationResult, ValidationError


class CommandInstalledValidationConfig(BaseModel):
    tools: List[str]


class CommandInstalledValidation(Validation):
    async def setup(self):
        pass

    async def run(
        self, **kwargs
    ) -> Tuple[List[ValidationResult], List[ValidationError]]:
        """
        This validation checks whether the specified commands are installed on the machine.
        """

        tools = kwargs.get("tools", [])

        results = []
        errors = []

        for tool in tools:
            if shutil.which(tool) is None:
                errors.append(
                    ValidationError(
                        message=f"Command '{tool}' is not installed on the system.",
                        details={"tool": tool},
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        content=f"Command '{tool}' is installed on the system."
                    )
                )
        return results, errors
