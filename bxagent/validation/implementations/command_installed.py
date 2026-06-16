import shutil
from typing import List, Tuple
from pydantic import BaseModel

from ..types import Validation, ValidationResult, ValidationError


class CommandInstalledValidationConfig(BaseModel):
    commands: List[str]


class CommandInstalledValidation(Validation):
    async def setup(self):
        pass

    async def run(
        self, **kwargs
    ) -> Tuple[List[ValidationResult], List[ValidationError]]:
        """
        This validation checks whether the specified commands are installed on the machine.
        """

        commands = kwargs.get("commands", [])

        results = []
        errors = []

        for command in commands:
            if shutil.which(command) is None:
                errors.append(
                    ValidationError(
                        message=f"Command '{command}' is not installed on the system.",
                        details={"command": command},
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        content=f"Command '{command}' is installed on the system."
                    )
                )
        return results, errors
