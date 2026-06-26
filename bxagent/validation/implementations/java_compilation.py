import shutil
import logging
import re
import subprocess

from typing import List, Tuple
from pydantic import BaseModel
from pathlib import Path

from ..types import ValidationResult, ValidationError, Validation


class JavaCompilationValidationConfig(BaseModel):
    files: List[Path]


class JavaCompilationValidation(Validation):
    """
    Validation that checks if Java files can be compiled successfully.

    It uses the `javac` command to attempt to compile the provided Java files. If `javac` is not installed on the system, the setup will fail. The run method will return an empty list of results and errors for now, as the actual compilation logic is not implemented yet.
    """

    async def setup(self):
        """Check if `javac` is available on the system. If not, raise a RuntimeError."""
        if shutil.which("javac") is None:
            logging.error(
                "javac is not installed on the system, but is required for Java compilation."
            )
            raise RuntimeError("javac is not installed on the system.")

        logging.debug(
            "javac is available on the system. JavaCompilationValidation setup completed successfully."
        )

    async def run(
        self, **kwargs
    ) -> Tuple[List[ValidationResult], List[ValidationError]]:
        """Attempt to compile the provided Java files using `javac`. If there are compilation errors, parse the output and return them as ValidationErrors.

        Returns:
            Tuple[List[ValidationResult], List[ValidationError]]: A tuple containing a list of successful validation results and a list of validation errors.
        """
        config = JavaCompilationValidationConfig(**kwargs)
        files = config.files

        results: List[ValidationResult] = []
        errors: List[ValidationError] = []

        for file in files:
            try:
                subprocess_result = subprocess.run(
                    ["javac", file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                if subprocess_result.returncode != 0:
                    logging.error(
                        f"Compilation failed for {file} with error: {subprocess_result.stderr}"
                    )
                    results.extend(parse_javac_output(subprocess_result.stderr))
                else:
                    logging.debug(f"Compilation succeeded for {file}.")
                    results.append(
                        ValidationResult(
                            content=f"Compilation succeeded for {file}",
                            metadata={"file": file, "success": True},
                        )
                    )
            except Exception as e:
                logging.exception(f"An error occurred while compiling {file}: {e}")
                errors.append(
                    ValidationError(
                        message=f"An error occurred while compiling {file}: {str(e)}",
                        type=type(e).__name__,
                        details={"file": file},
                    )
                )

        return results, errors


def parse_javac_output(output: str) -> List[ValidationResult]:
    """
    Parse the output of the `javac` command to extract compilation errors. It uses regular expressions to identify error messages and their associated file, line number, and code block. The extracted information is then used to create a list of ValidationError objects.

    Args:
        output (str): The output from the `javac` command.

    Returns:
        List[ValidationResult]: A list of ValidationResult objects representing the compilation errors.
    """
    errors = []
    error_pattern = re.compile(r"^(.*\.java):(\d+): (.*)$", re.MULTILINE)
    matches = error_pattern.findall(output)

    for file, line, message in matches:
        errors.append(
            ValidationResult(
                content=message,
                metadata={
                    "file": file,
                    "line": int(line),
                    "success": False,
                    "include_in_report": True,
                },
            )
        )

    return errors
