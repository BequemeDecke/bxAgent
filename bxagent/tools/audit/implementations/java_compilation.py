import shutil
import logging
import re
import subprocess

from typing import List, Tuple

from ..types import AuditResult, AuditError, Audit


class JavaCompilationAudit(Audit):
    """
    Audit that checks if Java files can be compiled successfully.

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
            "javac is available on the system. JavaCompilationAudit setup completed successfully."
        )

    async def run(self, **kwargs) -> Tuple[List[AuditResult], List[AuditError]]:
        """Attempt to compile the provided Java files using `javac`. If there are compilation errors, parse the output and return them as AuditErrors.

        Returns:
            Tuple[List[AuditResult], List[AuditError]]: A tuple containing a list of successful audit results and a list of audit errors.
        """
        if "files" not in kwargs:
            raise ValueError("Missing required parameter: 'files'")

        files = kwargs["files"]
        if not isinstance(files, list):
            raise ValueError("Parameter 'files' must be a list of file paths.")

        results: List[AuditResult] = []
        errors: List[AuditError] = []

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
                    errors.extend(parse_javac_output(subprocess_result.stderr))
                else:
                    logging.debug(f"Compilation succeeded for {file}.")
                    results.append(AuditResult(content=f"Compilation succeeded for {file}"))
            except Exception as e:
                logging.exception(f"An error occurred while compiling {file}: {e}")
                errors.append(
                    AuditError(
                        message=f"An error occurred while compiling {file}: {str(e)}",
                        details={"file": file},
                    )
                )

        return results, errors


def parse_javac_output(output: str) -> List[AuditError]:
    """
    Parse the output of the `javac` command to extract compilation errors. It uses regular expressions to identify error messages and their associated file, line number, and code block. The extracted information is then used to create a list of AuditError objects.

    Args:
        output (str): The output from the `javac` command.

    Returns:
        List[AuditError]: A list of AuditError objects representing the compilation errors.
    """
    errors = []
    error_pattern = re.compile(r"^(.*\.java):(\d+): (.*)$", re.MULTILINE)
    matches = error_pattern.findall(output)

    for file, line, message in matches:
        errors.append(
            AuditError(
                message=message,
                details={
                    "file": file,
                    "line": int(line),
                },
            )
        )

    return errors
