import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mdeagent.evaluation.types import Evaluation, EvaluationError, EvaluationResult
from mdeagent.preparation.maven import MavenProject

logger = logging.getLogger(__name__)


@dataclass
class JavaCompilationMetadata:
    """Metadata for Java compilation evaluation results."""

    success: bool
    file: str | None = None
    line: int | None = None
    column: int | None = None
    error_type: str | None = None


class JavaCompilationEvaluationConfig(BaseModel):
    project_path: Path


class JavaCompilationEvaluation(Evaluation):
    """
    Evaluation that checks if a Maven project can be compiled successfully.

    It uses the `mvn compile` tool to attempt to compile the Maven project.
    The output is parsed using regular expressions to extract syntax errors
    and other compilation issues.
    """

    def __init__(
        self,
    ):
        """
        Initialize the JavaCompilationEvaluation.
        """

    async def setup(self, **kwargs) -> None:
        """
        Setup method (no-op). The Maven project is loaded lazily in run().

        :param kwargs: Configuration arguments, must include 'project_path'.
        """
        pass

    async def run(
        self, **kwargs
    ) -> tuple[list[EvaluationResult], list[EvaluationError]]:
        """
        Compile the Maven project and parse any compilation errors.

        :param kwargs: Configuration arguments, must include 'project_path'.
        :return: A tuple containing a list of evaluation results and a list of evaluation errors.
        """
        project_path = kwargs.get("project_path", None)
        if project_path is None:
            raise ValueError("project_path is required (passed to constructor or run())")

        try:
            self._maven_project = MavenProject.load(project_path)
        except Exception as e:
            logger.exception(f"Failed to load Maven project at {project_path}")
            return [], [
                EvaluationError(
                    message=f"Failed to load Maven project at {project_path}: {e}",
                    type="MavenProjectLoadError",
                    details={"project_path": str(project_path)},
                )
            ]

        results: list[EvaluationResult] = []
        errors: list[EvaluationError] = []

        try:
            success, output = self._maven_project.compile()

            if success:
                results.append(
                    EvaluationResult(
                        content="Maven project compiled successfully.",
                        metadata={"success": True},
                    )
                )
            else:
                parsed_results = parse_mvn_compile_output(output)
                results.extend(parsed_results)

        except Exception as e:
            logger.exception(
                f"An error occurred while compiling the Maven project: {e}"
            )
            errors.append(
                EvaluationError(
                    message=f"An error occurred while compiling the Maven project: {str(e)}",
                    type=type(e).__name__,
                    details={"project_path": str(self._maven_project.workspace)},
                )
            )

        return results, errors


def parse_mvn_compile_output(output: str) -> list[EvaluationResult]:
    """
    Parse the output of the `mvn compile` tool to extract compilation errors.

    The parser handles various error formats from the Java compiler (javac) as
    invoked by Maven. It extracts file paths, line numbers, columns, and error messages.

    Example error formats:
    - [ERROR] /path/to/File.java:[line:column] error message
    - /path/to/File.java:line: error message
    - /path/to/File.java:line:column: error message

    Args:
        output (str): The output from the `mvn compile` tool.

    Returns:
        list[EvaluationResult]: A list of EvaluationResult objects representing the compilation errors.
    """
    results: list[EvaluationResult] = []

    # Pattern 1: Maven error format with [ERROR] prefix
    # [ERROR] /path/to/File.java:[line:column] error message
    maven_error_pattern = re.compile(
        r"^\[ERROR\]\s+([^\s:]+\.java):\[(\d+):(\d+)\]\s+(.+)$",
        re.MULTILINE,
    )

    # Pattern 2: Direct javac format: /path/to/File.java:line: error message
    javac_line_pattern = re.compile(
        r"^([^\s:]+\.java):(\d+):\s*(.+)$",
        re.MULTILINE,
    )

    # Pattern 3: Direct javac format with column: /path/to/File.java:line:column: error message
    javac_column_pattern = re.compile(
        r"^([^\s:]+\.java):(\d+):(\d+):\s*(.+)$",
        re.MULTILINE,
    )

    # First try Maven error format
    for match in maven_error_pattern.finditer(output):
        file_path = match.group(1)
        line = int(match.group(2))
        column = int(match.group(3))
        message = match.group(4).strip()

        results.append(
            EvaluationResult(
                content=message,
                metadata={
                    "success": False,
                    "file": file_path,
                    "line": line,
                    "column": column,
                },
            )
        )

    # If no Maven errors found, try javac patterns
    if not results:
        # Try column pattern first (more specific)
        for match in javac_column_pattern.finditer(output):
            file_path = match.group(1)
            line = int(match.group(2))
            column = int(match.group(3))
            message = match.group(4).strip()

            results.append(
                EvaluationResult(
                    content=message,
                    metadata={
                        "success": False,
                        "file": file_path,
                        "line": line,
                        "column": column,
                    },
                )
            )

        # Then try line-only pattern
        if not results:
            for match in javac_line_pattern.finditer(output):
                file_path = match.group(1)
                line = int(match.group(2))
                message = match.group(3).strip()

                results.append(
                    EvaluationResult(
                        content=message,
                        metadata={
                            "success": False,
                            "file": file_path,
                            "line": line,
                        },
                    )
                )

    return results
