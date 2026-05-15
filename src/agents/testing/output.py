"""
This module defines the output schema for the testing agent. It specifies the structure of the data that the testing agent will return after it has completed its task. The schema includes a list of test results, which are represented as PathLike objects. This allows for a standardized way to represent the output of the testing agent, making it easier to integrate with other components of the system.

Following these guides:
https://docs.langchain.com/oss/python/langchain/structured-output
https://docs.langchain.com/oss/python/deepagents/customization#structured-output
"""

from typing import Optional
from pydantic import BaseModel, Field, FilePath


class TestingOutputSchema(BaseModel):
    """
    Schema for the output of the testing agent.
    """

    file_structure_errors: dict[FilePath, list[tuple[str, str]]] = Field(
        description="A dictionary mapping file paths to lists of structure errors."
    )
    compilation_errors: dict[FilePath, list[tuple[str, str]]] = Field(
        description="A dictionary mapping file paths to lists of compilation errors."
    )
    benchmark_error: Optional[tuple[str, str]] = Field(
        description="The error encountered during benchmarking, if any."
    )
    result_path: Optional[FilePath] = Field(description="The path to the test results.")
