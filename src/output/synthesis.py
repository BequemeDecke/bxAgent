"""
This module defines the output schema for the synthesis agent. It specifies the structure of the data that the synthesis agent will return after it has completed its task. The schema includes a list of written files, which are represented as PathLike objects. This allows for a standardized way to represent the output of the synthesis agent, making it easier to integrate with other components of the system.

Following these guides:
https://docs.langchain.com/oss/python/langchain/structured-output
https://docs.langchain.com/oss/python/deepagents/customization#structured-output
"""

from os import PathLike
from pydantic import BaseModel, Field


class SynthesisOutputSchema(BaseModel):
    """
    Schema for the output of the synthesis agent.
    """

    written_files: list[PathLike] = Field(
        description="A list of paths to the written files."
    )
    is_transformation_possible: bool = Field(
        description="Indicates whether the transformation is possible."
    )
    reason: str = Field(description="The reason for the transformation result.")
