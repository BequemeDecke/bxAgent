"""
This module defines the output schema for the repair agent. It specifies the structure of the data that the repair agent will return after it has completed its task. The schema includes a list of repaired files, which are represented as PathLike objects. This allows for a standardized way to represent the output of the repair agent, making it easier to integrate with other components of the system.

Following these guides:
https://docs.langchain.com/oss/python/langchain/structured-output
https://docs.langchain.com/oss/python/deepagents/customization#structured-output
"""

from os import PathLike
from pydantic import BaseModel, Field


class RepairOutputSchema(BaseModel):
    """
    Schema for the output of the repair agent.
    """

    repaired_files: list[PathLike] = Field(
        description="A list of paths to the repaired files."
    )
