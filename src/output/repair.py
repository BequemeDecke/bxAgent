from os import PathLike
from pydantic import BaseModel


class RepairOutputSchema(BaseModel):
    """
    Schema for the output of the repair agent.
    """

    repaired_files: list[PathLike]
