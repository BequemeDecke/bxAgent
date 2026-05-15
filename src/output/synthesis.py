from os import PathLike
from pydantic import BaseModel


class SynthesisOutputSchema(BaseModel):
    """
    Schema for the output of the synthesis agent.
    """

    written_files: list[PathLike]
    is_transformation_possible: bool
    reason: str
