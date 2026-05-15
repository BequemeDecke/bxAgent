from os import PathLike
from typing import Optional
from pydantic import BaseModel


class TestingOutputSchema(BaseModel):
    """
    Schema for the output of the testing agent.
    """

    file_structure_errors: dict[PathLike, list[Exception]]
    compilation_errors: dict[PathLike, list[Exception]]
    benchmark_error: Optional[Exception]
    result_path: Optional[PathLike]
