"""
This module defines the tool strategy for handling structured output in the synthesis agent.

See https://docs.langchain.com/oss/python/langchain/structured-output#tool-calling-strategy
"""


from typing import Generic, TypeVar, Union, Callable
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ]
    