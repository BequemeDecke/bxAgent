"""
This module defines the output schema for the synthesis agent. It specifies the structure of the data that the synthesis agent will return after it has completed its task. The schema includes a list of written files, which are represented as PathLike objects. This allows for a standardized way to represent the output of the synthesis agent, making it easier to integrate with other components of the system.

Some LLMs do not support response formats
"For models that don’t support native structured output, LangChain uses tool calling to achieve the same result."

Following these guides:
https://docs.langchain.com/oss/python/langchain/structured-output
https://docs.langchain.com/oss/python/deepagents/customization#structured-output
"""

from pydantic import BaseModel, Field


class SynthesisResponseFormat(BaseModel):
    implementation_instructions: str = Field(
        description="The implementation instructions for the transformation class. This should include a detailed description of how to implement the transformation class, including any important details or considerations that should be taken into account during implementation."
    )
