import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

from jinja2 import Environment, FileSystemLoader, Template
from langchain.chat_models import BaseChatModel
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class StructuredResponseParser(ABC):
    """Interface for parsing LLM responses into structured Pydantic models.
    
    This abstraction allows handling different response formats (JSON, YAML-like)
    from various LLM providers.
    """
    
    @abstractmethod
    def parse(self, response_content: str, model_class: type[T]) -> T:
        """Parse response content into a Pydantic model instance.
        
        Args:
            response_content: Raw text response from the LLM.
            model_class: The Pydantic model class to parse into.
            
        Returns:
            A validated instance of the Pydantic model.
            
        Raises:
            ValueError: If the response cannot be parsed into the target model.
        """
        pass


class JsonParser(StructuredResponseParser):
    """Parser for JSON-formatted LLM responses."""
    
    def parse(self, response_content: str, model_class: type[T]) -> T:
        """Parse JSON response into a Pydantic model."""
        data = json.loads(response_content)
        return model_class.model_validate(data)


class YamlLikeParser(StructuredResponseParser):
    """Parser for YAML-like or plain text LLM responses.
    
    Handles cases where the LLM returns key:value pairs instead of proper JSON,
    including markdown formatting, bold keys, and code blocks.
    """
    
    def parse(self, response_content: str, model_class: type[T]) -> T:
        """Parse YAML-like response into a Pydantic model with fallback strategies."""
        # First try parsing as JSON directly
        try:
            data = json.loads(response_content)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response_content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return model_class.model_validate(data)
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Get expected field names from the Pydantic model
        model_fields = set(model_class.model_fields.keys())
        
        # Convert various text formats to dict
        data = {}
        for line in response_content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Pattern 1: **Key:** value or **Key:** `value`
            bold_match = re.match(r'^\*\*([^:]+):\*\*\s*(.+)$', line)
            if bold_match:
                key = bold_match.group(1).strip().lower().replace(' ', '_')
                value = bold_match.group(2).strip()
                value = re.sub(r'`([^`]*)`', r'\1', value)
                value = re.sub(r'\([^)]*\)$', '', value).strip()
                value = value.rstrip('.,;:')
                if key in model_fields:
                    data[key] = value
                continue
            
            # Pattern 2: Key: value (simple YAML style)
            yaml_match = re.match(r'^([A-Za-z][A-Za-z0-9_ ]*):\s*(.+)$', line)
            if yaml_match:
                key = yaml_match.group(1).strip().lower().replace(' ', '_')
                value = yaml_match.group(2).strip()
                value = re.sub(r'`([^`]*)`', r'\1', value)
                value = re.sub(r'\([^)]*\)$', '', value).strip()
                value = value.rstrip('.,;:')
                if key in model_fields:
                    data[key] = value
                continue
        
        if data:
            return model_class.model_validate(data)
        
        # Special handling for single-field models
        if len(model_fields) == 1:
            field_name = list(model_fields)[0]
            code_match = re.search(r'```(?:\w+)?\s*([\s\S]*?)```', response_content)
            if code_match:
                data[field_name] = code_match.group(1).strip()
                return model_class.model_validate(data)
            data[field_name] = response_content.strip()
            return model_class.model_validate(data)
        
        raise ValueError(
            f"Failed to parse response into {model_class.__name__}. "
            f"Raw content: {response_content[:500]}..."
        )


class FallbackParser(StructuredResponseParser):
    """Parser that tries multiple parsers in sequence until one succeeds.
    
    This is the default parser used by the CodeGenerator, providing maximum
    flexibility for handling different LLM response formats.
    """
    
    def __init__(self, parsers: list[StructuredResponseParser] | None = None):
        """Initialize with a list of parsers to try in order.
        
        Args:
            parsers: List of parsers to try. Defaults to [JsonParser(), YamlLikeParser()].
        """
        self.parsers = parsers or [JsonParser(), YamlLikeParser()]
    
    def parse(self, response_content: str, model_class: type[T]) -> T:
        """Try each parser in sequence until one succeeds."""
        last_error: Exception | None = None
        for parser in self.parsers:
            try:
                return parser.parse(response_content, model_class)
            except Exception as e:
                last_error = e
                continue
        
        if last_error:
            raise last_error
        raise ValueError(f"All parsers failed for {model_class.__name__}")


def invoke_and_parse(
    llm: BaseChatModel,
    prompt: str,
    model_class: type[T],
    parser: StructuredResponseParser | None = None,
) -> T:
    """Invoke an LLM synchronously and parse the response.
    
    Args:
        llm: The chat model to invoke.
        prompt: The prompt to send to the LLM.
        model_class: The Pydantic model class to parse the response into.
        parser: Parser to use. Defaults to FallbackParser().
        
    Returns:
        Parsed Pydantic model instance.
    """
    parser = parser or FallbackParser()
    response = llm.invoke(prompt)
    
    # Handle cases where response is already a Pydantic model (e.g., in tests with mocks)
    if isinstance(response, model_class):
        return response
    
    content = response.content if hasattr(response, 'content') else str(response)
    return parser.parse(content, model_class)


async def ainvoke_and_parse(
    llm: BaseChatModel,
    prompt: str,
    model_class: type[T],
    parser: StructuredResponseParser | None = None,
) -> T:
    """Invoke an LLM asynchronously and parse the response.
    
    Args:
        llm: The chat model to invoke.
        prompt: The prompt to send to the LLM.
        model_class: The Pydantic model class to parse the response into.
        parser: Parser to use. Defaults to FallbackParser().
        
    Returns:
        Parsed Pydantic model instance.
    """
    parser = parser or FallbackParser()
    response = await llm.ainvoke(prompt)
    
    # Handle cases where response is already a Pydantic model (e.g., in tests with mocks)
    if isinstance(response, model_class):
        return response
    
    content = response.content if hasattr(response, 'content') else str(response)
    return parser.parse(content, model_class)


class CodeGenerator:
    """Generator class for producing code from templates using structured LLM responses.
    
    This class coordinates the generation of code by:
    1. Taking a template and schema (Pydantic model) as input
    2. Using the LLM to fill in the template via structured output
    3. Supporting both sync (invoke) and async (astream) generation
    4. Using a pluggable parser interface for different response formats
    
    Attributes:
        llm: The chat model used for generation.
        parser: Parser for converting LLM responses to structured data.
        template_resolver: Resolver for loading and rendering templates.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        parser: StructuredResponseParser | None = None,
        template_path: Path = Path.cwd() / "templates",
    ):
        """Initialize the code generator.
        
        Args:
            llm: The chat model to use for generation.
            parser: Parser for LLM responses. Defaults to FallbackParser().
            template_path: Path to the templates directory.
        """
        self.llm = llm
        self.parser = parser or FallbackParser()
        self.template_path = template_path
    
    def generate(
        self,
        prompt: str,
        model_class: type[T],
        template_resolver: type,
        **template_kwargs,
    ) -> tuple[T, str]:
        """Generate code synchronously.
        
        Args:
            prompt: The prompt to send to the LLM.
            model_class: The Pydantic model class for structured output.
            template_resolver: Template resolver class (e.g., TransformationClassTemplateResolver).
            **template_kwargs: Additional kwargs for template rendering.
            
        Returns:
            Tuple of (parsed_model, rendered_code).
        """
        parsed_model = invoke_and_parse(self.llm, prompt, model_class, self.parser)
        
        # Handle class_name in template_kwargs or from model
        if 'class_name' not in template_kwargs and hasattr(parsed_model, 'class_name'):
            template_kwargs['class_name'] = parsed_model.class_name
        
        resolver = template_resolver(template_path=self.template_path)
        rendered_code = resolver.render_template(parsed_model, **template_kwargs)
        
        return parsed_model, rendered_code
    
    async def generate_async(
        self,
        prompt: str,
        model_class: type[T],
        template_resolver: type,
        **template_kwargs,
    ) -> tuple[T, str]:
        """Generate code asynchronously.
        
        Args:
            prompt: The prompt to send to the LLM.
            model_class: The Pydantic model class for structured output.
            template_resolver: Template resolver class (e.g., TransformationClassTemplateResolver).
            **template_kwargs: Additional kwargs for template rendering.
            
        Returns:
            Tuple of (parsed_model, rendered_code).
        """
        parsed_model = await ainvoke_and_parse(self.llm, prompt, model_class, self.parser)
        
        # Handle class_name in template_kwargs or from model
        if 'class_name' not in template_kwargs and hasattr(parsed_model, 'class_name'):
            template_kwargs['class_name'] = parsed_model.class_name
        
        resolver = template_resolver(template_path=self.template_path)
        rendered_code = resolver.render_template(parsed_model, **template_kwargs)
        
        return parsed_model, rendered_code
    
    async def generate_streaming(
        self,
        prompt: str,
        model_class: type[T],
        template_resolver: type,
        on_chunk: Callable | None = None,
        **template_kwargs,
    ) -> tuple[T, str]:
        """Generate code with streaming support.
        
        Args:
            prompt: The prompt to send to the LLM.
            model_class: The Pydantic model class for structured output.
            template_resolver: Template resolver class.
            on_chunk: Optional callback for each streamed chunk.
            **template_kwargs: Additional kwargs for template rendering.
            
        Returns:
            Tuple of (parsed_model, rendered_code).
        """
        # Collect streamed chunks
        chunks = []
        async for chunk in self.llm.astream(prompt):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunks.append(content)
            if on_chunk:
                on_chunk(content)
        
        # Parse the complete response
        full_content = ''.join(chunks)
        parsed_model = self.parser.parse(full_content, model_class)
        
        # Handle class_name in template_kwargs or from model
        if 'class_name' not in template_kwargs and hasattr(parsed_model, 'class_name'):
            template_kwargs['class_name'] = parsed_model.class_name
        
        resolver = template_resolver(template_path=self.template_path)
        rendered_code = resolver.render_template(parsed_model, **template_kwargs)
        
        return parsed_model, rendered_code
    
    async def generate_parallel(
        self,
        prompts_with_models: list[tuple[str, type[T]]],
    ) -> list[T]:
        """Generate multiple structured responses in parallel.
        
        Args:
            prompts_with_models: List of (prompt, model_class) tuples.
            
        Returns:
            List of parsed Pydantic model instances.
        """
        import asyncio
        
        async def parse_one(prompt: str, model_class: type[T]) -> T:
            return await ainvoke_and_parse(self.llm, prompt, model_class, self.parser)
        
        tasks = [parse_one(prompt, model_class) for prompt, model_class in prompts_with_models]
        return await asyncio.gather(*tasks)


class _TransformationClassFields(BaseModel):
    """Shared fields describing the *body* of a transformation class.

    The class name is intentionally not part of this base model: naming is now
    decided in the ``prepare_workspace`` node (see
    :mod:`mdeagent.preparation.naming`) and the ``implement_transformation``
    node must no longer ask the LLM for a name. The legacy
    :class:`TransformationClassSpec` (used by
    :func:`create_generate_transformation_node`) keeps a ``class_name`` field,
    while :class:`ImplementationTransformationSpec` (used by
    :func:`mdeagent.implementation.implement_transformation.create_implement_transformation_node`)
    does not and receives the name separately via
    :meth:`TransformationClassTemplateResolver.render_template`.
    """

    package_name: str = Field(
        description="The Java package for the generated transformation class."
    )
    source_type: str = Field(
        description="The source model type used in AgentTransformationForEMF."
    )
    target_type: str = Field(
        description="The target model type used in AgentTransformationForEMF."
    )
    decision_type: str = Field(
        description="The decision type used in AgentTransformationForEMF."
    )
    transformation_package: str = Field(
        default="com.example",
        description="The package where AgentTransformationForEMF is declared.",
    )
    fields: list[dict] = Field(default_factory=list)
    constructor: dict | None = Field(default=None)
    forward_body: str | None = Field(default=None)
    backward_body: str | None = Field(default=None)
    synch_body: str | None = Field(default=None)
    transform_source_to_target_body: str | None = Field(default=None)
    transform_target_to_source_body: str | None = Field(default=None)


class TransformationClassSpec(_TransformationClassFields):
    """Legacy structured-output spec that *also* asks the LLM for the class name.

    Only used by :func:`create_generate_transformation_node`. The
    ``implement_transformation`` node uses
    :class:`ImplementationTransformationSpec` instead so that it does not ask
    the LLM for a name (the name is determined in ``prepare_workspace``).
    """

    class_name: str = Field(description="The name of the transformation class.")


class TransformationClassMetadata(BaseModel):
    """Structured-output spec for transformation metadata (package and type names).

    This is the first step in the piecewise generation approach. The metadata
    determines the basic structure and is used as context for generating fields
    and method bodies.
    """

    package_name: str = Field(
        description="The Java package for the generated transformation class."
    )
    source_type: str = Field(
        description="The source model type used in AgentTransformationForEMF."
    )
    target_type: str = Field(
        description="The target model type used in AgentTransformationForEMF."
    )
    decision_type: str = Field(
        description="The decision type used in AgentTransformationForEMF."
    )
    transformation_package: str = Field(
        default="com.example",
        description="The package where AgentTransformationForEMF is declared.",
    )


class TransformationFieldsAndConstructor(BaseModel):
    """Structured-output spec for class fields and constructor.

    Generated in parallel with method bodies, using metadata as context.
    """

    fields: list[dict] = Field(
        default_factory=list,
        description="List of field declarations with 'type' and 'name'.",
    )
    constructor: dict | None = Field(
        default=None,
        description="Constructor with 'parameters' and 'assignments', or null if no constructor needed.",
    )


class ForwardMethodBody(BaseModel):
    """Structured-output spec for the forward method body."""

    forward_body: str = Field(
        description="Java code for the forward transformation method body."
    )


class BackwardMethodBody(BaseModel):
    """Structured-output spec for the backward method body."""

    backward_body: str = Field(
        description="Java code for the backward transformation method body."
    )


class SynchMethodBody(BaseModel):
    """Structured-output spec for the synch method body."""

    synch_body: str = Field(
        description="Java code for the synchronization method body."
    )


class ImplementationTransformationSpec(_TransformationClassFields):
    """Structured-output spec for the ``implement_transformation`` node.

    Unlike :class:`TransformationClassSpec` this spec does **not** contain a
    ``class_name`` field: the ``implement_transformation`` node no longer asks
    the LLM for a name. The class name is determined in the ``prepare_workspace``
    node and reaches this node encoded in the ``transformation_class_path``
    state field. It is passed to
    :meth:`TransformationClassTemplateResolver.render_template` separately.

    DEPRECATED: Use the piecewise generation approach with separate specs instead.
    This class is kept for backwards compatibility but should not be used for new
    implementations.
    """


PROMPT_TEMPLATE = """
You are a Java transformation code generator for EMF-based model transformations.
Generate a concrete implementation of the AgentTransformationForEMF interface based on the task specification and the provided template.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN TEMPLATE ---
{template}
--- END TEMPLATE ---

Return a valid structured result matching the required Java class structure.
The implementation must use the EMF interface methods and the Java generic types for source, target, and decisions.
"""


class TransformationClassTemplateResolver:
    template: Template

    def __init__(self, template_path: Path = Path.cwd() / "templates"):
        self.template = Environment(
            loader=FileSystemLoader(template_path)
        ).get_template("transformation_class.jinja")
        self.raw_template = (template_path / "transformation_class.jinja").read_text()

    def get_raw_template(self) -> str:
        return self.raw_template

    def render_template(
        self,
        transformation_spec: _TransformationClassFields,
        class_name: str | None = None,
    ) -> str:
        """Render the transformation class template.

        ``class_name`` may be passed explicitly for specs that do not carry a
        class name themselves (i.e. :class:`ImplementationTransformationSpec`).
        When ``class_name`` is ``None`` the value already present in
        ``transformation_spec`` (e.g. for the legacy
        :class:`TransformationClassSpec`) is used.
        """
        data = transformation_spec.model_dump()
        if class_name is not None:
            data["class_name"] = class_name
        return self.template.render(**data)


def create_generate_transformation_node(llm: BaseChatModel, workspace: Path):
    resolver = TransformationClassTemplateResolver()

    def generate_transformation(state: dict) -> dict:
        task_specification = state["task_specification"]
        raw_template = resolver.get_raw_template()
        input_prompt = PROMPT_TEMPLATE.format(
            task_specification=task_specification,
            template=raw_template,
        )

        response: TransformationClassSpec = invoke_and_parse(llm, input_prompt, TransformationClassSpec)
        rendered_code = resolver.render_template(response)

        file_name = response.class_name + ".java"
        file_path = workspace / file_name
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
            file_path.touch()
        file_path.write_text(rendered_code)

        return {
            "written_java_files": state.get("written_java_files", []) + [file_path],
            "transformation_implementation": rendered_code,
        }

    return generate_transformation
