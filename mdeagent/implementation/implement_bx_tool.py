import json
import re
from pathlib import Path
from typing import TypeVar

from langchain.chat_models import BaseChatModel
from pydantic import BaseModel

from mdeagent.implementation.bxtool import BxToolForEMF, BxToolTemplateResolver

from .state import ImplementationState

T = TypeVar("T", bound=BaseModel)


def _parse_yaml_like_response(
    response_content: str, model_class: type[T]
) -> T:
    """
    Parse a YAML-like or JSON response into a Pydantic model.
    
    This handles cases where the LLM returns key:value pairs instead of proper JSON.
    """
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
    
    raise ValueError(f"Failed to parse response into {model_class.__name__}. Raw content: {response_content[:500]}...")


async def _invoke_and_parse(llm, prompt: str, model_class: type[T]) -> T:
    """Invoke an LLM and parse the response with fallback handling."""
    response = await llm.ainvoke(prompt)
    
    # Handle cases where response is already a Pydantic model (e.g., in tests with mocks)
    if isinstance(response, model_class):
        return response
    
    content = response.content if hasattr(response, 'content') else str(response)
    return _parse_yaml_like_response(content, model_class)

PROMPT_TEMPLATE = """
You are a coding assistant for implementing a bx tool for EMF model transformations. 
The provided transformation class is already implemented.
Your task is to implement an adapter with the BxToolForEMF interface structure that integrates the transformation logic into a bx tool, so the transformation can be tested with benchmarx.

--- BEGIN TASK SPECIFICATION ---
{task_specification}
--- END TASK SPECIFICATION ---

--- BEGIN BX TOOL TEMPLATE ---
{template}
--- END BX TOOL TEMPLATE ---

--- BEGIN TRANSFORMATION IMPLEMENTATION ---
{transformation_implementation}
--- END TRANSFORMATION IMPLEMENTATION ---
"""


def create_input_prompt(
    task_specification: str, template: str, transformation_implementation: str
) -> str:
    return PROMPT_TEMPLATE.format(
        task_specification=task_specification,
        template=template,
        transformation_implementation=transformation_implementation,
    )


def create_implement_bx_tool_node(llm: BaseChatModel, workspace: Path, benchmarx_path: Path):
    resolver = BxToolTemplateResolver()

    async def implement_bx_tool(state: ImplementationState) -> ImplementationState:
        # 1. Collect information and construct the prompt
        task_specification = state["task_specification"]
        transformation_implementation = state["transformation_implementation"]
        bxtool_path = state["bxtool_path"]
        raw_template = resolver.get_raw_template()
        input_prompt = create_input_prompt(
            task_specification=task_specification,
            template=raw_template,
            transformation_implementation=transformation_implementation,
        )

        # 2. Invoke the llm to get the bx tool implementation
        response: BxToolForEMF = await _invoke_and_parse(llm, input_prompt, BxToolForEMF)
        bx_tool = resolver.render_template(response)

        # 3. Write the implementation to the workspace file
        bxtool_path.touch(exist_ok=True)
        bxtool_path.write_text(bx_tool)

        # NOTE: The iteration counter is advanced in the ``evaluate_implementation``
        # node (see ``agent.py``), not in the work nodes. See
        # ``implement_transformation`` for the rationale.
        return {
            "written_java_files": state.get("written_java_files", []) + [bxtool_path],
        }

    return implement_bx_tool
