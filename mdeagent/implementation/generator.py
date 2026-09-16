from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from langchain.chat_models import BaseChatModel
from pydantic import BaseModel, Field


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


class ImplementationTransformationSpec(_TransformationClassFields):
    """Structured-output spec for the ``implement_transformation`` node.

    Unlike :class:`TransformationClassSpec` this spec does **not** contain a
    ``class_name`` field: the ``implement_transformation`` node no longer asks
    the LLM for a name. The class name is determined in the ``prepare_workspace``
    node and reaches this node encoded in the ``transformation_class_path``
    state field. It is passed to
    :meth:`TransformationClassTemplateResolver.render_template` separately.
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
    structured_llm = llm.with_structured_output(TransformationClassSpec)
    resolver = TransformationClassTemplateResolver()

    def generate_transformation(state: dict) -> dict:
        task_specification = state["task_specification"]
        raw_template = resolver.get_raw_template()
        input_prompt = PROMPT_TEMPLATE.format(
            task_specification=task_specification,
            template=raw_template,
        )

        response: TransformationClassSpec = structured_llm.invoke(input=input_prompt)
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
