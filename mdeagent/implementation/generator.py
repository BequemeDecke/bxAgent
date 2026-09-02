from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from langchain.chat_models import BaseChatModel
from pydantic import BaseModel, Field


class TransformationClassSpec(BaseModel):
    package_name: str = Field(
        description="The Java package for the generated transformation class."
    )
    class_name: str = Field(description="The name of the transformation class.")
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

    def render_template(self, transformation_spec: TransformationClassSpec) -> str:
        return self.template.render(**transformation_spec.model_dump())


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
