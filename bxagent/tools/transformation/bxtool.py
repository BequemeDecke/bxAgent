from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel


class Decisions(BaseModel):
    import_path: str


class Class(BaseModel):
    import_path: str
    class_name: str
    instance_name: str


class InitiationDialogue(BaseModel):
    set_configuration: str
    initiate_dialogue: str


class TransformationImplementation(Class):
    decisions: Decisions
    initiation_dialogue: InitiationDialogue
    perform_and_propagate_target_edit: str
    perform_and_propagate_source_edit: str
    perform_and_propagate_concurrent_edit: str


class Model(BaseModel):
    name: str
    factory: Class
    registry: Class
    comparator: Class


class BxToolForEMF(BaseModel):
    transformation_package: str
    transformation_implementation: TransformationImplementation
    source_model: Model
    target_model: Model
    additional_imports: list[str]


class BxToolTemplateResolver:
    template: Template

    def __init__(self, template_path: Path = Path.cwd() / "templates"):
        self.template = Environment(
            loader=FileSystemLoader(template_path)
        ).get_template("bxtool.jinja")

    def render_template(self, bx_tool_for_emf: BxToolForEMF) -> str:
        return self.template.render(**bx_tool_for_emf.model_dump())
