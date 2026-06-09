from abc import ABC, abstractmethod
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
from typing import TypedDict


class TransformationPlanData(TypedDict):
    source_model_package: str
    target_model_package: str
    iteration: int
    source_model_implementation: str
    target_model_implementation: str
    transformation_direction: str
    difficulties: str
    implementation_steps: str


class TransformationPlanParser(ABC):
    @abstractmethod
    def parse(self) -> TransformationPlanData:
        """
        Parsed the transformation plan.

        Returns:
            TransformationPlanData: The parsed transformation plan data.
        """
        pass

    @abstractmethod
    def save(self, data: TransformationPlanData) -> None:
        pass
    

class FileTransformationPlanParser(TransformationPlanParser):
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def parse(self) -> TransformationPlanData:
        file_content = self.file_path.read_text()


class TransformationPlan:
    data: TransformationPlanData
    parser: TransformationPlanParser
    template: Template

    def __init__(self, parser: TransformationPlanParser):
        self.parser = parser
        self.template = Environment(
            loader=FileSystemLoader("bxagent/templates")
        ).get_template("transformation_plan.jinja")

        try:
            self.data = self.parser.parse()
        except Exception as e:
            print(f"Error occurred while parsing transformation plan: {e}")
            self.data = {
                "source_model_package": "",
                "target_model_package": "",
                "iteration": 0,
                "source_model_implementation": "",
                "target_model_implementation": "",
                "transformation_direction": "",
                "difficulties": "",
                "implementation_steps": "",
            }

    def __str__(self):
        rendered_content = self.template.render(
            source_model=self.data["source_model_package"],
            target_model=self.data["target_model_package"],
            iteration=self.data["iteration"],
            source_model_implementation=self.data["source_model_implementation"],
            target_model_implementation=self.data["target_model_implementation"],
            transformation_direction=self.data["transformation_direction"],
            difficulties=self.data["difficulties"],
            implementation_steps=self.data["implementation_steps"],
        )
        return rendered_content

    def update_package_information(
        self, source_model_package: str, target_model_package: str
    ):
        self.data["source_model_package"] = source_model_package
        self.data["target_model_package"] = target_model_package
        self.parser.save(str(self))

    def update_iteration(self, iteration: int):
        self.data["iteration"] = iteration
        self.parser.save(str(self))

    def update_model_implementation(
        self, source_model_implementation: str, target_model_implementation: str
    ):
        self.data["source_model_implementation"] = source_model_implementation
        self.data["target_model_implementation"] = target_model_implementation
        self.parser.save(str(self))

    def update_transformation_direction(self, transformation_direction: str):
        self.data["transformation_direction"] = transformation_direction
        self.parser.save(str(self))

    def update_transformation_difficulties(self, difficulties: str):
        self.data["difficulties"] = difficulties
        self.parser.save(str(self))

    def update_implementation_steps(self, implementation_steps: str):
        self.data["implementation_steps"] = implementation_steps
        self.parser.save(str(self))
