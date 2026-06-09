import re

from abc import ABC, abstractmethod
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
from typing import Literal, TypedDict


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
    def save(self, data: str) -> None:
        pass


class FileTransformationPlanParser(TransformationPlanParser):
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def _parse_header(self, content: str) -> dict:
        """
        Parses the header of the transformation plan file to extract source_model_package, target_model_package, and iteration.

        ---
        source_model_package: {{source_model_package}}
        target_model_package: {{target_model_package}}
        iteration: {{iteration}}
        ---

        Args:
            content (str): The content of the transformation plan file.
        Returns:
            dict: A dictionary containing source_model_package, target_model_package, and iteration.
        """
        match = re.search(
            r"---\s*source_model_package:\s*(?P<source_model_package>.*?)\s*target_model_package:\s*(?P<target_model_package>.*?)\s*iteration:\s*(?P<iteration>\d+)\s*---",
            content,
            re.DOTALL,
        )

        if not match:
            raise ValueError("Failed to parse transformation plan header.")

        return {
            "source_model_package": match.group("source_model_package"),
            "target_model_package": match.group("target_model_package"),
            "iteration": int(match.group("iteration")),
        }

    def _parse_model_implementation(self, content: str, model_type: Literal["source", "target"]) -> str:
        """
        Parses the source or target model implementation from the transformation plan file.
        With following regex:
        ```md
        --- BEGIN SOURCE MODEL ---
        {{source_model_implementation}}
        --- END SOURCE MODEL ---
        ```
        """
        if model_type == "source":
            pattern = r"---\s*BEGIN SOURCE MODEL\s*---\s*(?P<source_model_implementation>.*?)\s*---\s*END SOURCE MODEL\s*---"
        else:
            pattern = r"---\s*BEGIN TARGET MODEL\s*---\s*(?P<target_model_implementation>.*?)\s*---\s*END TARGET MODEL\s*---"

        match = re.search(
            pattern,
            content,
            re.DOTALL,
        )

        if not match:
            raise ValueError("Failed to parse model implementation.")

        return match.group(f"{model_type}_model_implementation")

    def parse(self) -> TransformationPlanData:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Transformation plan file not found at {self.file_path}"
            )
        content = self.file_path.read_text()
        if content.strip() == "":
            raise ValueError("The transformation plan file is empty.")

        header_data = self._parse_header(content)
        source_model_implementation = self._parse_model_implementation(content, "source")
        target_model_implementation = self._parse_model_implementation(content, "target")

        return {
            "source_model_package": header_data["source_model_package"],
            "target_model_package": header_data["target_model_package"],
            "iteration": header_data["iteration"],
            "source_model_implementation": source_model_implementation,
            "target_model_implementation": target_model_implementation,
            "transformation_direction": "",
            "difficulties": "",
            "implementation_steps": "",
        }

    def save(self, data: str) -> None:
        self.file_path.write_text(data)


class TransformationPlan:
    data: TransformationPlanData
    parser: TransformationPlanParser
    template: Template

    def __init__(self, parser: TransformationPlanParser):
        self.parser = parser
        self.template = Environment(loader=FileSystemLoader("templates")).get_template(
            "transformation_plan.jinja"
        )

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

    def __str__(self) -> str:
        rendered_content = self.template.render(
            source_model_package=self.data["source_model_package"],
            target_model_package=self.data["target_model_package"],
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
