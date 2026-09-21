from abc import ABC, abstractmethod

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationRun


class TransformationClassGenerator(ABC):
    @abstractmethod
    def synthesize_transformation_class(
        self,
        tp: TransformationPlan,
        transformation_class_name: str,
        transformation_package: str,
        specific_task: str | None = None,
        evaluation_results: dict[str, EvaluationRun] | None = None,
    ) -> str:
        """Synthesizes the transformation class based on the provided transformation plan and an optional specific task.

        Args:
            tp (TransformationPlan): The transformation plan to use for generating the transformation class.
            transformation_class_name (str): The name of the transformation class to generate.
            transformation_package (str): The package of the transformation class to generate.
            specific_task (str | None, optional): An optional specific task to focus on when generating the transformation class. Defaults to None.
            evaluation_results (dict[str, EvaluationRun] | None, optional): Optional evaluation results that can be used to inform the generation of the transformation class. Defaults to None.

        Returns:
            str: The generated transformation class as a string."""
