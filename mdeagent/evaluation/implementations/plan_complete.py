from pydantic import BaseModel

from mdeagent.comprehension.plan import SerializedTransformationPlan
from mdeagent.evaluation.types import Evaluation, EvaluationError, EvaluationResult


class PlanCompleteSchema(BaseModel):
    """TypedDict schema describing the parameters expected by
    ``TransformationPlanEvaluation``.

    Pydantic's ``model_validate`` is used by the ``EvaluationExecutor``
    to validate the kwargs before they reach ``Evaluation.run``.
    We provide a ``model_validate`` classmethod so the executor can
    treat this as a regular Pydantic model.
    """

    transformation_plan: SerializedTransformationPlan | None


class PlanCompleteEvaluation(Evaluation):
    """Evaluates whether the transformation plan has all required fields populated."""

    def __init__(self) -> None:
        super().__init__()
        self.REQUIRED_FIELDS = [
            ("source_model_implementation", "Source model implementation"),
            ("target_model_implementation", "Target model implementation"),
            ("transformation_direction", "Transformation direction"),
            ("difficulties", "Transformation difficulties"),
            ("implementation_steps", "Implementation steps"),
        ]

    async def setup(self) -> None:
        pass

    async def run(
        self, **kwargs
    ) -> tuple[list[EvaluationResult], list[EvaluationError]]:
        serialized_plan: SerializedTransformationPlan = kwargs.get(
            "transformation_plan"
        )

        if serialized_plan is None or "data" not in serialized_plan:
            return (
                [
                    EvaluationResult(
                        content="No transformation plan provided.",
                        metadata={"success": False, "include_in_report": True},
                    )
                ],
                [],
            )

        results: list[EvaluationResult] = []

        for field_key, label in self.REQUIRED_FIELDS:
            value = str(serialized_plan["data"].get(field_key, "")).strip()
            if value:
                results.append(
                    EvaluationResult(
                        content=f"{label} is populated.",
                        metadata={"success": True, "include_in_report": False},
                    )
                )
            else:
                results.append(
                    EvaluationResult(
                        content=f"{label} is NOT populated.",
                        metadata={"success": False, "include_in_report": True},
                    )
                )

        return results, []
