from mdeagent.evaluation.types import Evaluation, EvaluationError, EvaluationResult


class TransformationPlanSchema:
    """TypedDict schema describing the parameters expected by
    ``TransformationPlanEvaluation``.

    Pydantic's ``model_validate`` is used by the ``EvaluationExecutor``
    to validate the kwargs before they reach ``Evaluation.run``.
    We provide a ``model_validate`` classmethod so the executor can
    treat this as a regular Pydantic model.
    """

    @classmethod
    def model_validate(cls, data: dict) -> "TransformationPlanSchema":
        """Validate and return the input dict as-is (no transformation)."""
        if "transformation_plan" not in data:
            raise ValueError("Missing required field: transformation_plan")
        return cls()

    @classmethod
    def model_dump(cls) -> dict:
        return {}


class TransformationPlanEvaluation(Evaluation):
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

    @staticmethod
    def _get_plan_data(tp_dict: dict) -> dict:
        """Extract the plan data dict from a SerializedTransformationPlan.

        Handles both the full serialized form (with nested ``data`` key) and
        a flat dict that was passed directly.
        """
        if "data" in tp_dict:
            return tp_dict["data"]
        return tp_dict

    async def run(
        self, **kwargs
    ) -> tuple[list[EvaluationResult], list[EvaluationError]]:
        tp_dict = kwargs.get("transformation_plan")

        if tp_dict is None or not tp_dict:
            return (
                [
                    EvaluationResult(
                        content="No transformation plan provided.",
                        metadata={"success": False, "include_in_report": True},
                    )
                ],
                [],
            )

        plan_data = self._get_plan_data(tp_dict)
        results: list[EvaluationResult] = []

        for field_key, label in self.REQUIRED_FIELDS:
            value = str(plan_data.get(field_key, "")).strip()
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
