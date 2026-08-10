import asyncio
import datetime

from typing import List, Dict, Tuple, TypedDict, Any
from pydantic import BaseModel

from .types import Evaluation, EvaluationResult, EvaluationRun, EvaluationError


class EvaluationInit(TypedDict):
    evaluation: Evaluation
    evaluation_schema: BaseModel


class LinkedEvaluation(Evaluation):
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation

    async def setup(self):
        await self.evaluation.setup()

    async def run(
        self, **kwargs
    ) -> Tuple[List[EvaluationResult], List[EvaluationError]]:
        return await self.evaluation.run(**kwargs)


class EvaluationExecutor:
    def __init__(self, evaluations: Dict[str, EvaluationInit]):
        self.evaluations = evaluations
        self.iterations: Dict[str, List[EvaluationRun]] = {
            evaluation_id: [] for evaluation_id in evaluations
        }

    def register_linked_evaluation(
        self, new_evaluation_id: str, existing_evaluation_id: str
    ):
        if existing_evaluation_id not in self.evaluations:
            raise ValueError(f"Evaluation with id {existing_evaluation_id} not found.")

        if new_evaluation_id in self.evaluations:
            raise ValueError(f"Evaluation with id {new_evaluation_id} already exists.")

        linked_evaluation = LinkedEvaluation(
            evaluation=self.evaluations[existing_evaluation_id]["evaluation"]
        )
        self.evaluations[new_evaluation_id] = {
            "evaluation": linked_evaluation,
            "evaluation_schema": self.evaluations[existing_evaluation_id][
                "evaluation_schema"
            ],
        }
        self.iterations[new_evaluation_id] = []

    async def execute_all(
        self, input: Dict[str, Dict[str, Any]]
    ) -> List[EvaluationRun]:
        results = []
        tasks = [
            self.execute_specific(evaluation_id, input=input[evaluation_id])
            for evaluation_id in self.evaluations.keys()
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def execute_specific(
        self, evaluation_id: str, input: Dict[str, Any]
    ) -> EvaluationRun:
        if evaluation_id not in self.evaluations:
            raise ValueError(f"Evaluation with id {evaluation_id} not found.")

        evaluation_init = self.evaluations[evaluation_id]
        evaluation = evaluation_init["evaluation"]
        evaluation_schema = evaluation_init["evaluation_schema"]

        validated_params = evaluation_schema.model_validate(input)

        started_at = datetime.datetime.now()
        iteration = (
            self.iterations[evaluation_id][-1].iteration + 1
            if self.iterations[evaluation_id]
            else 1
        )

        try:
            run_tuple = await evaluation.run(**validated_params.model_dump())
        except Exception as e:
            run_tuple = (
                [],
                [
                    EvaluationError(
                        message=str(e),
                        type=type(e).__name__,
                        details={"exception_type": type(e).__name__},
                    )
                ],
            )
        execution_time_ms = int(
            (datetime.datetime.now() - started_at).total_seconds() * 1000
        )

        run = EvaluationRun(
            started_at=started_at,
            execution_time_ms=execution_time_ms,
            iteration=iteration,
            results=run_tuple[0],
            errors=run_tuple[1],
        )
        self.iterations[evaluation_id].append(run)
        return run

    def get_latest_results(self) -> List[EvaluationRun]:
        return [runs[-1] for runs in self.iterations.values() if runs]
