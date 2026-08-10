import asyncio
import datetime

from typing import List, Dict, Tuple, TypedDict, Any
from pydantic import BaseModel

from .types import Validation, ValidationResult, ValidationRun, ValidationError


class ValidationInit(TypedDict):
    validation: Validation
    validation_schema: BaseModel


class LinkedValidation(Validation):
    def __init__(self, validation: Validation):
        self.validation = validation

    async def setup(self):
        await self.validation.setup()

    async def run(
        self, **kwargs
    ) -> Tuple[List[ValidationResult], List[ValidationError]]:
        return await self.validation.run(**kwargs)


class ValidationExecutor:
    def __init__(self, validations: Dict[str, ValidationInit]):
        self.validations = validations
        self.iterations: Dict[str, List[ValidationRun]] = {
            validation_id: [] for validation_id in validations
        }

    def register_linked_validation(
        self, new_validation_id: str, existing_validation_id: str
    ):
        if existing_validation_id not in self.validations:
            raise ValueError(f"Validation with id {existing_validation_id} not found.")

        if new_validation_id in self.validations:
            raise ValueError(f"Validation with id {new_validation_id} already exists.")

        linked_validation = LinkedValidation(
            validation=self.validations[existing_validation_id]["validation"]
        )
        self.validations[new_validation_id] = {
            "validation": linked_validation,
            "validation_schema": self.validations[existing_validation_id][
                "validation_schema"
            ],
        }
        self.iterations[new_validation_id] = []

    async def execute_all(
        self, input: Dict[str, Dict[str, Any]]
    ) -> List[ValidationRun]:
        results = []
        tasks = [
            self.execute_specific(validation_id, input=input[validation_id])
            for validation_id in self.validations.keys()
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def execute_specific(
        self, validation_id: str, input: Dict[str, Any]
    ) -> ValidationRun:
        if validation_id not in self.validations:
            raise ValueError(f"Validation with id {validation_id} not found.")

        validation_init = self.validations[validation_id]
        validation = validation_init["validation"]
        validation_schema = validation_init["validation_schema"]

        validated_params = validation_schema.model_validate(input)

        started_at = datetime.datetime.now()
        iteration = (
            self.iterations[validation_id][-1].iteration + 1
            if self.iterations[validation_id]
            else 1
        )

        try:
            run_tuple = await validation.run(**validated_params.model_dump())
        except Exception as e:
            run_tuple = (
                [],
                [
                    ValidationError(
                        message=str(e),
                        type=type(e).__name__,
                        details={"exception_type": type(e).__name__},
                    )
                ],
            )
        execution_time_ms = int(
            (datetime.datetime.now() - started_at).total_seconds() * 1000
        )

        run = ValidationRun(
            started_at=started_at,
            execution_time_ms=execution_time_ms,
            iteration=iteration,
            results=run_tuple[0],
            errors=run_tuple[1],
        )
        self.iterations[validation_id].append(run)
        return run

    def get_latest_results(self) -> List[ValidationRun]:
        return [runs[-1] for runs in self.iterations.values() if runs]
