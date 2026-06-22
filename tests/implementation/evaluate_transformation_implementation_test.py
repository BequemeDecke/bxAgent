from unittest import TestCase

from bxagent.implementation.evaluate_transformation_implementation import (
    create_evaluate_transformation_implementation,
)
from bxagent.validation.types import ValidationRun, ValidationError
from bxagent.implementation.state import ImplementationState


class TestEvaluateTransformationImplementation(TestCase):

    def test_evaluate_transformation_implementation__max_iteration_reached(self):
        """
        Test that the evaluation correctly identifies when the maximum number of iterations has been reached.
        """
        # Prepare function under test
        self.evaluate_transformation_implementation = (
            create_evaluate_transformation_implementation()
        )
        MAX_ITERATIONS = 5
        agent_state = ImplementationState(
            transformation_md=None,
            transformation_source_model_description="Source model description",
            transformation_target_model_description="Target model description",
            implementation_iteration=MAX_ITERATIONS,
            bxtool_file=None,
            latest_validation_results={},
        )

        # Call the function under test
        evaluation = self.evaluate_transformation_implementation(
            agent_state=agent_state, max_iterations=MAX_ITERATIONS
        )

        # Assertions
        self.assertEqual(
            evaluation,
            "max_iteration_reached",
            "The evaluation should indicate that the maximum number of iterations has been reached.",
        )

    def test_evaluate_transformation_implementation__implementation_error(self):
        """
        Test that the evaluation correctly identifies when there is an implementation error based on the latest validation results.
        """
        # Prepare function under test
        self.evaluate_transformation_implementation = (
            create_evaluate_transformation_implementation()
        )
        agent_state = ImplementationState(
            transformation_md=None,
            transformation_source_model_description="Source model description",
            transformation_target_model_description="Target model description",
            implementation_iteration=1,
            bxtool_file=None,
            latest_validation_results={
                "validation1": ValidationRun(
                    started_at=None,
                    execution_time_ms=100,
                    iteration=1,
                    results=[],
                    errors=[ValidationError(message="Error in implementation")],
                ),
                "integration_compilation": ValidationRun(
                    started_at=None,
                    execution_time_ms=100,
                    iteration=1,
                    results=[],
                    errors=[],
                ),
            },
        )

        # Call the function under test
        evaluation = self.evaluate_transformation_implementation(
            agent_state=agent_state, max_iterations=5
        )

        # Assertions
        self.assertEqual(
            evaluation,
            "implementation_error",
            "The evaluation should indicate that there is an implementation error based on the latest validation results.",
        )

    def test_evaluate_transformation_implementation__integration_error(self):
        """
        Test that the evaluation correctly identifies when there is an integration error based on the latest validation results.
        """
        # Prepare function under test
        self.evaluate_transformation_implementation = (
            create_evaluate_transformation_implementation()
        )
        agent_state = ImplementationState(
            transformation_md=None,
            transformation_source_model_description="Source model description",
            transformation_target_model_description="Target model description",
            implementation_iteration=1,
            bxtool_file=None,
            latest_validation_results={
                "integration_compilation": ValidationRun(
                    started_at=None,
                    execution_time_ms=100,
                    iteration=1,
                    results=[],
                    errors=[ValidationError(message="Error in integration")],
                )
            },
        )

        # Call the function under test
        evaluation = self.evaluate_transformation_implementation(
            agent_state=agent_state, max_iterations=5
        )

        # Assertions
        self.assertEqual(
            evaluation,
            "integration_error",
            "The evaluation should indicate that there is an integration error based on the latest validation results.",
        )

    def test_evaluate_transformation_implementation__implementation_success(self):
        """
        Test that the evaluation correctly identifies when the implementation is successful based on the latest validation results.
        """
        # Prepare function under test
        self.evaluate_transformation_implementation = (
            create_evaluate_transformation_implementation()
        )

        agent_state = ImplementationState(
            transformation_md=None,
            transformation_source_model_description="Source model description",
            transformation_target_model_description="Target model description",
            implementation_iteration=1,
            bxtool_file=None,
            latest_validation_results={
                "validation1": ValidationRun(
                    started_at=None,
                    execution_time_ms=100,
                    iteration=1,
                    results=[],
                    errors=[],
                ),
                "integration_compilation": ValidationRun(
                    started_at=None,
                    execution_time_ms=100,
                    iteration=1,
                    results=[],
                    errors=[],
                ),
            },
        )

        # Call the function under test
        evaluation = self.evaluate_transformation_implementation(
            agent_state=agent_state, max_iterations=5
        )

        # Assertions
        self.assertEqual(
            evaluation,
            "implementation_success",
            "The evaluation should indicate that the implementation is successful based on the latest validation results.",
        )
