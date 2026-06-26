"""
This test checks if the transformation iteration control node correctly limits the number of iterations the agent performs when trying to implement a model transformation.
"""

from datetime import datetime, timedelta
from unittest import TestCase
from unittest.mock import Mock

from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage, HumanMessage

from bxagent.agents.workflow.state import WorkflowState
from bxagent.agents.workflow.transformation_iteration_control import (
    IterationRoute,
    create_check_transformation_iteration_function,
)
from bxagent.validation.types import ValidationError, ValidationResult, ValidationRun


class TestTransformationIterationControl(TestCase):
    def setUp(self):
        mocked_llm = Mock(spec=BaseChatModel)
        mocked_llm_structured_output = Mock(spec=BaseChatModel)
        self.mocked_llm = mocked_llm_structured_output
        mocked_llm.with_structured_output.return_value = mocked_llm_structured_output
        mocked_llm_structured_output.invoke.return_value = IterationRoute(
            decision="stop"
        )
        self.check_transformation_iteration = (
            create_check_transformation_iteration_function(llm=mocked_llm)
        )

    def test_transformation_iteration_control__stop_on_max_iterations(self):
        max_iterations = 3
        state: WorkflowState = {
            "transformation_source_model_description": "A model that needs to be transformed.",
            "transformation_target_model_description": "The desired model after transformation.",
            "iteration": 3,
            "latest_validation_runs": [],
        }
        result = self.check_transformation_iteration(state, max_iterations)
        self.assertEqual(result, "stop")

    def test_transformation_iteration_control__run_results_have_errors(self):
        max_iterations = 3
        state: WorkflowState = {
            "transformation_source_model_description": "A model that needs to be transformed.",
            "transformation_target_model_description": "The desired model after transformation.",
            "iteration": 2,
            "latest_validation_runs": [
                ValidationRun(
                    started_at=datetime.now() - timedelta(minutes=5),
                    execution_time_ms=200,
                    iteration=1,
                    results=[
                        ValidationResult(
                            content="Validation result content",
                        )
                    ],
                    errors=[
                        ValidationError(
                            message="An error occurred during the validation.",
                            type="ValidationError",
                            details={
                                "error_code": "AUDIT_ERROR",
                                "severity": "high",
                            },
                        )
                    ],
                )
            ],
        }
        result = self.check_transformation_iteration(state, max_iterations)
        self.assertEqual(result, "error")

    def test_transformation_iteration_control__run_results_no_errors(self):
        max_iterations = 3
        state: WorkflowState = {
            "transformation_source_model_description": "A model that needs to be transformed.",
            "transformation_target_model_description": "The desired model after transformation.",
            "iteration": 2,
            "latest_validation_runs": [
                ValidationRun(
                    started_at=datetime.now() - timedelta(minutes=5),
                    execution_time_ms=200,
                    iteration=1,
                    results=[
                        ValidationResult(
                            content="Validation result content",
                            metadata={"include_in_report": True, "success": True},
                        ),
                        ValidationResult(
                            content="Another validation result content",
                            metadata={"include_in_report": False, "success": True},
                        ),
                        ValidationResult(
                            content="Validation result with error",
                            metadata={"include_in_report": True, "success": False},
                        ),
                    ],
                    errors=[],
                )
            ],
        }
        result = self.check_transformation_iteration(state, max_iterations)
        self.assertEqual(result, "stop")

        called_args = self.mocked_llm.invoke.call_args[0][0]
        self.assertIsInstance(called_args[0], SystemMessage)

        human_message = called_args[1]
        self.assertIsInstance(human_message, HumanMessage)
        self.assertNotIn("Validation result content", human_message.content)
        self.assertNotIn("Another validation result content", human_message.content)
        self.assertIn("Validation result with error", human_message.content)
