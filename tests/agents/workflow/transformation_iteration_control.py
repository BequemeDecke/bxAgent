"""
This test checks if the transformation iteration control node correctly limits the number of iterations the agent performs when trying to implement a model transformation.
"""

from unittest import TestCase

from bxagent.agents.workflow.state import WorkflowState
from bxagent.agents.workflow.transformation_iteration_control import check_transformation_iteration


class TestTransformationIterationControl(TestCase):
    def test_transformation_iteration_control__stop_on_max_iterations(self):
        max_iterations = 3
        state: WorkflowState = {
            "transformation_source_model_description": "A model that needs to be transformed.",
            "transformation_target_model_description": "The desired model after transformation.",
            "iteration": 3,
            "latest_audit_runs": []
        }
        result = check_transformation_iteration(state, max_iterations)
        self.assertEqual(result, "stop")
        
    def test_transformation_iteration_control__continue_before_max_iterations(self):
        max_iterations = 3
        state: WorkflowState = {
            "transformation_source_model_description": "A model that needs to be transformed.",
            "transformation_target_model_description": "The desired model after transformation.",
            "iteration": 2,
            "latest_audit_runs": []
        }
        result = check_transformation_iteration(state, max_iterations)
        self.assertEqual(result, "continue")
