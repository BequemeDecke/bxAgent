from unittest import TestCase

from mdagent.implementation.state import ImplementationState # Fix cyclic import from __init__ files
from mdagent.agents.workflow.state import WorkflowState
from mdagent.mapping import map_workflow_to_commands

class TestWorkflowToCommandsMapping(TestCase):
    def test_mapping(self):
        state = WorkflowState(required_commands=["command1", "command2"])
        evaluation_params = map_workflow_to_commands(state)
        self.assertIn("commands", evaluation_params)
        self.assertEqual(evaluation_params["commands"], ["command1", "command2"])
