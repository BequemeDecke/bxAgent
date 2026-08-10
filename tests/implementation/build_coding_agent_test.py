from unittest import TestCase
from unittest.mock import patch, Mock

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from bxagent.implementation.agent import build_coding_agent_subgraph


class TestBuildCodingAgent(TestCase):

    @patch("bxagent.tools.coding.agent.create_implement_transformation_node")
    @patch("bxagent.tools.coding.agent.create_implement_bx_tool_node")
    @patch("bxagent.tools.coding.agent.create_evaluation_node")
    @patch("bxagent.tools.coding.agent.create_evaluate_transformation_implementation")
    def test_build_coding_agent_subgraph(
        self,
        mock_evaluate_transformation_implementation: Mock,
        mock_create_evaluation_agent_work_function: Mock,
        mock_create_implement_bx_tool_node: Mock,
        mock_create_implement_transformation_node: Mock,
    ):
        # Mocks
        mock_create_implement_transformation_node.return_value = Mock(
            name="implement_transformation_node"
        )
        mock_create_implement_bx_tool_node.return_value = Mock(
            name="implement_bx_tool_node"
        )
        mock_create_evaluation_agent_work_function.return_value = Mock(
            name="evaluation_agent_work_function"
        )
        mock_evaluate_transformation_implementation.return_value = Mock(
            name="evaluate_transformation_implementation"
        )

        # Function under test
        graph = build_coding_agent_subgraph(
            evaluation_executor=Mock(name="evaluation_executor"),
            coding_deep_agent=Mock(spec=CompiledStateGraph),
        )

        # Assertions
        self.assertIsInstance(graph, StateGraph)
        mock_evaluate_transformation_implementation.assert_called_once()
        mock_create_evaluation_agent_work_function.assert_called_once()
        mock_create_implement_bx_tool_node.assert_called_once()
        mock_create_implement_transformation_node.assert_called_once()
