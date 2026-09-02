from unittest import TestCase
from unittest.mock import Mock, patch

from langgraph.graph import StateGraph

from mdeagent.implementation.agent import build_implementation_graph


class TestBuildImplementationGraph(TestCase):

    @patch("mdeagent.implementation.agent.create_implement_transformation_node")
    @patch("mdeagent.implementation.agent.create_implement_bx_tool_node")
    @patch("mdeagent.implementation.agent.create_evaluation_node")
    @patch("mdeagent.implementation.agent.create_evaluate_transformation_implementation")
    @patch("mdeagent.implementation.agent.build_base_model")
    def test_build_implementation_graph(
        self,
        mock_build_base_model: Mock,
        mock_evaluate_transformation_implementation: Mock,
        mock_create_evaluation_agent_work_function: Mock,
        mock_create_implement_bx_tool_node: Mock,
        mock_create_implement_transformation_node: Mock,
    ):
        # Mocks
        mock_build_base_model.return_value = Mock(name="base_model")
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
        graph = build_implementation_graph(
            evaluation_executor=Mock(name="evaluation_executor"),
            workspace_path=Mock(name="workspace_path"),
        )

        # Assertions
        self.assertIsInstance(graph, StateGraph)
        mock_evaluate_transformation_implementation.assert_called_once()
        mock_create_evaluation_agent_work_function.assert_called_once()
        mock_create_implement_bx_tool_node.assert_called_once()
        mock_create_implement_transformation_node.assert_called_once()

    @patch("mdeagent.implementation.agent.create_implement_transformation_node")
    @patch("mdeagent.implementation.agent.create_implement_bx_tool_node")
    @patch("mdeagent.implementation.agent.create_evaluation_node")
    @patch("mdeagent.implementation.agent.create_evaluate_transformation_implementation")
    @patch("mdeagent.implementation.agent.build_base_model")
    def test_build_implementation_graph__registers_linked_evaluation(
        self,
        mock_build_base_model: Mock,
        mock_evaluate_transformation_implementation: Mock,
        mock_create_evaluation_agent_work_function: Mock,
        mock_create_implement_bx_tool_node: Mock,
        mock_create_implement_transformation_node: Mock,
    ):
        # Mocks
        mock_build_base_model.return_value = Mock(name="base_model")
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
        mock_evaluation_executor = Mock(name="evaluation_executor")

        # Function under test
        build_implementation_graph(
            evaluation_executor=mock_evaluation_executor,
            workspace_path=Mock(name="workspace_path"),
        )

        # Assertions
        mock_evaluation_executor.register_linked_evaluation.assert_called_once_with(
            "integration_compilation", "java_compilation"
        )