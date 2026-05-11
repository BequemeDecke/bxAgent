import unittest

from langchain.messages import HumanMessage

from src.agents.synthesis import build_synthesis_agent

class TestSynthesisAgent(unittest.TestCase):
    """Test the SynthesisAgent by invoking it with a simple input and checking the output and state."""
    def setUp(self):
        self.agent = build_synthesis_agent()
        self.config = {"configurable": {"thread_id": "test_thread_id"}}
    
    def test_synthesis_agent(self):
        """Test the SynthesisAgent by invoking it with a simple input and checking the output and state."""
        # Invoke the agent with a simple input
        response = self.agent.invoke(
            [
                HumanMessage(content="How would you transform a source model into a target model given the following requirements: ...")
            ],
            config=self.config,
        )

        # Check that the response is not empty
        self.assertIsNotNone(response)
        self.assertNotEqual(response.content, "")

        # Check that the transformation plan has been updated in the state
        transformation_plan = self.agent.state.get("transformation_plan")
        self.assertIsNotNone(transformation_plan)