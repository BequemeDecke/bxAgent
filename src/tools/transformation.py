from langchain.agents import AgentState as BaseAgentState
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage


class TransformationPlan:
    decisions: list[str] = (
        []
    )  # The decisions that can be made in the transformation process
    reasoning: list[str] = []  # The reasioning behind the options the agent has made
    steps: list[str] = (
        []
    )  # The transformation plan which is a list of steps to transform the source model into the target model

    def add_decision(self, decision: str, reasoning: str):
        self.decisions.append(decision)
        self.reasoning.append(reasoning)

    def update_existing_decision(self, index: int, decision: str, reasoning: str):
        self.decisions[index] = decision
        self.reasoning.append(reasoning)

    def change_decisions(self, decisions: list[str], reasoning: str):
        self.decisions = decisions
        self.reasoning.append(reasoning)

    def add_step(self, step: str, reasoning: str):
        self.steps.append(step)
        self.reasoning.append(reasoning)

    def update_existing_step(self, index: int, step: str, reasoning: str):
        self.steps[index] = step
        self.reasoning.append(reasoning)

    def change_steps(self, steps: list[str], reasoning: str):
        self.steps = steps
        self.reasoning.append(reasoning)

    def __str__(self) -> str:
        """Stringifies the transformation plan to a markdown format."""
        return (
            "# Transformation Plan\n\n"
            "## Decisions within Transformation\n\n"
            + "\n".join(
                f"{i+1}. {decision}" for i, decision in enumerate(self.decisions)
            )
            + "## Reasoning behind decisions\n\n"
            + "\n".join(f"- {r}" for r in self.reasoning)
            + "## Steps for implementation of the transformation\n\n"
            + "\n".join(f"{i+1}. {step}" for i, step in enumerate(self.steps))
        )

    @classmethod
    def from_markdown(cls, markdown: str) -> "TransformationPlan":
        """Parses a markdown string to create a TransformationPlan object."""
        # This is a very simple parser and assumes that the markdown format is correct.
        # In a real implementation, you would want to add error handling and more robust parsing.
        decisions = []
        reasoning = []
        steps = []

        lines = markdown.splitlines()
        current_section = None

        for line in lines:
            if line.startswith("# "):
                continue  # Skip the main title
            elif line.startswith("## "):
                if "Decisions" in line:
                    current_section = "decisions"
                elif "Reasoning" in line:
                    current_section = "reasoning"
                elif "Steps" in line:
                    current_section = "steps"
            else:
                if current_section == "decisions" and line.startswith("- "):
                    decisions.append(line[2:])
                elif current_section == "reasoning" and line.startswith("- "):
                    reasoning.append(line[2:])
                elif current_section == "steps" and line.startswith("- "):
                    steps.append(line[2:])

        return cls(decisions=decisions, reasoning=reasoning, steps=steps)


class TransformationPlanState(BaseAgentState):
    """
    Agent state that holds the transformation plan. This is crucial for the synthesis agent to keep track of the decisions, reasoning and steps that have been made in the transformation process.
    """

    transformation_plan: TransformationPlan = TransformationPlan()


@tool
def read_transformation_plan(runtime: ToolRuntime) -> str:
    """Reads the transformation plan and adds it to the agent's state."""
    # In a real implementation, you would read from a file or a database. Here we just return an empty string for simplicity.
    return ""


@tool
def add_decision_to_transformation_plan(
    runtime: ToolRuntime, decision: str, reasoning: str
) -> None:
    """Adds a decision to the transformation plan."""
    # Implementation for adding a decision to the transformation plan
    transformation_plan = runtime.state.get("transformation_plan")
    if transformation_plan is None:
        raise AssertionError(
            "Transformation plan not found in state. Use the TransformationPlanState for the agent's state!"
        )

    transformation_plan: TransformationPlan = transformation_plan
    transformation_plan.add_decision(decision, reasoning)
    runtime.state["transformation_plan"] = transformation_plan
    return Command(
        update={
            "transformation_plan": transformation_plan,
            "messages": [
                ToolMessage(
                    content=str(transformation_plan), tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


@tool
def add_step_to_transformation_plan(
    runtime: ToolRuntime, step: str, reasoning: str
) -> None:
    """Adds a step to the transformation plan."""
    # Implementation for adding a step to the transformation plan
    transformation_plan = runtime.state.get("transformation_plan")
    if transformation_plan is None:
        raise AssertionError(
            "Transformation plan not found in state. Use the TransformationPlanState for the agent's state!"
        )

    transformation_plan: TransformationPlan = transformation_plan
    transformation_plan.add_step(step, reasoning)
    runtime.state["transformation_plan"] = transformation_plan
    return Command(
        update={
            "transformation_plan": transformation_plan,
            "messages": [
                ToolMessage(
                    content=str(transformation_plan), tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


@tool
def update_decision_in_transformation_plan(
    runtime: ToolRuntime, index: int, decision: str, reasoning: str
) -> None:
    """Updates an existing decision in the transformation plan."""
    # Implementation for updating a decision in the transformation plan
    transformation_plan = runtime.state.get("transformation_plan")
    if transformation_plan is None:
        raise AssertionError(
            "Transformation plan not found in state. Use the TransformationPlanState for the agent's state!"
        )

    transformation_plan: TransformationPlan = transformation_plan
    transformation_plan.update_existing_decision(index, decision, reasoning)
    runtime.state["transformation_plan"] = transformation_plan
    return Command(
        update={
            "transformation_plan": transformation_plan,
            "messages": [
                ToolMessage(
                    content=str(transformation_plan), tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


@tool
def update_step_in_transformation_plan(
    runtime: ToolRuntime, index: int, step: str, reasoning: str
) -> None:
    """Updates an existing step in the transformation plan."""
    # Implementation for updating a step in the transformation plan
    transformation_plan = runtime.state.get("transformation_plan")
    if transformation_plan is None:
        raise AssertionError(
            "Transformation plan not found in state. Use the TransformationPlanState for the agent's state!"
        )

    transformation_plan: TransformationPlan = transformation_plan
    transformation_plan.update_existing_step(index, step, reasoning)
    runtime.state["transformation_plan"] = transformation_plan
    return Command(
        update={
            "transformation_plan": transformation_plan,
            "messages": [
                ToolMessage(
                    content=str(transformation_plan), tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )

transformation_plan_tools = [
    read_transformation_plan,
    add_decision_to_transformation_plan,
    add_step_to_transformation_plan,
    update_decision_in_transformation_plan,
    update_step_in_transformation_plan,
]