from langchain.tools import tool
from pathlib import Path

TRANSFORMATION_FILE_PATH = Path.cwd() / ".bx-agent-workspace" / "transformation.md"


def _read_transformation_plan() -> str:
    """Reads the current transformation plan from the the `transformation.md` file."""
    with open(TRANSFORMATION_FILE_PATH, "r") as f:
        return f.read()


@tool("read_transformation_plan", return_direct=True)
def read_transformation_plan() -> str:
    """Reads the current transformation plan from the the `transformation.md` file."""
    return _read_transformation_plan()


def _update_transformation_plan(new_plan: str) -> None:
    """Updates the transformation plan in the `transformation.md` file with the new plan."""
    with open(TRANSFORMATION_FILE_PATH, "w") as f:
        f.write(new_plan)


@tool("update_transformation_plan", return_direct=True)
def update_transformation_plan(new_plan: str) -> None:
    """Updates the transformation plan in the `transformation.md` file with the new plan."""
    _update_transformation_plan(new_plan)

transformation_plan_tools = [read_transformation_plan, update_transformation_plan]