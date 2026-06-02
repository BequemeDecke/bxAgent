from langchain.tools import tool
from pathlib import Path

from bxagent.config import Config

TRANSFORMATION_FILE_PATH = Config.get_instance().WORKSPACE.PATH / "TRANSFORMATION.md"

def ensure_transformation_file_exists() -> None:
    """Ensures that the transformation file exists. If it does not exist, it creates an empty file."""
    if not TRANSFORMATION_FILE_PATH.exists():
        TRANSFORMATION_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRANSFORMATION_FILE_PATH.touch()


def _read_transformation_plan() -> str:
    """Reads the current transformation plan from the the `TRANSFORMATION.md` file."""
    ensure_transformation_file_exists()
    # Return a clear placeholder if the file is empty so tools/agents
    # receive a non-empty response and can continue processing.
    with open(TRANSFORMATION_FILE_PATH, "r") as f:
        content = f.read()
        if content is None or content.strip() == "":
            return "<TRANSFORMATION.md is empty>"
        return content


@tool("read_transformation_plan")
def read_transformation_plan() -> str:
    """Reads the current transformation plan from the the `TRANSFORMATION.md` file."""
    return _read_transformation_plan()


def _update_transformation_plan(new_plan: str) -> bool:
    """Updates the transformation plan in the `TRANSFORMATION.md` file with the new plan."""
    ensure_transformation_file_exists() 
    with open(TRANSFORMATION_FILE_PATH, "w") as f:
        f.write(new_plan)
    return True


@tool("update_transformation_plan")
def update_transformation_plan(new_plan: str) -> bool:
    """Updates the transformation plan in the `TRANSFORMATION.md` file with the new plan."""
    return _update_transformation_plan(new_plan)

transformation_plan_tools = (read_transformation_plan, update_transformation_plan)