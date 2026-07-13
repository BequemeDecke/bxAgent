import logging
import xml.etree.ElementTree as ET
from pathlib import Path

# Source - https://stackoverflow.com/a/54491129
# Posted by rez, modified by community. See post 'Timeline' for change history
# Retrieved 2026-07-13, License - CC BY-SA 4.0


def get_all_namespaces(filename: Path):
    namespaces = dict([node for _, node in ET.iterparse(filename, events=["start-ns"])])
    return namespaces


def log_workspace_structure(workspace: Path):
    """
    Log the structure of the workspace directory for debugging purposes.
    """
    logging.debug(f"Workspace structure at {workspace}:")
    for path in workspace.rglob("*"):
        logging.debug(f" - {path.relative_to(workspace)}")


def copy_workspace(workspace: Path, destination: Path):
    """
    Copy the entire workspace directory to a new destination.
    """
    import shutil

    if destination.exists():
        logging.warning(
            f"Destination {destination} already exists. It will be overwritten."
        )
        shutil.rmtree(destination)

    shutil.copytree(workspace, destination)
    logging.info(f"Workspace copied from {workspace} to {destination}.")
