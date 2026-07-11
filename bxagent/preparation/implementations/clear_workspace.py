import shutil
import os

from ..prepare_workspace import StructureFixStrategy
from ..state import PreparationState

class ClearWorkspaceStrategy(StructureFixStrategy):
    """
    A strategy to clear the workspace by removing all files and directories.
    """

    def fix_structure(self, state: PreparationState) -> PreparationState:
        workspace_path = state.get("workspace_path") # This is a safe operation        

        # Clear the workspace by removing all files and directories
        for root, dirs, files in os.walk(workspace_path):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path)
            
        return {}