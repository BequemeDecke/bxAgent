from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from bxagent import models
from bxagent.config import Config

from .middleware import CodingDeepAgentState, CodingDeepAgentStateMiddleware


def build_coding_deep_agent(workspace_path: Path):
    config = Config.get_instance()
    model = models.build_coding_model()
    backend = FilesystemBackend(root_dir=workspace_path, virtual_mode=True)

    agent = create_deep_agent(
        model=model,
        state_schema=CodingDeepAgentState,
        middleware=[
            CodingDeepAgentStateMiddleware(
                updated_file_index=config.VARIABLES.UPDATED_FILE_INDEX,
                workspace_path=workspace_path,
                file_extension_filter=".java",  # Only track Java files in the coding agent state
            )
        ],
        backend=backend,
    )

    return agent
