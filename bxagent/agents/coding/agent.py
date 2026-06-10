from deepagents import create_deep_agent

from bxagent.config import Config
from bxagent import models
from .middleware import CodingDeepAgentState, CodingDeepAgentStateMiddleware


def build_coding_deep_agent():
    config = Config.get_instance()
    model = models.build_coding_model()

    agent = create_deep_agent(
        model=model,
        middleware=[
            CodingDeepAgentStateMiddleware(
                updated_file_index=config.VARIABLES.UPDATED_FILE_INDEX,
                workspace_path=config.WORKSPACE.PATH,
                file_extension_filter=".java",  # Only track Java files in the coding agent state
            )
        ],
    )

    return agent
