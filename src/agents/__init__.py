from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend, CompositeBackend, FilesystemBackend
from pathlib import Path

from src.prompt import SYSTEM_PROMPT
from src.config import langfuse_config
from .models import build_chat_model


# --- Builder Functions ---
def build_langfuse_client():
    # Initialize Langfuse client with constructor arguments
    client = Langfuse(
        secret_key=langfuse_config.SECRET_KEY.get_secret_value(),
        public_key=langfuse_config.PUBLIC_KEY.get_secret_value(),
        host=langfuse_config.BASE_URL,
    )

    # Initialize the Langfuse handler
    langfuse_handler = LangfuseCallbackHandler()
    return client, langfuse_handler


def build_backend(workspace_dir: Path):
    """
    Builds the backend for the BxAgent.
    """

    bxagent_skills_dir = Path.cwd() / "bxagent-skills" / "skills"

    return lambda rt: CompositeBackend(
        default=LocalShellBackend(root_dir=workspace_dir, virtual_mode=True),
        routes={
            "/skills/": FilesystemBackend(
                root_dir=bxagent_skills_dir, virtual_mode=True
            )
        },
    )


def build_bx_agent(
    workspace_dir: Path = Path("agent_data"), system_prompt: str = SYSTEM_PROMPT
):
    """Builds the BxAgent using the chat model."""
    model = build_chat_model()
    backend = build_backend(workspace_dir)

    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=SystemMessage(system_prompt),
        checkpointer=InMemorySaver(),
        skills=["/skills/"],
    )
