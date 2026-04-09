import os
import logging
import sys
import uuid

from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, Field
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langfuse import Langfuse, get_client as get_langfuse_client
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend, CompositeBackend, FilesystemBackend

from prompt import SYSTEM_PROMPT

# Load environment variables from the .env file
dotenv_path = (
    Path.cwd() / ".env"
)  # Env file is located one level up from the src directory
has_env_loaded = load_dotenv(dotenv_path=dotenv_path)
assert has_env_loaded, f"Failed to load environment variables from {dotenv_path}"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,  # TODO: Let this be configurable; Counter: 2
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# Save the loaded environment variables to a config class for easy access
class BxAgentConfig(BaseModel):
    """Configuration class for BxAgent."""

    API_KEY: SecretStr = Field()
    BASE_URL: str = Field()
    MODEL_ID: str = Field()


agent_config = BxAgentConfig(
    API_KEY=os.getenv("API_KEY"),
    BASE_URL=os.getenv("BASE_URL"),
    MODEL_ID=os.getenv("MODEL_ID"),
)

class LangFuseConfig(BaseModel):
    SECRET_KEY: SecretStr = Field()
    PUBLIC_KEY: SecretStr = Field()
    BASE_URL: str = Field()
    
langfuse_config = LangFuseConfig(
    SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY"),
    PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY"),
    BASE_URL=os.getenv("LANGFUSE_BASE_URL"),
)


# --- Builder Functions ---
def build_langfuse_client():
    # Initialize Langfuse client with constructor arguments
    client = Langfuse(
        secret_key=langfuse_config.SECRET_KEY.get_secret_value(),
        public_key=langfuse_config.PUBLIC_KEY.get_secret_value(),
        host=langfuse_config.BASE_URL
    )

    # Initialize the Langfuse handler
    langfuse_handler = LangfuseCallbackHandler()
    return client, langfuse_handler

def build_chat_model():
    """Builds the chat model using the loaded configuration."""
    return init_chat_model(
        model_provider="openai",  # TODO: Make this configurable later on; Counter: 1
        base_url=agent_config.BASE_URL,
        api_key=agent_config.API_KEY.get_secret_value(),
        model=agent_config.MODEL_ID,
    )


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
        skills=["/skills/"]
    )


# --- Main Execution ---
def main():
    if len(sys.argv) < 3:
        logging.error(
            "No custom workspace directory or prompt provided. Using default values. To specify custom values, run the script with: python main.py <workspace_dir> <prompt>"
        )
        exit(1)

    workspace_dir = Path(sys.argv[1])
    input_prompt = sys.argv[2]

    logging.debug(f"Using workspace directory: {workspace_dir}")
    logging.info("Starting BxAgent with configuration: %s", agent_config)

    bx_agent = build_bx_agent(workspace_dir=workspace_dir)
    logging.debug(f"BxAgent initialized successfully.")
    
    langfuse_client, langfuse_handler = build_langfuse_client()
    logging.debug("Langfuse client and handler initialized successfully.")

    response = bx_agent.invoke(
        {"messages": [HumanMessage(content=input_prompt)]},
        {
            "configurable": {
                "thread_id": str(
                    uuid.uuid4()
                ),  # Maybe there are better ways to do that
            },
            "callbacks": [langfuse_handler],
        },
    )
    logging.info(f"Received response from bxAgent: {response}")
    langfuse_client.flush()  # Ensure all events are sent to Langfuse


if __name__ == "__main__":
    main()
