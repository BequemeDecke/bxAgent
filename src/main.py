import os
import logging

from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, Field
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

# Load environment variables from the .env file
dotenv_path = (
    Path.cwd() / ".env"
)  # Env file is located one level up from the src directory
has_env_loaded = load_dotenv(dotenv_path=dotenv_path)
assert has_env_loaded, f"Failed to load environment variables from {dotenv_path}"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
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


# --- Builder Functions ---
def build_chat_model():
    """Builds the chat model using the loaded configuration."""
    return init_chat_model(
        model_provider="openai",  # TODO: Make this configurable later on; Counter: 1
        base_url=agent_config.BASE_URL,
        api_key=agent_config.API_KEY.get_secret_value(),
        model=agent_config.MODEL_ID,
    )


def build_backend():
    """
    Builds the backend for the BxAgent.

    When using the LocalShellBackend, we set virtual_mode to True to enable in-memory operations.
    """
    return LocalShellBackend(
        root_dir=Path("./agent_data"), virtual_mode=True
    )  # TODO: Make this configurable later on; Counter: 1


def build_bx_agent(system_prompt: str):
    """Builds the BxAgent using the chat model."""
    model = build_chat_model()
    backend = build_backend()

    return create_deep_agent(
        model=model, backend=backend, system_prompt=SystemMessage(system_prompt)
    )


# --- Main Execution ---
def main():
    logging.info("Starting BxAgent with configuration: %s", agent_config)

    bx_agent = build_bx_agent(
        system_prompt="You are a helpful assistant with coding capabilities in Python."
    )
    logging.debug(f"BxAgent initialized successfully.")

    # TODO: Remove this example after testing
    response = bx_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write a Python function that adds two numbers and returns the result. The file should be named add.py and the function should be named add_numbers."
                )
            ]
        },
    )
    logging.info(f"Received response from bxAgent: {response}")


if __name__ == "__main__":
    main()
