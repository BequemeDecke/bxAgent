import os
import logging

from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, Field
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from deepagents import create_deep_agent

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


def build_chat_model():
    """Builds the chat model using the loaded configuration."""
    return init_chat_model(
        model_provider="openai",  # TODO: Make this configurable later on; Counter: 1
        base_url=agent_config.BASE_URL,
        api_key=agent_config.API_KEY.get_secret_value(),
        model=agent_config.MODEL_ID,
    )


def build_bx_agent():
    """Builds the BxAgent using the chat model."""
    model = build_chat_model()
    return create_deep_agent(model=model)


def main():
    logging.info("Starting BxAgent with configuration: %s", agent_config)

    bx_agent = build_bx_agent()
    logging.debug(f"BxAgent initialized successfully.")

    # TODO: Remove this example after testing
    response = bx_agent.invoke(
        {"messages": [HumanMessage(content="Hello, how are you?")]}
    )
    logging.info(f"Received response from chat model: {response}")


if __name__ == "__main__":
    main()
