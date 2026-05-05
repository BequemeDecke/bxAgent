import logging
import os

from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, SecretStr, Field

# Load environment variables from the .env file
dotenv_path = (
    Path.cwd() / ".env"
)  # Env file is located one level up from the src directory
has_env_loaded = load_dotenv(dotenv_path=dotenv_path)
assert has_env_loaded, f"Failed to load environment variables from {dotenv_path}"


# Save the loaded environment variables to a config class for easy access
class BxAgentConfig(BaseModel):
    """Configuration class for BxAgent."""

    API_KEY: SecretStr = Field()
    BASE_URL: str = Field()
    BASE_MODEL: str = Field()
    CODING_MODEL: str = Field()


agent_config = BxAgentConfig(
    API_KEY=os.getenv("API_KEY"),
    BASE_URL=os.getenv("BASE_URL"),
    BASE_MODEL=os.getenv("BASE_MODEL"),
    CODING_MODEL=os.getenv("CODING_MODEL"),
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

# Log the loaded configurations
logging.debug("--- Loaded Configurations ---")
logging.debug(f"Loaded BxAgentConfig: {agent_config}")
logging.debug(f"Loaded LangFuseConfig: {langfuse_config}")
logging.debug("-----------------------------")