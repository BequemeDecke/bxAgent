import logging
import os

from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, SecretStr, Field


class BxAgentConfig(BaseModel):
    """Configuration class for BxAgent."""

    API_KEY: SecretStr = Field()
    BASE_URL: str = Field()
    BASE_MODEL: str = Field()
    CODING_MODEL: str = Field()


class LangFuseConfig(BaseModel):
    SECRET_KEY: SecretStr = Field()
    PUBLIC_KEY: SecretStr = Field()
    BASE_URL: str = Field()


class Config(BaseModel):
    """Main configuration class that holds all configurations for the application."""

    BX_AGENT: BxAgentConfig
    LANGFUSE: LangFuseConfig
    
    @classmethod
    def get_instance(cls) -> 'Config':
        """Singleton pattern to get a single instance of the configuration."""
        if not hasattr(cls, "_instance"):
            cls._instance = load_config()
        return cls._instance    


def load_config(env_path: Path = Path.cwd() / ".env") -> BaseModel:
    # Load environment variables from the .env file
    has_env_loaded = load_dotenv(dotenv_path=env_path)
    assert has_env_loaded, f"Failed to load environment variables from {env_path}"

    # Save the loaded environment variables to a config class for easy access
    agent_config = BxAgentConfig(
        API_KEY=os.getenv("API_KEY"),
        BASE_URL=os.getenv("BASE_URL"),
        BASE_MODEL=os.getenv("BASE_MODEL"),
        CODING_MODEL=os.getenv("CODING_MODEL"),
    )

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
    return Config(BX_AGENT=agent_config, LANGFUSE=langfuse_config)
