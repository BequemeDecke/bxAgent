import os
import logging

from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, Field

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
    MODEL_ID=os.getenv("MODEL_ID")
)

def main():
    logging.info("Starting BxAgent with configuration: %s", agent_config)


if __name__ == "__main__":
    main()
