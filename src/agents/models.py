from langchain.chat_models import init_chat_model

from src.config import agent_config


def build_chat_model():
    """Builds the chat model using the loaded configuration."""
    return init_chat_model(
        model_provider="openai",  # TODO: Make this configurable later on; Counter: 1
        base_url=agent_config.BASE_URL,
        api_key=agent_config.API_KEY.get_secret_value(),
        model=agent_config.MODEL_ID,
    )
