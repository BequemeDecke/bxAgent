from langchain.chat_models import init_chat_model

from mdeagent.config import Config


def build_base_model():
    """Builds the base model using the loaded configuration."""
    agent_config = Config.get_instance().BX_AGENT
    return init_chat_model(
        model_provider="openai",  # TODO: Make this configurable later on; Counter: 1
        base_url=agent_config.BASE_URL,
        api_key=agent_config.API_KEY.get_secret_value(),
        model=agent_config.BASE_MODEL,
    )


def build_coding_model():
    """Builds the coding model using the loaded configuration."""
    agent_config = Config.get_instance().BX_AGENT
    return init_chat_model(
        model_provider="openai",  # TODO: Make this configurable later on; Counter: 2
        base_url=agent_config.BASE_URL,
        api_key=agent_config.API_KEY.get_secret_value(),
        model=agent_config.CODING_MODEL,
    )
