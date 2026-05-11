from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from src.config import Config


# --- Builder Functions ---
def build_langfuse_client():
    langfuse_config = Config.get_instance().LANGFUSE
    # Initialize Langfuse client with constructor arguments
    client = Langfuse(
        secret_key=langfuse_config.SECRET_KEY.get_secret_value(),
        public_key=langfuse_config.PUBLIC_KEY.get_secret_value(),
        host=langfuse_config.BASE_URL,
    )

    # Initialize the Langfuse handler
    langfuse_handler = LangfuseCallbackHandler()
    return client, langfuse_handler
