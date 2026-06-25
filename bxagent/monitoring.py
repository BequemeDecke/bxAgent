from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from bxagent.config import Config


# --- Builder Functions ---
def build_langfuse_client():
    """Builds and returns a Langfuse client and its corresponding callback handler for monitoring.

        Example usage:
        ```python
        langfuse_client, langfuse_handler = build_langfuse_client()

        agent.invoke(input=input_state, config={callbacks=[langfuse_handler]})

        # Flush events to Langfuse in short-lived applications
        langfuse.flush()
        ```
    """
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
