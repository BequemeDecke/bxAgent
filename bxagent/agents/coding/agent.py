from deepagents import create_deep_agent

from bxagent.config import Config
from bxagent import models


def build_coding_deep_agent():
    config = Config.get_instance()
    model = models.build_coding_model()

    return create_deep_agent(
        model=model,
    )
