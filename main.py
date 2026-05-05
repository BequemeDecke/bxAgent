import logging
import sys
import uuid

from pathlib import Path
from langchain.messages import HumanMessage

from src.agent import build_bx_agent, build_langfuse_client


# --- Main Execution ---
def main():
    if len(sys.argv) < 3:
        logging.error(
            "No custom workspace directory or prompt provided. Using default values. To specify custom values, run the script with: python main.py <workspace_dir> <prompt>"
        )
        exit(1)

    workspace_dir = Path(sys.argv[1])
    input_prompt = sys.argv[2]

    logging.debug(f"Using workspace directory: {workspace_dir}")

    bx_agent = build_bx_agent(workspace_dir=workspace_dir)
    logging.debug(f"BxAgent initialized successfully.")
    
    langfuse_client, langfuse_handler = build_langfuse_client()
    logging.debug("Langfuse client and handler initialized successfully.")

    response = bx_agent.invoke(
        {"messages": [HumanMessage(content=input_prompt)]},
        {
            "configurable": {
                "thread_id": str(
                    uuid.uuid4()
                ),  # Maybe there are better ways to do that
            },
            "callbacks": [langfuse_handler],
        },
    )
    logging.info(f"Received response from bxAgent: {response}")
    langfuse_client.flush()  # Ensure all events are sent to Langfuse


if __name__ == "__main__":
    main()
