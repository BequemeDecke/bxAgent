import logging
import uuid

from argparse import ArgumentParser
from pathlib import Path
from langchain.messages import HumanMessage

from mdagent.agents import build_bx_agent


def parse_arguments():
    parser = ArgumentParser(
        description="Run the BxAgent with specified workspace and prompt."
    )
    parser.add_argument(
        "--workspace-dir", "-w", type=str, help="Directory for the agent's workspace.", required=True
    )
    parser.add_argument(
        "--prompt", "-p", type=str, help="The prompt to send to the BxAgent.", required=True
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--use-langfuse",
        action="store_true",
        help="Whether to use Langfuse for logging and monitoring.",
    )
    return parser.parse_args()


# --- Main Execution ---
def main():
    args = parse_arguments()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    workspace_dir = Path(args.workspace_dir)
    logging.debug(f"Using workspace directory: {workspace_dir}")
    
    input_prompt = args.prompt
    logging.debug(f"Using input prompt: {input_prompt}")

    bx_agent = build_bx_agent(workspace_dir=workspace_dir)
    logging.debug(f"BxAgent initialized successfully.")
    
    config = {
            "configurable": {
                "thread_id": str(
                    uuid.uuid4()
                ),  # Maybe there are better ways to do that
            },
        }
    
    if args.use_langfuse:
        from mdagent.monitoring import build_langfuse_client # Dynamic import to avoid unnecessary dependency if not using Langfuse
        
        langfuse_client, langfuse_handler = build_langfuse_client()
        config["callbacks"] = [langfuse_handler]
        logging.debug("Langfuse client and handler initialized successfully.")

    response = bx_agent.invoke(
        {"messages": [HumanMessage(content=input_prompt)]},
        config,
    )
    logging.info(f"Received response from mdagent: {response}")
    
    if args.use_langfuse:
        langfuse_client.flush()  # Ensure all events are sent to Langfuse


if __name__ == "__main__":
    main()
