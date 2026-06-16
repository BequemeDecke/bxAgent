from pathlib import Path

from .state import PreparationState


def create_explore_models_node():
    def explore_models(state: PreparationState):
        source_model_path: Path = state.get("source_model_path")
        target_model_path: Path = state.get("target_model_path")

        if source_model_path is None or target_model_path is None:
            raise ValueError(
                "Source model path and target model path must be set in the state."
            )
        elif not source_model_path.exists():
            raise ValueError(f"Source model path '{source_model_path}' does not exist.")
        elif not source_model_path.is_file():
            raise ValueError(f"Source model path '{source_model_path}' is not a file.")
        elif not target_model_path.exists():
            raise ValueError(f"Target model path '{target_model_path}' does not exist.")
        elif not target_model_path.is_file():
            raise ValueError(f"Target model path '{target_model_path}' is not a file.")

        source_model_implementation = source_model_path.read_text()
        if source_model_implementation.strip() == "":
            raise ValueError(f"Source model at '{source_model_path}' is empty.")

        target_model_implementation = target_model_path.read_text()
        if target_model_implementation.strip() == "":
            raise ValueError(f"Target model at '{target_model_path}' is empty.")

        return {
            "source_model_implementation": source_model_implementation,
            "target_model_implementation": target_model_implementation,
        }

    return explore_models
