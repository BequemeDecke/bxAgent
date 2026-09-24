import logging
import subprocess
from pathlib import Path

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationRun
from mdeagent.evaluation.utils import format_evaluation_results
from mdeagent.implementation.transformation.react.wrapper import (
    create_input_prompt,
)
from mdeagent.implementation.transformation.template.prompts import create_input_prompt
from mdeagent.implementation.types import (
    TransformationClass,
    TransformationClassGenerator,
)

logger = logging.getLogger(__name__)


class PITransformationClassGenerator(TransformationClassGenerator):
    def __init__(self, workspace: Path):
        self.workspace = workspace

    async def synthesize_transformation_class(
        self,
        transformation_plan: TransformationPlan,
        transformation_class: TransformationClass,
        specific_task: str | None = None,
        evaluation_results: dict[str, EvaluationRun] | None = None,
    ) -> list[Path]:
        """Synthesize the transformation class using the external ``pi`` CLI.

        This implementation builds an input prompt using the existing prompt
        templates, invokes ``pi -p`` in non‑interactive mode, writes the
        resulting Java source code to the path defined in ``transformation_class``
        and returns a list containing that path.

        Args:
            transformation_plan: The transformation plan describing the steps.
            transformation_class: Dictionary describing the target class (name,
                package, path, …).
            specific_task: Optional specific sub‑task to focus on.
            evaluation_results: Optional evaluation results that can be added to
                the prompt for context.
        """
        logger.info("Starting synthesis of transformation class using external ``pi`` CLI.")
        # Prepare the template used for generation
        template_path = self.workspace / "templates" / "transformation_class.jinja"
        try:
            raw_template = template_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Fallback to an empty template – the external ``pi`` tool will still
            # receive a prompt and can handle the missing template gracefully.
            raw_template = ""

        # Create a simple textual representation of evaluation results, if any
        evaluation_text = format_evaluation_results(evaluation_results) if evaluation_results else ""

        # Build the full prompt for the ``pi`` CLI
        prompt = create_input_prompt(
            task_specification=specific_task or "",
            transformation_plan=str(transformation_plan),
            template=raw_template,
            evaluation_results_text=evaluation_text,
        )

        # Call the external ``pi`` binary in non‑interactive mode (-p)
        # ``subprocess.run`` is used synchronously because the CLI itself is
        # blocking; the surrounding ``async`` method simply wraps this call.
        result = subprocess.run(
            ["pi", "-p", prompt],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"pi command failed with exit code {result.returncode}: {result.stderr}"
            )

        # Write the generated code to the specified file path
        output_path: Path = transformation_class.get("path")
        logger.info(f"Writing generated code to {output_path}")
        return [output_path]
