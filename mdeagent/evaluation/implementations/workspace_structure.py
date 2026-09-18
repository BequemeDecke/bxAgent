import logging
from pathlib import Path

from pydantic import BaseModel

from mdeagent.comprehension import FileTransformationPlanParser
from mdeagent.evaluation.types import Evaluation, EvaluationError, EvaluationResult
from mdeagent.preparation.maven import MavenProject
from mdeagent.preparation.pom import Pom

logger = logging.getLogger(__name__)

# Artifact id of the Spotless Maven plugin that has to be registered in the
# transformation module's pom.xml (see mdeagent.preparation.prepare_workspace.SPOTLESS_PLUGIN).
SPOTLESS_PLUGIN_ARTIFACT_ID = "spotless-maven-plugin"

# Java class that has to be present in the transformation module's package path.
AGENT_TRANSFORMATION_CLASS_NAME = "AgentTransformationForEMF"


class WorkspaceStructureSchema(BaseModel):
    workspace_path: Path
    artifact_id: str
    package_path: str


class WorkspaceStructureEvaluation(Evaluation):
    async def setup(self) -> None:
        pass

    async def run(
        self, **kwargs
    ) -> tuple[list[EvaluationResult], list[EvaluationError]]:
        results: list[EvaluationResult] = []
        errors: list[EvaluationError] = []

        workspace_path: Path = kwargs.get("workspace_path")
        artifact_id: str = kwargs.get("artifact_id")
        package_path: str = kwargs.get("package_path")

        # 1. The workspace directory itself must exist.
        if not workspace_path.exists() or not workspace_path.is_dir():
            results.append(
                EvaluationResult(
                    content=f"Workspace path '{workspace_path}' does not exist or is not a directory.",
                    metadata={"success": False, "include_in_report": False},
                )
            )
            return results, errors

        module_path = workspace_path / artifact_id

        # 2. The transformation module (workspace/[artifact_id]) must be loadable as a
        #    Maven project. If this fails, something is fundamentally wrong with the
        #    workspace structure already.
        #    Note: In early iterations the Maven project may not exist yet, which is
        #    expected behavior. We handle this gracefully without logging an error.
        maven_project: MavenProject | None = None
        try:
            maven_project = MavenProject.load(module_path)
        except FileNotFoundError as e:
            # Expected in early iterations when Maven project hasn't been created yet
            results.append(
                EvaluationResult(
                    content=f"Maven project not found at '{module_path}': {e}",
                    metadata={"success": False, "include_in_report": True},
                )
            )
        except Exception as e:
            # Unexpected error - log it but still return as evaluation result
            logger.debug(f"Unexpected error loading Maven project at {module_path}: {e}")
            results.append(
                EvaluationResult(
                    content=f"Failed to load Maven project at '{module_path}': {e}",
                    metadata={"success": False, "include_in_report": True},
                )
            )

        # 3. The parent pom.xml (workspace/pom.xml) must register the module [artifact_id].
        parent_pom_path = workspace_path / "pom.xml"
        try:
            parent_pom = Pom(parent_pom_path)
            registered_modules = [module.artifact_id for module in parent_pom.modules]
            if artifact_id not in registered_modules:
                results.append(
                    EvaluationResult(
                        content=f"Module '{artifact_id}' is not registered in '{parent_pom_path}'.",
                        metadata={"success": False, "include_in_report": False},
                    )
                )
        except Exception as e:
            results.append(
                EvaluationResult(
                    content=f"Failed to parse parent pom.xml at '{parent_pom_path}': {e}",
                    metadata={"success": False, "include_in_report": False},
                )
            )

        # 4. The module pom.xml (workspace/[artifact_id]/pom.xml) must register the
        #    Spotless plugin.
        if maven_project is not None:
            registered_plugins = [
                plugin.artifact_id for plugin in maven_project.pom.plugins
            ]
            if SPOTLESS_PLUGIN_ARTIFACT_ID not in registered_plugins:
                results.append(
                    EvaluationResult(
                        content=f"Spotless plugin '{SPOTLESS_PLUGIN_ARTIFACT_ID}' is not registered in '{module_path / 'pom.xml'}'.",
                        metadata={"success": False, "include_in_report": False},
                    )
                )

        # 5. The package path must exist under src/main/java.
        java_source_root = module_path / "src" / "main" / "java"
        package_parts = package_path.split(".")
        package_dir = java_source_root.joinpath(*package_parts)
        current_path = java_source_root
        package_path_valid = True
        for part in package_parts:
            current_path = current_path / part
            if not current_path.exists() or not current_path.is_dir():
                results.append(
                    EvaluationResult(
                        content=f"Package path '{package_path}' is invalid. Missing directory: '{current_path}'.",
                        metadata={"success": False, "include_in_report": False},
                    )
                )
                package_path_valid = False
                break

        # 6. The AgentTransformationForEMF.java file must exist in the package path.
        if package_path_valid:
            agent_transformation_path = (
                package_dir / f"{AGENT_TRANSFORMATION_CLASS_NAME}.java"
            )
            if (
                not agent_transformation_path.exists()
                or not agent_transformation_path.is_file()
            ):
                results.append(
                    EvaluationResult(
                        content=f"Required file '{AGENT_TRANSFORMATION_CLASS_NAME}.java' is missing in '{package_dir}'.",
                        metadata={"success": False, "include_in_report": False},
                    )
                )

        # 7. The TRANSFORMATION.md file must be parseable by the TransformationPlan
        #    machinery from the Comprehension module.
        transformation_md_path = module_path / "TRANSFORMATION.md"
        try:
            FileTransformationPlanParser(transformation_md_path).parse()
        except Exception as e:
            results.append(
                EvaluationResult(
                    content=f"Required file 'TRANSFORMATION.md' is missing or invalid at '{transformation_md_path}': {e}",
                    metadata={"success": False, "include_in_report": False},
                )
            )

        return results, errors
