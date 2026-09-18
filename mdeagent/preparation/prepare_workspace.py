from abc import ABC, abstractmethod
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from mdeagent.comprehension import FileTransformationPlanParser, TransformationPlan
from mdeagent.evaluation.types import EvaluationRun
from mdeagent.preparation.maven import MavenProject
from mdeagent.preparation.naming import determine_transformation_class_name
from mdeagent.preparation.pom import Dependency, Plugin
from mdeagent.preparation.state import PreparationState

EMF_DEPENDENCIES: list[Dependency] = [
    Dependency(
        group_id="org.eclipse.emf",
        artifact_id="org.eclipse.emf.ecore",
        version="2.42.0",
    ),
    Dependency(
        group_id="org.eclipse.emf",
        artifact_id="org.eclipse.emf.common",
        version="2.42.0",
    ),
    Dependency(
        group_id="org.eclipse.emf",
        artifact_id="org.eclipse.emf.ecore.xmi",
        version="2.40.0",
    ),
]

# This is tested on Java 23.0.1
SPOTLESS_PLUGIN = Plugin(
    group_id="com.diffplug.spotless",
    artifact_id="spotless-maven-plugin",
    version="3.0.0",
    configuration="""
        <java>
            <includes>
                <include>src/main/java/**/*.java</include>
                <include>src/test/java/**/*.java</include>
            </includes>
            <palantirJavaFormat>
                <version>2.71.0</version>
            </palantirJavaFormat>
        </java>
    """,
)


class StructureFixStrategy(ABC):
    """
    Abstract base class for strategies to fix the workspace structure.
    """

    @abstractmethod
    def fix_structure(self, state: PreparationState) -> PreparationState:
        pass


# Evaluation id under which the ``WorkspaceStructureEvaluation`` is registered
# in the preparation graph (see ``mdeagent.preparation.agent.build_preparation_graph``).
# The preparation graph runs in ``execution_mode="specific"`` which stores the
# evaluation runs as a ``dict[str, EvaluationRun]`` keyed by this id.
WORKSPACE_STRUCTURE_EVALUATION_ID = "workspace_structure"


def _latest_workspace_structure_run(
    state: PreparationState,
) -> EvaluationRun | None:
    """Return the latest ``WorkspaceStructureEvaluation`` run stored in the state.

    The preparation graph stores its evaluation results in
    ``latest_evaluation_runs``. With ``execution_mode="specific"`` (the mode used
    by the preparation graph) the value is a ``dict[str, EvaluationRun]`` keyed by
    evaluation id, where the workspace structure evaluation is stored under
    :data:`WORKSPACE_STRUCTURE_EVALUATION_ID`. The ``"all"`` execution mode returns
    a flat ``list[EvaluationRun]`` in which the workspace structure run cannot be
    identified reliably by id; in that case ``None`` is returned so that callers
    fall back to the safe default (treat the structure as *not* clean).
    """
    latest_results = state.get("latest_evaluation_runs") or {}
    if isinstance(latest_results, dict):
        return latest_results.get(WORKSPACE_STRUCTURE_EVALUATION_ID)
    return None


def workspace_structure_is_clean(state: PreparationState) -> bool:
    """Return ``True`` if the latest ``WorkspaceStructureEvaluation`` is clean.

    This replaces the former filesystem-based ``is_workspace_structure_correct``
    helper which was only a weak duplicate of the
    :class:`mdeagent.evaluation.implementations.workspace_structure.WorkspaceStructureEvaluation`.
    Instead of re-checking the workspace on disk, the decision is now based on the
    latest evaluation results produced by the ``evaluate_preparation`` node.

    A run is considered *clean* when it has no ``EvaluationError`` entries and none
    of its ``EvaluationResult`` entries carries ``success=False``. When no
    workspace_structure run is available yet (e.g. on a direct node invocation
    without a preceding evaluation), the structure is treated as *not* clean so
    that the :class:`StructureFixStrategy` gets a chance to repair the workspace.
    """
    run = _latest_workspace_structure_run(state)
    if run is None:
        return False
    if len(run.errors) > 0:
        return False
    return not any(
        result.metadata.get("success", True) is False for result in run.results
    )


def create_prepare_workspace_node(
    fix_strategy: StructureFixStrategy,
    benchmarx_path: Path | None = None,
    download_benchmarx: bool = False,
):
    def prepare_workspace_node(state: PreparationState) -> PreparationState:
        workspace = state.get("workspace_path")
        if workspace is None:
            raise ValueError("Workspace path is not set in the state.")

        group_id = state.get("group_id")
        if group_id is None:
            raise ValueError("Group ID is not set in the state.")

        artifact_id = state.get("artifact_id")
        if artifact_id is None:
            raise ValueError("Artifact ID is not set in the state.")

        # Get benchmarx_path from state (can override the parameter)
        state_benchmarx_path = state.get("benchmarx_path", benchmarx_path)

        # Create workspace if directory does not exist
        workspace.mkdir(parents=True, exist_ok=True)

        # Decide what to do based on the current workspace state. The decision
        # relies on the latest ``WorkspaceStructureEvaluation`` results stored in
        # the state (produced by the ``evaluate_preparation`` node) instead of a
        # separate filesystem check. See :func:`workspace_structure_is_clean`.
        fixed_state = {}  # State to overwrite
        if not any(workspace.iterdir()):
            # Anforderung 3: The workspace is empty (only the parent folder
            # exists) -> create the workspace as usual.
            parent_artifact_id = workspace.name
            parent_project = MavenProject.create(
                workspace, group_id, parent_artifact_id, None
            )
            project = MavenProject.create(
                workspace, group_id, artifact_id, parent_project
            )
        elif workspace_structure_is_clean(state):
            # Anforderung 1: A workspace from a previous iteration already exists
            # and the latest ``WorkspaceStructureEvaluation`` reported no problems.
            # There is nothing to do within this node, so return early without
            # touching the workspace. The previously prepared state
            # (transformation plan, paths, ...) is preserved in the LangGraph
            # state and only the iteration counter is advanced.
            current_iteration = state.get("iteration", 0)
            return PreparationState(iteration=current_iteration + 1)
        else:
            # Anforderung 2: The workspace is not clean -> apply the
            # ``StructureFixStrategy`` to repair the workspace structure.
            fixed_state = fix_strategy.fix_structure(state)
            project = MavenProject.load(
                workspace / artifact_id
            )  # Workspace should be fixed

        # Create the transformation module (Maven project) inside the workspace
        # Package path includes artifact_id as subpackage (e.g., de.example.mdeagent)
        full_package = f"{group_id}.{artifact_id}"
        package_path = project.get_package_path(full_package)

        # Create the TRANSFORMATION.md file
        transformation_md_path = workspace / artifact_id / "TRANSFORMATION.md"
        tp_parser = FileTransformationPlanParser(transformation_md_path)
        tp = TransformationPlan.parse(
            parser=tp_parser
        )  # Transformation plan is created if not existing, else loaded

        # Derive the transformation class name (and the BxTool adapter name)
        # deterministically from the *folder names* of the source and target
        # model packages. Each model package folder carries the name of its
        # metamodel, so the names follow the pattern
        # ``<Source>To<Target>Transformation`` / ``<Source>To<Target>BxToolAdapter``
        # (e.g. ``Families`` -> ``Persons`` yields ``FamiliesToPersonsTransformation``).
        # This replaces the former LLM-based naming which was too error-prone.
        source_model = state.get("source_model")
        target_model = state.get("target_model")
        naming = determine_transformation_class_name(
            source_model_path=source_model.get("path") if source_model else None,
            target_model_path=target_model.get("path") if target_model else None,
        )
        transformation_class_name = naming.transformation_class_name
        # Calculate transformation class path but don't create the file (user will implement it)
        transformation_class_path = (
            project.get_package_path(full_package) / f"{transformation_class_name}.java"
        )

        # Create the BxTool adapter Java file ONLY if BenchmarX is NOT being used
        # BenchmarX is not used when: benchmarx_path is None AND download_benchmarx is False
        create_bxtool_adapter = state_benchmarx_path is None and not download_benchmarx

        if create_bxtool_adapter:
            bxtool_class_name = naming.bxtool_adapter_class_name
            bxtool_path = project.add_java_class(
                package=full_package,
                class_name=bxtool_class_name,
                content=f"public class {bxtool_class_name} {{\n    // TODO: Implement the BxTool adapter logic here\n}}\n",
            )
        else:
            # BenchmarX will be used, no BxTool adapter needed
            bxtool_path = None

        # Copy the AgentTransformationForEMF.java file into the package path
        agent_transformation_template = Environment(
            loader=FileSystemLoader(Path.cwd() / "templates")
        ).get_template("agent_transformation_interface.jinja")
        project.add_java_class(
            package=full_package,
            class_name="AgentTransformationForEMF",
            content=agent_transformation_template.render(package_path=full_package),
        )

        # Delete the App.java file created by the Maven archetype
        app_java_path = package_path / ".." / "App.java"
        if app_java_path.exists():
            app_java_path.unlink()

        # Add EMF dependencies to the pom.xml of the transformation module
        for dependency in EMF_DEPENDENCIES:
            project.pom.add_dependency(dependency)
        project.pom.add_plugin(SPOTLESS_PLUGIN)
        project.pom.save()
        project.validate()

        # Track how often the workspace has been (re-)prepared. This counter is
        # used by the conditional edge after `evaluate_preparation` to detect the
        # very first run (iteration == 0) and to break the loop once the workspace
        # is correctly set up. Incrementing it here keeps prepare_workspace as the
        # single driver of the preparation loop.
        current_iteration = state.get("iteration", 0)

        # Update the state with the new paths and transformation plan
        new_state = PreparationState(
            transformation_plan=tp,
            bxtool_path=bxtool_path,
            transformation_class_path=transformation_class_path,
            maven_project_path=project.workspace,
            transformation_package_path=full_package,
            iteration=current_iteration + 1,
        )
        new_state.update(fixed_state)
        return new_state

    return prepare_workspace_node
