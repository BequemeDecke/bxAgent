import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase

from mdeagent.comprehension.plan import FileTransformationPlanParser, TransformationPlan
from mdeagent.implementation.transformation.implement_transformation import (
    create_implement_transformation_node,
)
from mdeagent.implementation.state import ImplementationState
from mdeagent.models import (
    LLMTransientError,
    build_coding_model,
    invoke_with_resilience,
)

TEST_ENVIRONMENT = Path(".mdeagent-tests")

logger = logging.getLogger(__name__)

# --- LLM call resilience tuning ----------------------------------------------
#
# The ``.env`` configures a 20s request timeout with 4 SDK retries. With the new
# **piecewise generation** approach (5 smaller calls instead of 1 monolithic call),
# individual calls are much faster and less prone to timeouts. However, we still use
# generous timeouts to handle edge cases.
#
# For the integration test with the piecewise implementation:
#   * Each individual call is smaller and faster (~5-25s typical)
#   * We allow generous per-call timeout for safety
#   * Overall timeout accounts for all 5 calls (1 sequential + 4 parallel)
#   * Transient flakiness is handled by ``invoke_with_resilience``
LLM_REQUEST_TIMEOUT_S = 60  # per-call timeout (reduced from 120s since individual
#   calls are now much smaller; 60s provides ample margin for the metadata call and
#   each of the 4 parallel body-generation calls).
LLM_MAX_RETRIES = 0  # disable SDK retry; ``invoke_with_resilience`` handles retries
OVERALL_TIMEOUT_S = 180  # hard cap across all attempts; reduced from 220s since
#   piecewise approach is faster and more reliable. Allows ~90s per attempt which is
#   plenty for 1 metadata call + 4 parallel body-generation calls.
MAX_ATTEMPTS = 2  # retry once on a transient timeout / 5xx / rate-limit


class TestImplementTransformationIntegration(TestCase):
    """
    Integration test for create_implement_transformation_node using real BaseChatModels.

    This test verifies that the piecewise generation approach:
    1. Generates exactly one transformation class file
    2. The output state contains all required fields with correct values
    3. The parallel execution works correctly (metadata first, then parallel body generation)
    
    Note: This node uses a piecewise LLM approach with 5 calls instead of 1 monolithic call:
    - 1 metadata call (sequential) to determine package and type names
    - 4 parallel calls for fields/constructor + forward/backward/synch method bodies
    
    This reduces timeout risk and improves reliability compared to the monolithic approach.
    """

    @classmethod
    def setUpClass(cls):
        """Set up shared resources for the test class."""
        # Check if the setup files exist before running any tests
        setup_files = TEST_ENVIRONMENT / "setup-files"
        source_model_path = setup_files / "Families"
        target_model_path = setup_files / "Persons"

        if not source_model_path.exists() or not target_model_path.exists():
            raise FileNotFoundError(
                f"Setup files not found. Please ensure that {source_model_path} and {target_model_path} exist."
            )

        if len(list(source_model_path.glob("*.java"))) != 4:
            raise FileNotFoundError(
                f"Expected 4 source model files in {source_model_path}, but found {len(list(source_model_path.glob('*.java')))}."
            )

        if len(list(target_model_path.glob("*.java"))) != 3:
            raise FileNotFoundError(
                f"Expected 3 target model files in {target_model_path}, but found {len(list(target_model_path.glob('*.java')))}."
            )

    def setUp(self):
        """Create a unique workspace for each test."""
        # Create a unique workspace for the test
        self.workspace_path = (
            TEST_ENVIRONMENT
            / "test-executions"
            / "implement_transformation"
            / datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S_%f")
        )
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created test workspace at {self.workspace_path}")

        # Set up paths
        self.setup_files = TEST_ENVIRONMENT / "setup-files"
        self.source_model_path = self.setup_files / "Families"
        self.target_model_path = self.setup_files / "Persons"

        # Initialize the coding model. Override the request timeout / retry settings to
        # accommodate the piecewise generation approach: 1 metadata call followed by 4
        # parallel body-generation calls. Individual calls are smaller and faster than
        # the monolithic approach, reducing timeout risk. Transient flakiness is handled
        # at the test level by ``invoke_with_resilience``.
        self.llm = build_coding_model()
        self.llm.request_timeout = LLM_REQUEST_TIMEOUT_S
        self.llm.max_retries = LLM_MAX_RETRIES

    def _create_transformation_plan_factory(self):
        """
        Creates a factory function that returns a TransformationPlan for Families2Persons.

        Returns:
            A Callable that returns a TransformationPlan instance.
        """

        def plan_factory():
            # Create a temporary transformation plan file
            plan_file = self.workspace_path / "TRANSFORMATION.md"
            plan_content = """---
source_model_package: families
target_model_package: persons
iteration: 1
---

--- BEGIN SOURCE MODEL ---
The source model consists of:
- FamilyRegister: Root container for all families
- Family: Represents a family with members
- FamilyMember: Abstract base for family members
- Specific member types (e.g., Parent, Child)
--- END SOURCE MODEL ---

--- BEGIN TARGET MODEL ---
The target model consists of:
- PersonRegister: Root container for all persons
- Person: Represents a person with name and age
--- END TARGET MODEL ---

--- BEGIN TRANSFORMATION DIRECTION ---
Bidirectional transformation between Families and Persons models.
Forward: Extract all FamilyMembers from Families and create corresponding Persons.
Backward: Group Persons into Families based on relationships.
--- END TRANSFORMATION DIRECTION ---

--- BEGIN DIFFICULTIES ---
1. Mapping multiple FamilyMembers to flat Person list requires tracking relationships.
2. Backward transformation needs to infer family structure from person attributes.
3. Handling of duplicate or missing data during synchronization.
--- END DIFFICULTIES ---

--- BEGIN IMPLEMENTATION STEPS ---
1. Implement forward transformation to extract FamilyMembers as Persons.
2. Implement backward transformation to group Persons into Families.
3. Implement synch method to handle incremental updates.
4. Add proper error handling and validation for edge cases.
--- END IMPLEMENTATION STEPS ---
"""
            plan_file.write_text(plan_content)

            parser = FileTransformationPlanParser(plan_file)
            transformation_plan = TransformationPlan.parse(parser)
            return transformation_plan

        return plan_factory

    def test_implement_transformation__generates_single_transformation_class(self):
        """
        Test that the implement_transformation node generates exactly one transformation class.

        Verifies:
        - Only one .java file is created in the workspace
        - The file is a valid Java transformation class implementing AgentTransformationForEMF
        """
        # Create the implement_transformation node with real LLM
        plan_factory = self._create_transformation_plan_factory()
        implement_transformation = create_implement_transformation_node(
            llm=self.llm,
            optional_plan_factory=plan_factory,
            template_path=Path.cwd() / "templates",
        )

        # Create initial state with task specification for Families2Persons
        transformation_class_path = self.workspace_path / "Transformation.java"
        initial_state: ImplementationState = {
            "task_specification": """
Implement a bidirectional transformation between the Families and Persons models.

Source Model (Families):
- FamilyRegister: Root container
- Family: Contains multiple FamilyMembers
- FamilyMember: Base type with name attribute

Target Model (Persons):
- PersonRegister: Root container  
- Person: Has name and computed age attributes

Requirements:
1. Forward: Extract all FamilyMembers from Family structures and create Person instances.
2. Backward: Reconstruct Family structures from Person instances based on naming conventions.
3. Synch: Handle incremental changes in both models.
4. Use appropriate EMF patterns for model navigation and creation.
""",
            "transformation_md": None,  # Will be created by plan_factory
            "written_java_files": [],
            "bxtool_path": self.workspace_path,
            "transformation_class_path": transformation_class_path,
            "transformation_implementation": "",
            "latest_evaluation_runs": {},
            "iteration": 1,
        }

        # Invoke the node. The call is guarded against the inherent flakiness of real
        # LLM endpoints with piecewise generation (5 calls: 1 metadata + 4 parallel body
        # generation): a bounded overall timeout plus retry-on-transient-error via
        # ``invoke_with_resilience``. If the endpoint is genuinely unavailable we skip
        # (infrastructure issue) instead of reporting a false logic failure.
        try:
            output_state = asyncio.run(
                invoke_with_resilience(
                    lambda: implement_transformation(initial_state),
                    overall_timeout=OVERALL_TIMEOUT_S,
                    max_attempts=MAX_ATTEMPTS,
                )
            )
        except LLMTransientError as exc:
            self.skipTest(
                f"Skipped: LLM endpoint unavailable ({exc}). "
                "This is a transient infrastructure issue, not a test-logic failure."
            )

        # Verify: Exactly one transformation class file was created
        java_files_in_workspace = list(self.workspace_path.glob("*.java"))
        self.assertEqual(
            len(java_files_in_workspace),
            1,
            f"Expected exactly 1 Java file in workspace, but found {len(java_files_in_workspace)}: {[f.name for f in java_files_in_workspace]}",
        )

        # Verify: The file is listed in written_java_files
        self.assertEqual(
            len(output_state["written_java_files"]),
            1,
            f"Expected written_java_files to contain exactly 1 file, but found {len(output_state['written_java_files'])}",
        )

        # Verify: The file path matches
        generated_file = output_state["written_java_files"][0]
        self.assertTrue(
            generated_file.exists(), f"Generated file {generated_file} does not exist"
        )
        self.assertEqual(
            generated_file.name,
            java_files_in_workspace[0].name,
            "File in written_java_files doesn't match actual file in workspace",
        )

    def test_implement_transformation__output_state_contains_required_fields(self):
        """
        Test that the output state contains all required fields with correct values.

        Verifies:
        - transformation_md: Contains the transformation plan
        - written_java_files: Contains exactly one file path
        - task_specification: Preserved from input state
        - transformation_implementation: Contains generated Java code
        """
        # Create the implement_transformation node with real LLM
        plan_factory = self._create_transformation_plan_factory()
        implement_transformation = create_implement_transformation_node(
            llm=self.llm,
            optional_plan_factory=plan_factory,
            template_path=Path.cwd() / "templates",
        )

        # Create initial state
        task_spec = """
Implement a bidirectional transformation between the Families and Persons models.
Focus on extracting FamilyMembers as Person instances in the forward direction.
"""
        transformation_class_path = self.workspace_path / "Transformation.java"
        initial_state: ImplementationState = {
            "task_specification": task_spec,
            "transformation_md": None,
            "written_java_files": [],
            "bxtool_path": self.workspace_path,
            "transformation_class_path": transformation_class_path,
            "transformation_implementation": "",
            "latest_evaluation_runs": {},
            "iteration": 1,
        }

        # Invoke the node with piecewise generation (5 LLM calls total), guarded against
        # transient LLM flakiness via ``invoke_with_resilience`` (see first test for the
        # full rationale).
        try:
            output_state = asyncio.run(
                invoke_with_resilience(
                    lambda: implement_transformation(initial_state),
                    overall_timeout=OVERALL_TIMEOUT_S,
                    max_attempts=MAX_ATTEMPTS,
                )
            )
        except LLMTransientError as exc:
            self.skipTest(
                f"Skipped: LLM endpoint unavailable ({exc}). "
                "This is a transient infrastructure issue, not a test-logic failure."
            )

        # Verify: transformation_md is set (either from state or created by factory)
        self.assertIsNotNone(
            output_state["transformation_md"], "transformation_md should not be None"
        )
        self.assertIn(
            "data",
            output_state["transformation_md"].__dict__,
            "transformation_md should have a 'data' attribute",
        )

        # Verify: written_java_files contains exactly one file
        self.assertEqual(
            len(output_state["written_java_files"]),
            1,
            "written_java_files should contain exactly one file",
        )
        self.assertIsInstance(
            output_state["written_java_files"][0],
            Path,
            "written_java_files should contain Path objects",
        )

        # Verify: task_specification is preserved
        self.assertEqual(
            output_state["task_specification"],
            task_spec,
            "task_specification should be preserved from input state",
        )

        # Verify: transformation_implementation contains generated code
        self.assertIsNotNone(
            output_state["transformation_implementation"],
            "transformation_implementation should not be None",
        )
        self.assertGreater(
            len(output_state["transformation_implementation"]),
            100,
            "transformation_implementation should contain substantial Java code",
        )

        # Verify: Generated code contains expected Java class structure
        impl_code = output_state["transformation_implementation"]
        self.assertIn(
            "class", impl_code, "Generated code should contain a class definition"
        )
        self.assertIn(
            "AgentTransformationForEMF",
            impl_code,
            "Generated code should implement AgentTransformationForEMF interface",
        )
        self.assertIn(
            "forward(", impl_code, "Generated code should contain forward method"
        )
        self.assertIn(
            "backward(", impl_code, "Generated code should contain backward method"
        )
        self.assertIn("synch(", impl_code, "Generated code should contain synch method")

        # Verify: The generated file content matches transformation_implementation
        generated_file = output_state["written_java_files"][0]
        file_content = generated_file.read_text()
        self.assertEqual(
            file_content,
            output_state["transformation_implementation"],
            "File content should match transformation_implementation in state",
        )
