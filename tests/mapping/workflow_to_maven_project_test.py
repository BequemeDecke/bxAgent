from pathlib import Path
from unittest import TestCase

from mdeagent.mapping.workflow_to_maven_project import map_workflow_to_maven_project
from mdeagent.state import MDEAgentState


class TestMapWorkflowToMavenProject(TestCase):
    """Test cases for map_workflow_to_maven_project function."""

    def test_map_with_maven_project_path(self):
        """Test mapping when maven_project_path is set in state."""
        workspace_path = Path("/workspace")
        maven_project_path = Path("/workspace/my-project")
        
        state = MDEAgentState(
            workspace_path=workspace_path,
            maven_project_path=maven_project_path,
            source_model_path=Path("/source"),
            target_model_path=Path("/target"),
            group_id="com.example",
            artifact_id="my-artifact",
        )
        
        result = map_workflow_to_maven_project(state)
        
        self.assertEqual(result, {"project_path": maven_project_path})
        self.assertEqual(result["project_path"], maven_project_path)

    def test_map_without_maven_project_path_raises_error(self):
        """Test mapping when maven_project_path is not set in state."""
        # Create state without maven_project_path
        state: MDEAgentState = {
            "workspace_path": Path("/workspace"),
            "source_model_path": Path("/source"),
            "target_model_path": Path("/target"),
            "group_id": "com.example",
            "artifact_id": "my-artifact",
            "transformation_plan": None,
            "transformation_package_path": "",
            "transformation_class_path": None,
            "bxtool_path": None,
            "required_commands": [],
            "written_files": [],
            "latest_evaluation_runs": [],
        }
        
        with self.assertRaises(KeyError) as context:
            map_workflow_to_maven_project(state)
        
        self.assertIn("maven_project_path", str(context.exception))

    def test_map_preserves_path_type(self):
        """Test that the returned value maintains Path type."""
        workspace_path = Path("/workspace")
        maven_project_path = Path("/workspace/project/submodule")
        
        state: MDEAgentState = {
            "workspace_path": workspace_path,
            "maven_project_path": maven_project_path,
            "source_model_path": Path("/source"),
            "target_model_path": Path("/target"),
            "group_id": "com.example",
            "artifact_id": "my-artifact",
            "transformation_plan": None,
            "transformation_package_path": "",
            "transformation_class_path": None,
            "bxtool_path": None,
            "required_commands": [],
            "written_files": [],
            "latest_evaluation_runs": [],
        }
        
        result = map_workflow_to_maven_project(state)
        
        self.assertIsInstance(result["project_path"], Path)
        self.assertEqual(result["project_path"], maven_project_path)

    def test_map_with_nested_maven_project_path(self):
        """Test mapping with deeply nested Maven project path."""
        maven_project_path = Path("/workspace/parent-module/child-module/grandchild-module")
        
        state: MDEAgentState = {
            "workspace_path": Path("/workspace"),
            "maven_project_path": maven_project_path,
            "source_model_path": Path("/source"),
            "target_model_path": Path("/target"),
            "group_id": "com.example",
            "artifact_id": "grandchild-module",
            "transformation_plan": None,
            "transformation_package_path": "",
            "transformation_class_path": None,
            "bxtool_path": None,
            "required_commands": [],
            "written_files": [],
            "latest_evaluation_runs": [],
        }
        
        result = map_workflow_to_maven_project(state)
        
        self.assertEqual(result["project_path"], maven_project_path)
        self.assertEqual(
            str(result["project_path"]),
            "/workspace/parent-module/child-module/grandchild-module"
        )
