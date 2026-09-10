"""
Integration tests for JavaCompilationEvaluation with the evaluation executor and mapper.

These tests verify that JavaCompilationEvaluation works correctly when used with
the EvaluationExecutor and the map_workflow_to_maven_project mapper.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mdeagent.evaluation import (
    EvaluationExecutor,
    JavaCompilationEvaluation,
    JavaCompilationEvaluationConfig,
)
from mdeagent.evaluation.node import create_evaluation_node
from mdeagent.mapping.workflow_to_maven_project import map_workflow_to_maven_project
from mdeagent.preparation.maven import MavenProject
from mdeagent.preparation.pom import Pom
from mdeagent.state import MDEAgentState


class TestJavaCompilationWithExecutor(TestCase):
    """Test JavaCompilationEvaluation integration with EvaluationExecutor."""

    def test_executor_with_java_compilation(self):
        """Test that EvaluationExecutor can execute JavaCompilationEvaluation."""
        mock_factory = MagicMock()
        mock_project = MagicMock(spec=MavenProject)
        mock_project.compile.return_value = (True, "BUILD SUCCESS")
        mock_project.workspace = Path("/test/workspace")
        mock_factory.return_value = mock_project
        
        evaluation = JavaCompilationEvaluation(maven_project_factory=mock_factory)
        
        executor = EvaluationExecutor(
            evaluations={
                "java_compilation": {
                    "evaluation": evaluation,
                    "evaluation_schema": JavaCompilationEvaluationConfig,
                }
            }
        )
        
        # Execute the evaluation
        result = asyncio.run(
            executor.execute_specific(
                evaluation_id="java_compilation",
                input={"project_path": Path("/test/workspace")},
            )
        )
        
        self.assertEqual(len(result.results), 1)
        self.assertTrue(result.results[0].metadata["success"])
        mock_factory.assert_called_once_with(Path("/test/workspace"))

    def test_executor_with_failed_compilation(self):
        """Test Executor handling failed compilation."""
        mock_factory = MagicMock()
        mock_project = MagicMock(spec=MavenProject)
        error_output = "[ERROR] /workspace/Test.java:[10:5] cannot find symbol"
        mock_project.compile.return_value = (False, error_output)
        mock_project.workspace = Path("/workspace")
        mock_factory.return_value = mock_project
        
        evaluation = JavaCompilationEvaluation(maven_project_factory=mock_factory)
        
        executor = EvaluationExecutor(
            evaluations={
                "java_compilation": {
                    "evaluation": evaluation,
                    "evaluation_schema": JavaCompilationEvaluationConfig,
                }
            }
        )
        
        result = asyncio.run(
            executor.execute_specific(
                evaluation_id="java_compilation",
                input={"project_path": Path("/workspace")},
            )
        )
        
        # Should have parsed error results
        self.assertGreater(len(result.results), 0)


class TestJavaCompilationWithMapperAndNode(TestCase):
    """Test JavaCompilationEvaluation with mapper and evaluation node."""

    def test_node_with_maven_project_mapper(self):
        """Test evaluation node uses map_workflow_to_maven_project correctly."""
        mock_factory = MagicMock()
        mock_project = MagicMock(spec=MavenProject)
        mock_project.compile.return_value = (True, "BUILD SUCCESS")
        mock_project.workspace = Path("/test/workspace/project")
        mock_factory.return_value = mock_project
        
        evaluation = JavaCompilationEvaluation(maven_project_factory=mock_factory)
        
        executor = EvaluationExecutor(
            evaluations={
                "java_compilation": {
                    "evaluation": evaluation,
                    "evaluation_schema": JavaCompilationEvaluationConfig,
                }
            }
        )
        
        node = create_evaluation_node(
            evaluation_executor=executor,
            mapper={"java_compilation": map_workflow_to_maven_project},
            execution_mode="all",
        )
        
        # Create state with maven_project_path
        state: MDEAgentState = {
            "workspace_path": Path("/test/workspace"),
            "maven_project_path": Path("/test/workspace/project"),
            "source_model_path": Path("/source"),
            "target_model_path": Path("/target"),
            "group_id": "com.example",
            "artifact_id": "test-artifact",
            "transformation_plan": None,
            "transformation_package_path": "",
            "transformation_class_path": None,
            "bxtool_path": None,
            "required_commands": [],
            "written_files": [],
            "latest_evaluation_runs": [],
        }
        
        result = asyncio.run(node(state))
        
        # Verify state was updated with evaluation results
        self.assertEqual(len(result["latest_evaluation_runs"]), 1)
        run = result["latest_evaluation_runs"][0]
        self.assertEqual(len(run.results), 1)
        self.assertTrue(run.results[0].metadata["success"])
        
        # Verify factory was called with correct project path
        mock_factory.assert_called_once_with(Path("/test/workspace/project"))

    def test_node_mapper_extracts_correct_path(self):
        """Test that mapper extracts maven_project_path from state."""
        mock_factory = MagicMock()
        mock_project = MagicMock(spec=MavenProject)
        mock_project.compile.return_value = (True, "SUCCESS")
        mock_factory.return_value = mock_project
        
        evaluation = JavaCompilationEvaluation(maven_project_factory=mock_factory)
        
        executor = EvaluationExecutor(
            evaluations={
                "java_compilation": {
                    "evaluation": evaluation,
                    "evaluation_schema": JavaCompilationEvaluationConfig,
                }
            }
        )
        
        node = create_evaluation_node(
            evaluation_executor=executor,
            mapper={"java_compilation": map_workflow_to_maven_project},
            execution_mode="all",
        )
        
        nested_project_path = Path("/workspace/parent/child/grandchild")
        state: MDEAgentState = {
            "workspace_path": Path("/workspace"),
            "maven_project_path": nested_project_path,
            "source_model_path": Path("/source"),
            "target_model_path": Path("/target"),
            "group_id": "com.example",
            "artifact_id": "grandchild",
            "transformation_plan": None,
            "transformation_package_path": "",
            "transformation_class_path": None,
            "bxtool_path": None,
            "required_commands": [],
            "written_files": [],
            "latest_evaluation_runs": [],
        }
        
        asyncio.run(node(state))
        
        # Verify the mapper passed the correct nested path
        mock_factory.assert_called_once_with(nested_project_path)

    def test_node_fails_without_maven_project_path(self):
        """Test that node fails gracefully when maven_project_path is missing."""
        evaluation = JavaCompilationEvaluation()
        
        executor = EvaluationExecutor(
            evaluations={
                "java_compilation": {
                    "evaluation": evaluation,
                    "evaluation_schema": JavaCompilationEvaluationConfig,
                }
            }
        )
        
        node = create_evaluation_node(
            evaluation_executor=executor,
            mapper={"java_compilation": map_workflow_to_maven_project},
            execution_mode="all",
        )
        
        # State without maven_project_path
        state: MDEAgentState = {
            "workspace_path": Path("/workspace"),
            "source_model_path": Path("/source"),
            "target_model_path": Path("/target"),
            "group_id": "com.example",
            "artifact_id": "test",
            "transformation_plan": None,
            "transformation_package_path": "",
            "transformation_class_path": None,
            "bxtool_path": None,
            "required_commands": [],
            "written_files": [],
            "latest_evaluation_runs": [],
        }
        
        with self.assertRaises(KeyError):
            asyncio.run(node(state))


class TestJavaCompilationWithRealMavenProject(TestCase):
    """Test JavaCompilationEvaluation with a real Maven project structure."""

    @patch("mdeagent.preparation.maven.subprocess.run")
    def test_end_to_end_with_real_project_structure(self, mock_subprocess):
        """Test full flow with realistic Maven project setup."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="BUILD SUCCESS",
            stderr=""
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            project_path = workspace / "transformation-module"
            project_path.mkdir()
            
            # Create minimal pom.xml
            pom_path = project_path / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>transformation-module</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            # Use real MavenProject.load as factory
            def factory(path: Path) -> MavenProject:
                return MavenProject.load(path)
            
            evaluation = JavaCompilationEvaluation(maven_project_factory=factory)
            
            executor = EvaluationExecutor(
                evaluations={
                    "java_compilation": {
                        "evaluation": evaluation,
                        "evaluation_schema": JavaCompilationEvaluationConfig,
                    }
                }
            )
            
            node = create_evaluation_node(
                evaluation_executor=executor,
                mapper={"java_compilation": map_workflow_to_maven_project},
                execution_mode="all",
            )
            
            state: MDEAgentState = {
                "workspace_path": workspace,
                "maven_project_path": project_path,
                "source_model_path": Path("/source"),
                "target_model_path": Path("/target"),
                "group_id": "com.example",
                "artifact_id": "transformation-module",
                "transformation_plan": None,
                "transformation_package_path": "",
                "transformation_class_path": None,
                "bxtool_path": None,
                "required_commands": [],
                "written_files": [],
                "latest_evaluation_runs": [],
            }
            
            result = asyncio.run(node(state))
            
            # Verify successful compilation
            self.assertEqual(len(result["latest_evaluation_runs"]), 1)
            run = result["latest_evaluation_runs"][0]
            self.assertEqual(len(run.results), 1)
            self.assertTrue(run.results[0].metadata["success"])
