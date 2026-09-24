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
    JavaCompilationSchema,
)
from mdeagent.evaluation.node import create_evaluation_node
from mdeagent.mapping.mde_to_maven_project import mde_to_maven_project
from mdeagent.preparation.maven import MavenProject
from mdeagent.state import MDEAgentState


class TestJavaCompilationWithExecutor(TestCase):
    """Test JavaCompilationEvaluation integration with EvaluationExecutor."""

    def test_executor_with_java_compilation(self):
        """Test that EvaluationExecutor can execute JavaCompilationEvaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            pom_path = workspace / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-app</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            mock_project = MagicMock()
            mock_project.compile.return_value = (True, "BUILD SUCCESS")
            mock_project.workspace = workspace
            
            evaluation = JavaCompilationEvaluation()
            
            executor = EvaluationExecutor(
                evaluations={
                    "java_compilation": {
                        "evaluation": evaluation,
                        "evaluation_schema": JavaCompilationSchema,
                    }
                }
            )
            
            # Execute the evaluation
            with patch.object(MavenProject, 'load', return_value=mock_project) as mock_load:
                result = asyncio.run(
                    executor.execute_specific(
                        evaluation_id="java_compilation",
                        input={"project_path": workspace},
                    )
                )
                mock_load.assert_called_once_with(workspace)
            
            self.assertEqual(len(result.results), 1)
            self.assertTrue(result.results[0].metadata["success"])

    def test_executor_with_failed_compilation(self):
        """Test Executor handling failed compilation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            pom_path = workspace / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-app</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            error_output = "[ERROR] /workspace/Test.java:[10:5] cannot find symbol"
            mock_project = MagicMock()
            mock_project.compile.return_value = (False, error_output)
            mock_project.workspace = workspace
            
            evaluation = JavaCompilationEvaluation()
            
            executor = EvaluationExecutor(
                evaluations={
                    "java_compilation": {
                        "evaluation": evaluation,
                        "evaluation_schema": JavaCompilationSchema,
                    }
                }
            )
            
            with patch.object(MavenProject, 'load', return_value=mock_project):
                result = asyncio.run(
                    executor.execute_specific(
                        evaluation_id="java_compilation",
                        input={"project_path": workspace},
                    )
                )
            
            # Should have parsed error results
            self.assertGreater(len(result.results), 0)


class TestJavaCompilationWithMapperAndNode(TestCase):
    """Test JavaCompilationEvaluation with mapper and evaluation node."""

    def test_node_with_maven_project_mapper(self):
        """Test evaluation node uses map_workflow_to_maven_project correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "project"
            project_path.mkdir()
            pom_path = project_path / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-app</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            mock_project = MagicMock()
            mock_project.compile.return_value = (True, "BUILD SUCCESS")
            mock_project.workspace = project_path
            
            evaluation = JavaCompilationEvaluation()
            
            executor = EvaluationExecutor(
                evaluations={
                    "java_compilation": {
                        "evaluation": evaluation,
                        "evaluation_schema": JavaCompilationSchema,
                    }
                }
            )
            
            node = create_evaluation_node(
                evaluation_executor=executor,
                mapper={"java_compilation": mde_to_maven_project},
                execution_mode="all",
            )
            
            # Create state with maven_project_path
            state: MDEAgentState = {
                "workspace_path": Path(temp_dir),
                "maven_project_path": project_path,
                "source_model_path": Path("/source"),
                "target_model_path": Path("/target"),
                "group_id": "com.example",
                "artifact_id": "test-artifact",
                "transformation_plan": None,
                "transformation_package_path": "",
                "transformation_class_path": None,
                "bxtool_path": None,
                "required_tools": [],
                "written_files": [],
                "latest_evaluation_runs": {},
            }
            
            with patch.object(MavenProject, 'load', return_value=mock_project) as mock_load:
                result = asyncio.run(node(state))
                mock_load.assert_called_once_with(project_path)
            
            # Verify state was updated with evaluation results
            self.assertEqual(len(result["latest_evaluation_runs"]), 1)
            run = list(result["latest_evaluation_runs"].values())[0]
            self.assertEqual(len(run.results), 1)
            self.assertTrue(run.results[0].metadata["success"])

    def test_node_mapper_extracts_correct_path(self):
        """Test that mapper extracts maven_project_path from state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_project_path = Path(temp_dir) / "parent" / "child" / "grandchild"
            nested_project_path.mkdir(parents=True)
            pom_path = nested_project_path / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>grandchild</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            mock_project = MagicMock()
            mock_project.compile.return_value = (True, "SUCCESS")
            mock_project.workspace = nested_project_path
            
            evaluation = JavaCompilationEvaluation()
            
            executor = EvaluationExecutor(
                evaluations={
                    "java_compilation": {
                        "evaluation": evaluation,
                        "evaluation_schema": JavaCompilationSchema,
                    }
                }
            )
            
            node = create_evaluation_node(
                evaluation_executor=executor,
                mapper={"java_compilation": mde_to_maven_project},
                execution_mode="all",
            )
            
            state: MDEAgentState = {
                "workspace_path": Path(temp_dir),
                "maven_project_path": nested_project_path,
                "source_model_path": Path("/source"),
                "target_model_path": Path("/target"),
                "group_id": "com.example",
                "artifact_id": "grandchild",
                "transformation_plan": None,
                "transformation_package_path": "",
                "transformation_class_path": None,
                "bxtool_path": None,
                "required_tools": [],
                "written_files": [],
                "latest_evaluation_runs": {},
            }
            
            with patch.object(MavenProject, 'load', return_value=mock_project) as mock_load:
                asyncio.run(node(state))
                mock_load.assert_called_once_with(nested_project_path)

    def test_node_fails_without_maven_project_path(self):
        """Test that node fails gracefully when maven_project_path is missing."""
        evaluation = JavaCompilationEvaluation()
        
        executor = EvaluationExecutor(
            evaluations={
                "java_compilation": {
                    "evaluation": evaluation,
                    "evaluation_schema": JavaCompilationSchema,
                }
            }
        )
        
        node = create_evaluation_node(
            evaluation_executor=executor,
            mapper={"java_compilation": mde_to_maven_project},
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
            "required_tools": [],
            "written_files": [],
            "latest_evaluation_runs": {},
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
            
            evaluation = JavaCompilationEvaluation()
            
            executor = EvaluationExecutor(
                evaluations={
                    "java_compilation": {
                        "evaluation": evaluation,
                        "evaluation_schema": JavaCompilationSchema,
                    }
                }
            )
            
            node = create_evaluation_node(
                evaluation_executor=executor,
                mapper={"java_compilation": mde_to_maven_project},
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
                "required_tools": [],
                "written_files": [],
                "latest_evaluation_runs": {},
            }
            
            result = asyncio.run(node(state))
            
            # Verify successful compilation
            self.assertEqual(len(result["latest_evaluation_runs"]), 1)
            run = list(result["latest_evaluation_runs"].values())[0]
            self.assertEqual(len(run.results), 1)
            self.assertTrue(run.results[0].metadata["success"])
