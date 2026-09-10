import asyncio
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mdeagent.evaluation.implementations.java_compilation import (
    JavaCompilationEvaluation,
    JavaCompilationEvaluationConfig,
    parse_mvn_compile_output,
)
from mdeagent.evaluation.types import EvaluationResult
from mdeagent.preparation.maven import MavenProject
from mdeagent.preparation.pom import Pom


class TestJavaCompilationEvaluationConfig(TestCase):
    """Test cases for JavaCompilationEvaluationConfig."""

    def test_config_requires_project_path(self):
        """Test that config requires project_path parameter."""
        with self.assertRaises(ValueError):
            JavaCompilationEvaluationConfig()

        config = JavaCompilationEvaluationConfig(project_path=Path("/test"))
        self.assertEqual(config.project_path, Path("/test"))


class TestJavaCompilationEvaluationInitialization(TestCase):
    """Test cases for JavaCompilationEvaluation initialization."""

    def test_init_with_default_factory(self):
        """Test initialization with default MavenProject.load factory."""
        evaluation = JavaCompilationEvaluation()
        self.assertIsNotNone(evaluation._maven_project_factory)
        self.assertIsNone(evaluation._maven_project)

    def test_init_with_custom_factory(self):
        """Test initialization with custom factory method."""
        custom_factory = MagicMock()
        evaluation = JavaCompilationEvaluation(maven_project_factory=custom_factory)
        self.assertEqual(evaluation._maven_project_factory, custom_factory)
        self.assertIsNone(evaluation._maven_project)


class TestJavaCompilationEvaluationSetup(TestCase):
    """Test cases for JavaCompilationEvaluation.setup() method."""

    def test_setup_loads_maven_project(self):
        """Test that setup loads the Maven project using the factory."""
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

            mock_project = MagicMock(spec=MavenProject)
            mock_factory = MagicMock(return_value=mock_project)

            evaluation = JavaCompilationEvaluation(maven_project_factory=mock_factory)
            asyncio.run(evaluation.setup(project_path=workspace))

            mock_factory.assert_called_once_with(workspace)
            self.assertEqual(evaluation._maven_project, mock_project)

    def test_setup_raises_on_factory_error(self):
        """Test that setup raises RuntimeError when factory fails."""
        mock_factory = MagicMock(side_effect=FileNotFoundError("pom.xml not found"))
        evaluation = JavaCompilationEvaluation(maven_project_factory=mock_factory)

        with self.assertRaises(RuntimeError) as context:
            asyncio.run(evaluation.setup(project_path=Path("/nonexistent")))

        self.assertIn("Failed to load Maven project", str(context.exception))
        self.assertIn("pom.xml not found", str(context.exception))


class TestJavaCompilationEvaluationRun(TestCase):
    """Test cases for JavaCompilationEvaluation.run() method."""

    def test_run_method_defined(self):
        """Test that run method is defined with correct signature."""
        self.assertTrue(
            hasattr(JavaCompilationEvaluation, "run"),
            "JavaCompilationEvaluation should have a 'run' method.",
        )

    def test_run_successful_compilation(self):
        """Test run method with successful compilation."""
        mock_project = MagicMock(spec=MavenProject)
        mock_project.compile.return_value = (True, "BUILD SUCCESS")
        mock_project.workspace = Path("/test/workspace")

        evaluation = JavaCompilationEvaluation()
        evaluation._maven_project = mock_project

        results, errors = asyncio.run(evaluation.run())

        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 0)
        self.assertEqual(results[0].content, "Maven project compiled successfully.")
        self.assertEqual(results[0].metadata["success"], True)

    def test_run_failed_compilation_with_errors(self):
        """Test run method with failed compilation and parseable errors."""
        mock_project = MagicMock(spec=MavenProject)
        error_output = """
[ERROR] /workspace/src/main/java/com/example/Test.java:[10:5] cannot find symbol
  symbol:   class UnknownClass
  location: class Test
[ERROR] /workspace/src/main/java/com/example/Test.java:[15:10] ';' expected
"""
        mock_project.compile.return_value = (False, error_output)
        mock_project.workspace = Path("/workspace")

        evaluation = JavaCompilationEvaluation()
        evaluation._maven_project = mock_project

        results, errors = asyncio.run(evaluation.run())

        self.assertEqual(len(errors), 0)
        self.assertGreater(len(results), 0)
        
        # Check that at least some results indicate failure
        failed_results = [r for r in results if not r.metadata.get("success", True)]
        self.assertGreater(len(failed_results), 0)

    def test_run_exception_during_compilation(self):
        """Test run method when compilation raises an exception."""
        mock_project = MagicMock(spec=MavenProject)
        mock_project.compile.side_effect = Exception("Compilation failed")
        mock_project.workspace = Path("/test/workspace")

        evaluation = JavaCompilationEvaluation()
        evaluation._maven_project = mock_project

        results, errors = asyncio.run(evaluation.run())

        self.assertEqual(len(results), 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("Compilation failed", errors[0].message)
        self.assertEqual(errors[0].type, "Exception")

    def test_run_without_setup_uses_kwargs(self):
        """Test that run can work without prior setup by using kwargs."""
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

            mock_project = MagicMock(spec=MavenProject)
            mock_project.compile.return_value = (True, "BUILD SUCCESS")
            mock_project.workspace = workspace

            mock_factory = MagicMock(return_value=mock_project)
            evaluation = JavaCompilationEvaluation(maven_project_factory=mock_factory)

            results, errors = asyncio.run(evaluation.run(project_path=workspace))

            mock_factory.assert_called_once_with(workspace)
            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 0)

    def test_run_missing_project_path_parameter(self):
        """Test that run raises ValueError when project_path is missing."""
        evaluation = JavaCompilationEvaluation()

        with self.assertRaises(ValueError):
            asyncio.run(evaluation.run())


class TestParseMvnCompileOutput(TestCase):
    """Test cases for parse_mvn_compile_output function."""

    def test_parse_no_errors(self):
        """Test parsing output with no errors."""
        output = """
[INFO] Compiling 1 source file
[INFO] BUILD SUCCESS
"""
        results = parse_mvn_compile_output(output)
        self.assertEqual(len(results), 0)

    def test_parse_maven_error_format(self):
        """Test parsing Maven-style error format with [ERROR] prefix."""
        output = """
[INFO] Compiling 1 source file
[ERROR] /workspace/src/main/java/com/example/Test.java:[10:5] cannot find symbol
  symbol:   class UnknownClass
  location: class Test
[ERROR] /workspace/src/main/java/com/example/Test.java:[15:10] ';' expected
[INFO] BUILD FAILURE
"""
        results = parse_mvn_compile_output(output)

        self.assertEqual(len(results), 2)
        
        self.assertEqual(
            results[0].content,
            "cannot find symbol",
        )
        self.assertEqual(results[0].metadata["success"], False)
        self.assertEqual(
            results[0].metadata["file"],
            "/workspace/src/main/java/com/example/Test.java",
        )
        self.assertEqual(results[0].metadata["line"], 10)
        self.assertEqual(results[0].metadata["column"], 5)

        self.assertEqual(
            results[1].content,
            "';' expected",
        )
        self.assertEqual(results[1].metadata["line"], 15)
        self.assertEqual(results[1].metadata["column"], 10)

    def test_parse_javac_line_format(self):
        """Test parsing javac-style error format with line number only."""
        output = """
/workspace/src/main/java/com/example/Test.java:10: error: cannot find symbol
    UnknownClass obj = new UnknownClass();
    ^
  symbol:   class UnknownClass
  location: class Test
1 error
"""
        results = parse_mvn_compile_output(output)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].content,
            "error: cannot find symbol",
        )
        self.assertEqual(results[0].metadata["success"], False)
        self.assertEqual(
            results[0].metadata["file"],
            "/workspace/src/main/java/com/example/Test.java",
        )
        self.assertEqual(results[0].metadata["line"], 10)
        self.assertNotIn("column", results[0].metadata)

    def test_parse_javac_column_format(self):
        """Test parsing javac-style error format with line and column."""
        output = """
/workspace/src/main/java/com/example/Test.java:10:5: error: cannot find symbol
    UnknownClass obj = new UnknownClass();
    ^
1 error
"""
        results = parse_mvn_compile_output(output)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].content,
            "error: cannot find symbol",
        )
        self.assertEqual(results[0].metadata["line"], 10)
        self.assertEqual(results[0].metadata["column"], 5)

    def test_parse_german_error_messages(self):
        """Test parsing German error messages from javac."""
        output = """
./workspace/test/Family.java:2: Fehler: <ID> erwartet
    ublic static void main(String[] args) {
         ^
./workspace/test/Family.java:6: Fehler: ';' erwartet
    String getName() 
                    ^
3 Fehler
"""
        results = parse_mvn_compile_output(output)

        self.assertEqual(len(results), 2)
        self.assertIn("<ID> erwartet", results[0].content)
        self.assertIn("';' erwartet", results[1].content)
        self.assertEqual(results[0].metadata["line"], 2)
        self.assertEqual(results[1].metadata["line"], 6)

    def test_parse_symbol_not_found_errors(self):
        """Test parsing 'Symbol nicht gefunden' errors."""
        output = """
./workspace/transformation/Family.java:29: Fehler: Symbol nicht gefunden
    FamilyMember getFather();
    ^
  Symbol: Klasse FamilyMember
  Ort: Schnittstelle Family
./workspace/transformation/Family.java:36: Fehler: Symbol nicht gefunden
    void setFather(FamilyMember father);
                   ^
  Symbol: Klasse FamilyMember
  Ort: Schnittstelle Family
6 Fehler
"""
        results = parse_mvn_compile_output(output)

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("Symbol nicht gefunden", result.content)
            self.assertEqual(result.metadata["success"], False)

    def test_parse_multiple_errors_same_file(self):
        """Test parsing multiple errors in the same file."""
        output = """
[ERROR] /workspace/Test.java:[5:1] class Test is public, should be declared in a file named Test.java
[ERROR] /workspace/Test.java:[10:15] cannot find symbol
[ERROR] /workspace/Test.java:[15:20] incompatible types: int cannot be converted to String
"""
        results = parse_mvn_compile_output(output)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(
                result.metadata["file"], "/workspace/Test.java"
            )
            self.assertEqual(result.metadata["success"], False)

    def test_parse_empty_output(self):
        """Test parsing empty output."""
        results = parse_mvn_compile_output("")
        self.assertEqual(len(results), 0)

    def test_parse_whitespace_only_output(self):
        """Test parsing whitespace-only output."""
        results = parse_mvn_compile_output("   \n\n   ")
        self.assertEqual(len(results), 0)


class TestJavaCompilationEvaluationIntegration(TestCase):
    """Integration tests for JavaCompilationEvaluation with real MavenProject."""

    def test_end_to_end_with_mock_project(self):
        """Test complete flow with mocked MavenProject."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create mock project
            mock_project = MagicMock(spec=MavenProject)
            mock_project.workspace = workspace
            mock_project.compile.return_value = (True, "BUILD SUCCESS")

            mock_factory = MagicMock(return_value=mock_project)
            evaluation = JavaCompilationEvaluation(maven_project_factory=mock_factory)

            # Setup
            asyncio.run(evaluation.setup(project_path=workspace))

            # Run
            results, errors = asyncio.run(evaluation.run())

            # Verify
            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 0)
            self.assertTrue(results[0].metadata["success"])

    def test_factory_called_only_in_setup(self):
        """Test that factory is called during setup, not during each run."""
        workspace = Path("/test/workspace")
        mock_project = MagicMock(spec=MavenProject)
        mock_project.workspace = workspace
        mock_project.compile.return_value = (True, "BUILD SUCCESS")
        
        mock_factory = MagicMock(return_value=mock_project)
        evaluation = JavaCompilationEvaluation(maven_project_factory=mock_factory)

        # Setup should call factory
        asyncio.run(evaluation.setup(project_path=workspace))
        self.assertEqual(mock_factory.call_count, 1)

        # Multiple runs should not call factory again
        asyncio.run(evaluation.run())
        asyncio.run(evaluation.run())
        self.assertEqual(mock_factory.call_count, 1)
