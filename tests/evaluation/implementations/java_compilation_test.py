import asyncio
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mdeagent.evaluation.implementations.java_compilation import (
    JavaCompilationEvaluation,
    JavaCompilationSchema,
    parse_mvn_compile_output,
)
from mdeagent.preparation.maven import MavenProject


class TestJavaCompilationEvaluationConfig(TestCase):
    """Test cases for JavaCompilationEvaluationConfig."""

    def test_config_requires_project_path(self):
        """Test that config requires project_path parameter."""
        from pydantic import ValidationError
        
        with self.assertRaises(ValidationError):
            JavaCompilationSchema()

        config = JavaCompilationSchema(project_path=Path("/test"))
        self.assertEqual(config.project_path, Path("/test"))


class TestJavaCompilationEvaluationSetup(TestCase):
    """Test cases for JavaCompilationEvaluation.setup() method."""

    def test_setup_is_noop(self):
        """Test that setup() does nothing (no-op)."""
        evaluation = JavaCompilationEvaluation()
        # Should not raise
        asyncio.run(evaluation.setup())


class TestJavaCompilationEvaluationRun(TestCase):
    """Test cases for JavaCompilationEvaluation.run() method."""

    def test_run_method_defined(self):
        """Test that run method is defined with correct signature."""
        self.assertTrue(
            hasattr(JavaCompilationEvaluation, "run"),
            "JavaCompilationEvaluation should have a 'run' method.",
        )

    def test_run_with_project_path_in_run_kwargs(self):
        """Test run method with project_path passed to run()."""
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

            with patch.object(MavenProject, 'load', return_value=MagicMock(compile=lambda: (True, "BUILD SUCCESS"), workspace=workspace)) as mock_load:
                evaluation = JavaCompilationEvaluation()
                results, errors = asyncio.run(evaluation.run(project_path=workspace))

                mock_load.assert_called_once_with(workspace)
                self.assertEqual(len(results), 1)
                self.assertEqual(len(errors), 0)
                self.assertEqual(results[0].content, "Maven project compiled successfully.")

    def test_run_without_project_path_raises(self):
        """Test that run raises ValueError when project_path is missing."""
        evaluation = JavaCompilationEvaluation()

        with self.assertRaises(ValueError) as context:
            asyncio.run(evaluation.run())

        self.assertIn("project_path", str(context.exception).lower())

    def test_run_successful_compilation(self):
        """Test run method with successful compilation."""
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
            with patch.object(MavenProject, 'load', return_value=mock_project):
                results, errors = asyncio.run(evaluation.run(project_path=workspace))

            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 0)
            self.assertEqual(results[0].content, "Maven project compiled successfully.")
            self.assertEqual(results[0].metadata["success"], True)

    def test_run_failed_compilation_with_errors(self):
        """Test run method with failed compilation and parseable errors."""
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

            error_output = """
[ERROR] /workspace/src/main/java/com/example/Test.java:[10:5] cannot find symbol
  symbol:   class UnknownClass
  location: class Test
[ERROR] /workspace/src/main/java/com/example/Test.java:[15:10] ';' expected
"""
            mock_project = MagicMock()
            mock_project.compile.return_value = (False, error_output)
            mock_project.workspace = workspace

            evaluation = JavaCompilationEvaluation()
            with patch.object(MavenProject, 'load', return_value=mock_project):
                results, errors = asyncio.run(evaluation.run(project_path=workspace))

            self.assertEqual(len(errors), 0)
            self.assertGreater(len(results), 0)
            
            # Check that at least some results indicate failure
            failed_results = [r for r in results if not r.metadata.get("success", True)]
            self.assertGreater(len(failed_results), 0)

    def test_run_exception_during_compilation(self):
        """Test run method when compilation raises an exception."""
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
            mock_project.compile.side_effect = Exception("Compilation failed")
            mock_project.workspace = workspace

            evaluation = JavaCompilationEvaluation()
            with patch.object(MavenProject, 'load', return_value=mock_project):
                results, errors = asyncio.run(evaluation.run(project_path=workspace))

            self.assertEqual(len(results), 0)
            self.assertEqual(len(errors), 1)
            self.assertIn("Compilation failed", errors[0].message)
            self.assertEqual(errors[0].type, "Exception")

    def test_run_fails_to_load_maven_project(self):
        """Test run method when MavenProject.load fails."""
        with patch.object(MavenProject, 'load', side_effect=FileNotFoundError("pom.xml not found")):
            evaluation = JavaCompilationEvaluation()
            results, errors = asyncio.run(evaluation.run(project_path=Path("/nonexistent")))

            self.assertEqual(len(results), 0)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].type, "MavenProjectLoadError")
            self.assertIn("pom.xml not found", errors[0].message)


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
    """Integration tests for JavaCompilationEvaluation."""

    def test_setup_is_noop(self):
        """Test that setup() does nothing."""
        evaluation = JavaCompilationEvaluation()
        # Setup should be no-op and not raise
        asyncio.run(evaluation.setup())

    def test_project_loaded_during_run(self):
        """Test that MavenProject.load is called during run()."""
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
            
            # Setup should not call load
            asyncio.run(evaluation.setup())
            
            # Run should call load with project_path
            with patch.object(MavenProject, 'load', return_value=mock_project) as mock_load:
                results, errors = asyncio.run(evaluation.run(project_path=workspace))
                mock_load.assert_called_once_with(workspace)
            
            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 0)

    def test_complete_flow_with_mock(self):
        """Test complete flow: constructor -> setup -> run."""
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

            # 1. Constructor - creates evaluation without loading
            
            # 2. Setup - still no load (no-op)
            asyncio.run(evaluation.setup())
            
            # 3. Run - load called with project_path, compilation executed
            with patch.object(MavenProject, 'load', return_value=mock_project) as mock_load:
                results, errors = asyncio.run(evaluation.run(project_path=workspace))
                mock_load.assert_called_once()
            
            # 4. Verify results
            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 0)
            self.assertTrue(results[0].metadata["success"])

    def test_project_path_from_run_kwargs(self):
        """Test that project_path can be passed to run() instead of constructor."""
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

            # No project_path in constructor
            evaluation = JavaCompilationEvaluation()

            with patch.object(MavenProject, 'load', return_value=mock_project) as mock_load:
                results, errors = asyncio.run(evaluation.run(project_path=workspace))
                mock_load.assert_called_once_with(workspace)
            
            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 0)
