import asyncio

from unittest import TestCase

from src.tools.audit.implementations.java_compilation import JavaCompilationAudit


class TestJavaCompilation(TestCase):
    def test_setup__do_nothing(self):
        self.assertTrue(
            hasattr(JavaCompilationAudit, "setup"),
            "JavaCompilationAudit should have a 'setup' method.",
        )

        java_compilation_audit = JavaCompilationAudit(files=[])

        self.assertIsNone(
            asyncio.run(java_compilation_audit.setup()),
            "JavaCompilationAudit's 'setup' method should return None.",
        )

    def test_execute__method_defined(self):
        self.assertTrue(
            hasattr(JavaCompilationAudit, "run"),
            "JavaCompilationAudit should have a 'run' method.",
        )
