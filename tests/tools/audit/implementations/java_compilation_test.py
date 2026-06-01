import asyncio

from unittest import TestCase
from unittest.mock import patch

from src.tools.audit.implementations.java_compilation import JavaCompilationAudit


JAVAC_ERROR_OUTPUT = """
./.bx-agent-workspace/test/Family.java:2: Fehler: <ID> erwartet
    ublic static void main(String[] args) {
         ^
./.bx-agent-workspace/test/Family.java:6: Fehler: ';' erwartet
    String getName() 
                    ^
./.bx-agent-workspace/test/Family.java:10: Fehler: class, interface, enum oder record erwartet
}
^
3 Fehler
"""

JAVAC_SUCCESS_OUTPUT = ""


class TestJavaCompilation(TestCase):
    
    @patch("shutil.which")
    def test_setup__fail_if_javac_not_found(self, mock_which):
        """
        Test that the setup method fails if javac is not installed on the system.
        """
        
        self.assertTrue(
            hasattr(JavaCompilationAudit, "setup"),
            "JavaCompilationAudit should have a 'setup' method.",
        )

        mock_which.return_value = None
        java_compilation_audit = JavaCompilationAudit(files=[])

        with self.assertRaises(RuntimeError, msg="JavaCompilationAudit's 'setup' method should raise RuntimeError if javac is not installed on the system."):
            asyncio.run(java_compilation_audit.setup())


    def test_execute__method_defined(self):
        self.assertTrue(
            hasattr(JavaCompilationAudit, "run"),
            "JavaCompilationAudit should have a 'run' method.",
        )
