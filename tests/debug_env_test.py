import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


from unittest import TestCase


class TestDebugEnv(TestCase):
    def setUp(self):
        pass

    def test_environment_debug(self):
        """Debug-Test zur Umgebung"""
        logger.info(f"\n--- Python Interpreter ---")
        logger.info(f"sys.executable: {sys.executable}")

        logger.info(f"\n--- PATH ---")
        logger.info(f"os.environ['PATH']: {os.environ.get('PATH')}")

        logger.info(f"\n--- shutil.which ---")
        logger.info(f"shutil.which('mvn'): {shutil.which('mvn')}")

        logger.info(f"\n--- which command ---")
        result = subprocess.run(["which", "mvn"], capture_output=True, text=True)
        logger.info(f"which mvn stdout: {result.stdout}")
        logger.info(f"which mvn stderr: {result.stderr}")

        logger.info(f"\n--- Venv info ---")
        logger.info(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")
        logger.info(f"sys.prefix: {sys.prefix}")

        self.assertFalse(
            shutil.which("mvn") is None,
            "Maven (mvn) should be found in PATH for the tests to run.",
        )
