"""Pytest configuration and custom fixtures."""

import os
import subprocess
import sys

import pytest
import logging

logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Lädt die volle Shell-Umgebung für pytest"""

    if sys.platform == "darwin":  # macOS
        try:
            # Lade die komplette Shell-Umgebung (aus .zshrc, .bashrc, etc.)
            result = subprocess.run(
                ["/bin/zsh", "-i", "-l", "-c", "echo $PATH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            if result.returncode == 0 and result.stdout.strip():
                # Setze den kompletten PATH
                os.environ["PATH"] = result.stdout.strip()

                # Setze auch die Venv-Variable
                venv_path = os.path.join(os.path.dirname(__file__), ".venv")
                if os.path.exists(venv_path):
                    os.environ["VIRTUAL_ENV"] = venv_path

                logger.info(f"\n✓ pytest PATH aktualisiert")
                logger.info(
                    f"  Maven verfügbar: {subprocess.run(['which', 'mvn'], capture_output=True, text=True, check=False).stdout.strip()}"
                )
        except Exception as e:
            logger.error(f"\n⚠ Fehler beim Laden der Shell-Umgebung: {e}")


def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--enable-langfuse",
        action="store_true",
        default=False,
        help="Enable Langfuse monitoring during tests",
    )
    parser.addoption(
        "--langfuse-public-key",
        action="store",
        default=None,
        help="Langfuse public key (overrides env variable)",
    )
    parser.addoption(
        "--langfuse-secret-key",
        action="store",
        default=None,
        help="Langfuse secret key (overrides env variable)",
    )


@pytest.fixture
def enable_langfuse(request):
    """Fixture to check if Langfuse monitoring is enabled."""
    return request.config.getoption("--enable-langfuse")


@pytest.fixture
def langfuse_credentials(request):
    """Fixture to get Langfuse credentials from command line or env."""
    return {
        "public_key": request.config.getoption("--langfuse-public-key"),
        "secret_key": request.config.getoption("--langfuse-secret-key"),
    }
