"""Pytest configuration and custom fixtures."""

import pytest


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
