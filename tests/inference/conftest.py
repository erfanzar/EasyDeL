from __future__ import annotations

import os


def pytest_configure() -> None:
    """Keep inference unit tests hermetic on hosts with occupied TPU runtimes."""
    if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cpu"
