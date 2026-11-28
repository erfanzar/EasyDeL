# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Introspection tests must not leave allocation tracing enabled for the suite."""

import os
import subprocess
import sys
from pathlib import Path


def test_introspection_tests_restore_allocation_tracking() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONTRACEMALLOC", "PYTEST_ADDOPTS"}}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pytest, tracemalloc; "
            "assert not tracemalloc.is_tracing(); "
            "result = pytest.main(['tests/test_process_introspection.py', '-q']); "
            "assert result == 0; "
            "assert not tracemalloc.is_tracing(), 'introspection tests leaked allocation tracing'",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
