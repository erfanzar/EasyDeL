# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prevent hosted validation failures before release jobs can start."""

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/eyvan-oci-release.yaml"


@pytest.mark.parametrize("job", ["pull-request-qualification", "release"])
def test_runner_context_is_not_used_in_job_environment(job: str) -> None:
    """GitHub does not expose runner context in jobs.<job_id>.env."""
    configuration = yaml.safe_load(WORKFLOW.read_text())
    environment = configuration["jobs"][job].get("env", {})
    assert all("runner." not in str(value) for value in environment.values())


@pytest.mark.parametrize("job", ["pull-request-qualification", "release"])
def test_tool_directory_is_exported_from_the_install_step(job: str) -> None:
    """The install step resolves its runner path and retains it for scanners."""
    configuration = yaml.safe_load(WORKFLOW.read_text())
    steps = configuration["jobs"][job]["steps"]
    install = next(step for step in steps if "oci/install_locked_tools.py" in step.get("run", ""))
    assert install["env"]["TOOL_DIR"].startswith("${{ runner.temp }}/")
    assert 'echo "TOOL_DIR=${TOOL_DIR}" >> "${GITHUB_ENV}"' in install["run"]
