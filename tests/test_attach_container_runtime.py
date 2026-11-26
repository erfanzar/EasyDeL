# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""An attached TPU given a container image can actually run TPU jobs in it.

``eyvan gcp connect --container-image`` (and ``runtime_container_image`` in the
defaults profile) built the container block without ``privileged``, while the
OCI executor refuses every TPU workload launched unprivileged. Generated fleets
already state ``privileged: true`` (eyvan/capacity/autoscale/compiler.py); the
attach path now states the same container block.
"""

from __future__ import annotations

from pathlib import Path

from eyvan.capacity.autoscale.models import AutoscaleDefaults
from eyvan.sdk.commands.gcp_host_access import resolve_attach_runtime
from eyvan.worker.runtime.container import DockerContainerExecutor
from tests.test_container_executor import FakeDockerSystemd, _context, _spec

IMAGE = "us-east5-docker.pkg.dev/project-a/eyvan/runtime@sha256:" + "0" * 64


def _executor(tmp_path: Path, runtime) -> tuple[DockerContainerExecutor, FakeDockerSystemd]:
    daemon = FakeDockerSystemd(images={IMAGE})
    executor = DockerContainerExecutor(
        tmp_path,
        container=runtime.container,
        command_runner=daemon,
        boot_id_reader=lambda: "boot-a",
    )
    return executor, daemon


def test_an_attach_image_flag_yields_a_container_that_runs_tpu_jobs(tmp_path: Path) -> None:
    runtime = resolve_attach_runtime(defaults=None, python="3.13", container_image=IMAGE)

    assert runtime.container is not None
    assert runtime.container.image == IMAGE
    assert runtime.container.privileged is True
    assert runtime.container.working_dir == "/var/lib/eyvan/workload"

    executor, daemon = _executor(tmp_path, runtime)
    executor.submit(_spec(resources={"tpu": 4}), _context())
    launch = next(command for command in daemon.commands if command[0] == "systemd-run")
    assert "--privileged" in launch[launch.index("docker") :]


def test_the_defaults_profile_image_gets_the_same_container_block(tmp_path: Path) -> None:
    defaults = AutoscaleDefaults(project="project-a", runtime_container_image=IMAGE)
    runtime = resolve_attach_runtime(defaults=defaults, python="3.13")

    assert runtime.container is not None and runtime.container.privileged is True


def test_no_image_means_no_container(tmp_path: Path) -> None:
    assert resolve_attach_runtime(defaults=None, python="3.13").container is None
