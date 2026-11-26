# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Distinguish a broken GPU host from a CPU host using observed devices rather than declared hardware."""

from __future__ import annotations

from pathlib import Path

import pytest

from eyvan.worker.runtime.host_accelerators import observe_nvidia_devices


def _dev(tmp_path: Path, *names: str) -> Path:
    root = tmp_path / "dev"
    root.mkdir()
    for name in names:
        (root / name).touch()
    return root


def test_a_working_gpu_host_reports_its_device(tmp_path: Path) -> None:
    """The T4 cell, verbatim: `nvidia-smi` reported `Tesla T4, 580.173.02`."""

    observed = observe_nvidia_devices(
        _dev(
            tmp_path,
            "nvidia0",
            "nvidiactl",
            "nvidia-modeset",
            "nvidia-uvm",
            "nvidia-uvm-tools",
            "nvidia-nvswitchctl",
        )
    )

    assert observed.devices == 1
    assert observed.driver_present_but_no_device is False


def test_a_driver_that_cannot_drive_this_gpu_is_not_mistaken_for_a_cpu_host(tmp_path: Path) -> None:
    """The P100 cell, verbatim, and the whole reason this module exists.

    The GPU was attached and `lspci` saw it; the 580 driver dropped Pascal, so no
    module bound it and only the control node appeared. The workload ran there
    and **succeeded**. Nothing the plane could see distinguished this host from a
    healthy one, and a rank on it would have been handed
    ``CUDA_VISIBLE_DEVICES=0`` naming a device node that does not exist.
    """

    observed = observe_nvidia_devices(_dev(tmp_path, "nvidiactl"))

    assert observed.devices == 0
    assert observed.driver_present_but_no_device is True


def test_a_cpu_host_is_not_reported_as_a_broken_gpu_host(tmp_path: Path) -> None:
    """The distinction that keeps this from failing every ordinary cell.

    A CPU host has no NVIDIA nodes and nothing is wrong with it. Conflating the
    two would turn a probe meant to catch a broken accelerator into one that
    condemns the whole fleet -- the failure mode `docs/proofs.md` already records
    for a criterion that fails a correct configuration.
    """

    observed = observe_nvidia_devices(_dev(tmp_path, "null", "zero"))

    assert observed.devices == 0
    assert observed.control_node_present is False
    assert observed.driver_present_but_no_device is False


def test_several_devices_are_counted(tmp_path: Path) -> None:
    """A multi-GPU host, so the count is a count and not a boolean."""

    observed = observe_nvidia_devices(_dev(tmp_path, "nvidia0", "nvidia1", "nvidia2", "nvidiactl"))

    assert observed.devices == 3


@pytest.mark.parametrize("name", ["nvidia-uvm", "nvidiactl", "nvidia-modeset", "nvidiafoo", "nvidia"])
def test_only_numbered_nodes_count_as_devices(tmp_path: Path, name: str) -> None:
    """The control and helper nodes are exactly what the broken host still had.

    Counting any of them would report the P100 cell as having a GPU, which is the
    reading this module was written to prevent.
    """

    assert observe_nvidia_devices(_dev(tmp_path, name)).devices == 0


def test_a_missing_dev_directory_is_reported_rather_than_raised(tmp_path: Path) -> None:
    """A probe that takes down the worker is worse than one that saw nothing."""

    observed = observe_nvidia_devices(tmp_path / "absent")

    assert observed.devices == 0
    assert observed.control_node_present is False
