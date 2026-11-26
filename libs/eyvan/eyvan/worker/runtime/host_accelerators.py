# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What a host's accelerators actually are, as opposed to what it was sold as.

Everything else in this package describes a cell from its declaration: the
hardware catalog says what a shape carries, ``CellAccelerator`` reports what the
catalog said, and ``GpuRankLayout`` hands a rank a ``CUDA_VISIBLE_DEVICES`` built
from that description. None of it looks at the machine.

An attached GPU is not necessarily usable. An incompatible driver can create
``/dev/nvidiactl`` without binding a GPU or creating ``/dev/nvidia0``, while a
callable still reports success. Only host-side probing distinguishes that case
from a usable accelerator.

Stdlib only, and no subprocess: this runs on the worker path, and shelling out to
``nvidia-smi`` would make the answer depend on a binary the failing case is
precisely the one that cannot run.

**Nothing calls this yet.** Wiring it belongs in ``_with_execution_readiness``
in ``eyvan.worker.agent.service``, which already downgrades a cell to
``EXECUTION_CONNECTING`` when a readiness probe refuses.

Integration must gate strictly on the GPU accelerator kind because the readiness
path also serves TPU cells. Qualify it against both usable and incompatible-driver
GPU hosts before treating this probe as an enforced readiness check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: ``/dev/nvidia0``, ``/dev/nvidia1``... The numbered nodes are the ones a
#: process opens. ``nvidiactl``, ``nvidia-uvm`` and friends are deliberately not
#: matched: they exist on a host whose driver loaded but bound no GPU, which is
#: the exact case this module exists to tell apart.
_DEVICE_NODE = re.compile(r"^nvidia(\d+)$")


@dataclass(frozen=True, slots=True)
class ObservedNvidiaDevices:
    """Numbered NVIDIA device nodes present on this host, and the control node."""

    devices: int
    control_node_present: bool

    @property
    def driver_present_but_no_device(self) -> bool:
        """Whether a driver is installed that cannot drive this host's GPU.

        The P100 signature. Worth separating from "no GPU here at all" because
        the two call for opposite responses: this one is a cell that should never
        have accepted GPU work, while a CPU host with no NVIDIA nodes is simply a
        CPU host and nothing is wrong with it.
        """

        return self.control_node_present and self.devices == 0


def observe_nvidia_devices(dev_root: Path | str = Path("/dev")) -> ObservedNvidiaDevices:
    """Return the NVIDIA device nodes this host exposes.

    A missing or unreadable ``/dev`` is reported as nothing observed rather than
    raising: this is a probe, and a probe that takes down the caller is worse
    than one that says it saw nothing.
    """

    root = Path(dev_root)
    try:
        names = [entry.name for entry in root.iterdir()]
    except OSError:
        return ObservedNvidiaDevices(devices=0, control_node_present=False)
    return ObservedNvidiaDevices(
        devices=sum(1 for name in names if _DEVICE_NODE.match(name)),
        control_node_present="nvidiactl" in names,
    )


__all__ = ("ObservedNvidiaDevices", "observe_nvidia_devices")
