# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unified experiment logging for Spectrax.

This subpackage provides a single :class:`~spectrax.loggers.Logger` that can
multiplex scalar, histogram, image, text, and hyper-parameter writes to any
combination of backends:

* **TensorBoard** — via SpectraX's native event-file backend.
* **Weights & Biases** — via ``wandb`` (if installed).
* **Console** — plain ``stdout`` for quick debugging.

All backends are **optional**; missing packages are silently skipped.
In multi-process JAX training, only process ``0`` performs actual I/O.

Example::

    from spectrax.loggers import SummaryWriter

    logger = SummaryWriter(log_dir="./runs/exp-1")
    logger.log_scalar("loss/train", 0.42, step=100)
    logger.log_histogram("weights/l1", params["l1"], step=100)
    logger.close()
"""

from __future__ import annotations

from .base import BaseBackend, Logger
from .console import ConsoleBackend
from .summary_writer import SummaryWriter
from .tensorboard import TensorBoardBackend
from .wandb import WandBBackend

__all__ = [
    "BaseBackend",
    "ConsoleBackend",
    "Logger",
    "SummaryWriter",
    "TensorBoardBackend",
    "WandBBackend",
]
