# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TensorBoard SummaryWriter compatibility wrapper."""

from __future__ import annotations

import os

from .base import Logger
from .tensorboard import TensorBoardBackend


class SummaryWriter(Logger):
    """TensorBoard writer backed by SpectraX's native event-file backend.

    The class intentionally follows the common ``SummaryWriter(log_dir=...)``
    shape while reusing :class:`Logger`'s TensorBoard-compatible methods such
    as ``add_scalar``, ``add_histogram``, ``scalar``, and ``histogram``.
    Extra keyword arguments are accepted for compatibility with TensorBoardX
    and PyTorch callers, but SpectraX does not need them.
    """

    def __init__(
        self,
        log_dir: str | os.PathLike[str] | None = None,
        *,
        auto_flush: bool = True,
        **kwargs: object,
    ):
        log_dir = log_dir or kwargs.pop("logdir", None) or "runs"
        super().__init__([TensorBoardBackend(log_dir)], auto_flush=auto_flush)
