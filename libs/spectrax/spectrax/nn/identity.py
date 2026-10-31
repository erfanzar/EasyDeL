# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Identity (pass-through) :class:`~spectrax.Module`.

A trivial layer that returns its input unchanged. Useful as a
structural placeholder where a :class:`~spectrax.Module` is required by
a container or by an architectural choice that may toggle between a
real sub-module and a no-op.
"""

from __future__ import annotations

from typing import TypeVar

from ..core.module import Module

T = TypeVar("T")


class Identity(Module):
    """Pass-through layer: returns its first argument unchanged.

    The canonical use is as a placeholder slot in a
    :class:`~spectrax.nn.Sequential` chain (e.g. to keep position
    indices stable when ablating a sub-module) or as a no-op branch in
    architectures that conditionally insert a real layer. Carries no
    parameters or buffers.
    """

    def __init__(self) -> None:
        """Initialize the layer.

        Calls :meth:`~spectrax.Module.__init__` so the instance is a
        registered Spectrax module; allocates no parameters or
        buffers.
        """
        super().__init__()

    def forward(self, x: T, **_: object) -> T:
        """Return ``x`` unmodified.

        Args:
            x: Any value. Returned untouched and untyped — pytrees,
                arrays, dataclasses all pass through identically.
            **_: Ignored. Accepted so the layer can be placed in
                containers that thread auxiliary kwargs through every
                child without having to special-case the no-op slots.

        Returns:
            ``x`` exactly as received (no copy, no dtype change).
        """
        return x
