# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared pipeline primitives.

Two unrelated tools used to express stage structure:

* :func:`boundary` is an inline, JVP-aware identity marker. Calling
  ``boundary(x)`` inside a module's ``forward`` declares an intended
  pipeline split point. Today it lowers to identity and is preserved
  in the jaxpr so a future jaxpr-splitting pass can cut the forward
  there.
* :func:`auto_split` (and the lower-level :func:`split_block_stack`)
  walks an existing :class:`~spectrax.core.module.Module` containing a
  ``ModuleList`` of repeated blocks and slices it into ``n_pp``
  per-stage modules without requiring any explicit pipeline
  annotations. Per-block placement can still be steered by setting a
  ``pp_stage`` attribute on the children.
"""

from __future__ import annotations

from .boundary import boundary
from .split import auto_split, split_block_stack

__all__ = [
    "auto_split",
    "boundary",
    "split_block_stack",
]
