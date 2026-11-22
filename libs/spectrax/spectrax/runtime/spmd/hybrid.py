# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Removed legacy SPMD/host-jit hybrid path.

The previous helper mixed a ``shard_map`` SPMD core with host-dispatched
edge jits while accepting pipeline-shaped meshes.  That made it too easy
to confuse the helper with true scheduled MPMD.  It is intentionally no
longer exported or executable.
"""

from __future__ import annotations

__all__: list[str] = []


def hybrid_linear_run(*args: object, **kwargs: object) -> object:
    """Reject calls to the removed hybrid helper.

    Raises:
            NotImplementedError: Always. Use
                :func:`spectrax.runtime.mpmd.sxcall` or
                :func:`spectrax.runtime.mpmd.sxjit` for true MPMD execution.

    Args:
        *args: Additional positional arguments forwarded to the wrapped callable or backend.
        **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

    Returns:
        Result described by this helper.
    """
    del args, kwargs
    raise NotImplementedError(
        "spectrax.runtime.spmd.hybrid_linear_run has been removed because it was not a true MPMD path. "
        "Use sxcall/sxjit or spx.run with an MPMD mesh instead."
    )
