# Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
"""TileLang ``mamba2`` alias package for ``state_space_v2``.

Re-exports the ``state_space_v2`` function under the name ``mamba2``.
The kernel registration for all three names (``state_space_v2``, ``mamba2``,
``ssm2``) happens in
:mod:`ejkernel.kernels._tilelang.state_space_v2._interface`.

Exports:
    mamba2: Mamba-2 selective scan (alias for ``state_space_v2``).
"""

from ._interface import mamba2

__all__ = ["mamba2"]
