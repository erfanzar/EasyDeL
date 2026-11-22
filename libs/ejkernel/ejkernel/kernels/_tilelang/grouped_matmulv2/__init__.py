# Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
"""TileLang ``grouped_matmulv2`` alias package.

Re-exports :func:`grouped_matmulv2` from
:mod:`ejkernel.kernels._tilelang.grouped_matmul._interface`, where both
``grouped_matmul`` and ``grouped_matmulv2`` are registered against the same
TileLang kernel.

Exports:
    grouped_matmulv2: Grouped matmul v2 (same kernel as v1).
"""

from ._interface import grouped_matmulv2

__all__ = ["grouped_matmulv2"]
