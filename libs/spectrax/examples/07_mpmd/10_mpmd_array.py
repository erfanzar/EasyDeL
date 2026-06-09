# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inspect :class:`StagesArray` — the multi-stage array abstraction.

Builds an :class:`StagesArray` via :func:`abstract_stages_array`, inspects
its shards, checks process locality, and gathers all shards to a
single process. Demonstrates the core data type underlying MPMD
pipeline outputs.

Run::

    python -m examples.07_mpmd.10_stages_array
"""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.runtime.types.array import abstract_stages_array


def main():
    """Build an StagesArray, inspect its properties, and gather to process 0."""
    a = abstract_stages_array((4, 256), jnp.float32, [0, 1, 2, 3])

    print(f"shape:  {a.shape}")
    print(f"dtype:  {a.dtype}")
    print(f"shards: {sorted(a.mpmd_idxs)}")
    print(f"local:  {sorted(a.local_shards.keys())}")
    print(f"remote: {sorted(a.remote_mpmd_idxs)}")
    print(f"partially_addressable: {a.partially_addressable}")

    b = a.gather_to_process(0)
    print(f"\ngathered to process 0: {sorted(b.shards.keys())}")
    print(f"gathered shape: {b.shape}")

    single = abstract_stages_array((2, 128), jnp.bfloat16, [2])
    print("\nsingle-shard array on stage 2:")
    print(f"  shape={single.shape}, dtype={single.dtype}")
    print(f"  to_local_array shape: {single.to_local_array().shape}")

    replicated = abstract_stages_array((8,), jnp.float32, [0, 1], replicated=True)
    print(f"\nreplicated array: stages={sorted(replicated.mpmd_idxs)}")
    print(f"  replicated_value shape: {replicated.replicated_value().shape}")


if __name__ == "__main__":
    main()
