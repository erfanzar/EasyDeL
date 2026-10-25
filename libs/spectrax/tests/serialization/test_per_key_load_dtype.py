# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``load_pytree(dtype=callable)``: a per-key read dtype on every TensorStore load path."""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest
from spectrax.serialization import AsyncCheckpointManager

_VALUE = 1.0001  # not representable in bfloat16


def _tree():
    return {
        "router": {"weight": jnp.full((4,), _VALUE, jnp.float32)},
        "mlp": {"weight": jnp.full((4,), _VALUE, jnp.float32)},
        "norm": {"weight": jnp.full((4,), _VALUE, jnp.float32)},
        "ids": jnp.arange(4, dtype=jnp.int32),
    }


def _pick(key: str):
    if key.split(".")[-2:] == ["router", "weight"]:
        return jnp.float32
    if key.split(".")[-2:] == ["norm", "weight"]:
        return None
    return jnp.bfloat16


@pytest.mark.parametrize("chunk_size", [None, 1])
@pytest.mark.parametrize("assume_metadata", [False, True])
@pytest.mark.parametrize("index_only", [False, True])
def test_callable_dtype_targets_each_key(tmp_checkpoint_dir, mesh, chunk_size, assume_metadata, index_only):
    mgr = AsyncCheckpointManager()
    mgr.save_pytree(_tree(), tmp_checkpoint_dir, mesh=mesh, prefix="model")
    if index_only:
        os.remove(os.path.join(tmp_checkpoint_dir, "model_structure.json"))

    loaded, _ = mgr.load_pytree(
        tmp_checkpoint_dir,
        mesh,
        prefix="model",
        dtype=_pick,
        chunk_size=chunk_size,
        tensorstore_assume_metadata=assume_metadata,
        can_skip_structure=index_only,
    )

    assert loaded["router"]["weight"].dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(loaded["router"]["weight"]), np.full((4,), _VALUE, np.float32))
    assert loaded["norm"]["weight"].dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(loaded["norm"]["weight"]), np.full((4,), _VALUE, np.float32))
    assert loaded["mlp"]["weight"].dtype == jnp.bfloat16
    assert loaded["ids"].dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(loaded["ids"]), np.arange(4))


def test_plain_dtype_still_casts_every_floating_leaf(tmp_checkpoint_dir, mesh):
    mgr = AsyncCheckpointManager()
    mgr.save_pytree(_tree(), tmp_checkpoint_dir, mesh=mesh, prefix="model")
    loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model", dtype=jnp.bfloat16)
    assert {loaded[name]["weight"].dtype for name in ("router", "mlp", "norm")} == {jnp.dtype(jnp.bfloat16)}
    assert loaded["ids"].dtype == jnp.int32
