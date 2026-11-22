# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Inference expert combine (local unpermute + psum) equals the ragged-all-to-all combine and a NumPy reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.moe import _communication_utils as communication
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

EP = 4


@pytest.mark.skipif(jax.default_backend() != "tpu" or jax.device_count() != EP, reason="four TPU devices required")
@pytest.mark.parametrize(("tokens", "experts", "k"), [(8, 16, 4), (3, 8, 2), (13, 16, 6)])
def test_psum_unpermute_matches_ragged_combine_and_numpy(tokens, experts, k):
    rng = np.random.default_rng(tokens * 31 + k)
    hidden = 128
    local_experts = experts // EP
    x = jnp.asarray(rng.standard_normal((1, tokens, hidden)), jnp.float32)
    logits = jnp.asarray(rng.standard_normal((tokens, experts)), jnp.float32)
    _, sorted_sel, weights, group_sizes, _ = communication.permute(
        x, logits, num_experts_per_tok=k, num_experts=experts, dtype=jnp.float32
    )
    counts = np.asarray(group_sizes).reshape(EP, local_experts).sum(axis=1)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rows = tokens * k
    values = rng.standard_normal((rows, hidden)).astype(np.float32)  # expert outputs in global sorted order

    # Each shard holds its own segment at [0, count) followed by garbage (the TPU grouped-matmul tail).
    local = np.full((EP, rows, hidden), 1e6, np.float32)
    for r in range(EP):
        local[r, : counts[r]] = values[starts[r] : starts[r] + counts[r]]
    local = jnp.asarray(local.reshape(EP * rows, hidden))
    counts_j = jnp.asarray(counts, jnp.int32)

    mesh = Mesh(np.array(jax.devices()), ("ep",))

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=(P("ep", None), P(), P(), P()), out_specs=P(), check_vma=False)
    def psum_path(local_rows, cnt, sel, w):
        return communication.replicated_expert_psum_unpermute(
            local_rows,
            cnt,
            sel,
            w,
            axis_name="ep",
            batch_size=1,
            sequence_length=tokens,
            num_experts_per_tok=k,
            dtype=jnp.float32,
        )

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=(P("ep", None), P(), P(), P()), out_specs=P(), check_vma=False)
    def ragged_path(local_rows, cnt, sel, w):
        combined = communication.replicated_expert_combine(local_rows, cnt, axis_name="ep", output_rows=rows)
        return communication.unpermute(
            combined, sel, w, batch_size=1, sequence_length=tokens, num_experts_per_tok=k, dtype=jnp.float32
        )

    got = np.asarray(psum_path(local, counts_j, sorted_sel, weights))
    ragged = np.asarray(ragged_path(local, counts_j, sorted_sel, weights))

    unsorted = values[np.argsort(np.asarray(sorted_sel))].reshape(tokens, k, hidden)
    want = np.einsum("tkh,tk->th", unsorted.astype(np.float64), np.asarray(weights, np.float64).reshape(tokens, k))

    assert got.shape == (1, tokens, hidden)
    np.testing.assert_allclose(got[0], want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got, ragged, rtol=1e-5, atol=1e-5)
