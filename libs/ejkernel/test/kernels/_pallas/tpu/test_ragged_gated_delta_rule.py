# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""TPU Pallas tests for ragged_gated_delta_rule decode."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule import ragged_gated_delta_rule as ragged_gdr_pallas
from ejkernel.kernels._xla.ragged_gated_delta_rule import ragged_gated_delta_rule as ragged_gdr_xla


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


def test_ragged_decode_pallas_matches_xla():
    """Decode-only ragged GDR uses the Pallas DMA path and matches XLA."""
    num_tokens, heads, qk_dim, v_dim, slots = 4, 2, 8, 8, 4
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    query = jax.random.normal(keys[0], (num_tokens, heads, qk_dim), dtype=jnp.float32)
    key = jax.random.normal(keys[1], (num_tokens, heads, qk_dim), dtype=jnp.float32)
    value = jax.random.normal(keys[2], (num_tokens, heads, v_dim), dtype=jnp.float32)
    beta = jax.nn.sigmoid(jax.random.normal(keys[3], (num_tokens, heads), dtype=jnp.float32))
    decay = jax.random.normal(keys[4], (num_tokens, heads), dtype=jnp.float32) * -0.01
    state = jax.random.normal(keys[5], (slots, heads, qk_dim, v_dim), dtype=jnp.float32) * 0.01
    query_start_loc = jnp.arange(num_tokens + 1, dtype=jnp.int32)
    state_indices = jnp.arange(num_tokens, dtype=jnp.int32)

    out_pallas, state_pallas = ragged_gdr_pallas(
        query,
        key,
        value,
        beta,
        decay,
        state,
        query_start_loc,
        state_indices,
        use_qk_l2norm=True,
    )
    out_xla, state_xla = ragged_gdr_xla(
        query,
        key,
        value,
        beta,
        decay,
        state,
        query_start_loc,
        state_indices,
        use_qk_l2norm=True,
    )

    assert jnp.allclose(out_pallas, out_xla, atol=1e-5, rtol=0)
    assert jnp.allclose(state_pallas, state_xla, atol=1e-5, rtol=0)
