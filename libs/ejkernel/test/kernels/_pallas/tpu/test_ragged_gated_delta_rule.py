# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""TPU Pallas tests for ragged_gated_delta_rule decode."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule import ragged_gated_delta_rule as ragged_gdr_pallas
from ejkernel.kernels._xla.ragged_gated_delta_rule import ragged_gated_delta_rule as ragged_gdr_xla
from ejkernel.modules import ragged_gated_delta_rule as ragged_gdr_module


def _has_tpu() -> bool:
    """Report whether a TPU backend is available to JAX.

    Returns:
        ``True`` if ``jax.devices("tpu")`` enumerates at least one device,
        otherwise ``False`` (including when device discovery raises, e.g. no
        TPU backend is registered). Used to gate the module-level ``skipif``
        so the Pallas TPU tests only run on TPU hardware.
    """
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


def _module_shard_specs():
    """Build the ``shard_map`` in/out partition specs for the ragged GDR module.

    All axes are left unsharded (replicated) so the tests exercise the public
    module ``shard_map`` dispatch on a single-device mesh without splitting any
    tensor.

    Returns:
        A ``(in_specs, out_specs)`` tuple where ``in_specs`` is the 8-tuple of
        ``PartitionSpec`` objects for ``(query, key, value, beta, decay, state,
        query_start_loc, state_indices)`` — three-dim replicated specs for the
        per-token/head tensors, two-dim replicated specs for ``beta``/``decay``,
        a four-dim replicated spec for ``state``, and empty ``P()`` specs for
        the two index arrays — and ``out_specs`` is the 2-tuple of replicated
        specs for the ``(output, updated_state)`` return values.
    """
    token_head_spec = P(None, None, None)
    beta_spec = P(None, None)
    state_spec = P(None, None, None, None)
    return (
        (
            token_head_spec,
            token_head_spec,
            token_head_spec,
            beta_spec,
            beta_spec,
            state_spec,
            P(),
            P(),
        ),
        (token_head_spec, state_spec),
    )


def test_ragged_decode_module_shard_map_matches_xla():
    """Public module shard_map path owns the TPU decode dispatch."""
    num_tokens, heads, qk_dim, v_dim, slots = 4, 2, 8, 8, 4
    keys = jax.random.split(jax.random.PRNGKey(1), 6)
    query = jax.random.normal(keys[0], (num_tokens, heads, qk_dim), dtype=jnp.float32)
    key = jax.random.normal(keys[1], (num_tokens, heads, qk_dim), dtype=jnp.float32)
    value = jax.random.normal(keys[2], (num_tokens, heads, v_dim), dtype=jnp.float32)
    beta = jax.nn.sigmoid(jax.random.normal(keys[3], (num_tokens, heads), dtype=jnp.float32))
    decay = jax.random.normal(keys[4], (num_tokens, heads), dtype=jnp.float32) * -0.01
    state = jax.random.normal(keys[5], (slots, heads, qk_dim, v_dim), dtype=jnp.float32) * 0.01
    query_start_loc = jnp.arange(num_tokens + 1, dtype=jnp.int32)
    state_indices = jnp.arange(num_tokens, dtype=jnp.int32)
    in_specs, out_specs = _module_shard_specs()
    mesh = Mesh(jax.devices("tpu")[:1], ("tp",))

    out_module, state_module = ragged_gdr_module(
        query,
        key,
        value,
        beta,
        decay,
        state,
        query_start_loc,
        state_indices,
        chunk_size=4,
        use_qk_l2norm=True,
        platform="pallas",
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
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
        chunk_size=4,
        use_qk_l2norm=True,
    )

    assert jnp.allclose(out_module, out_xla, atol=1e-5, rtol=0)
    assert jnp.allclose(state_module, state_xla, atol=1e-5, rtol=0)


def test_ragged_prefill_module_shard_map_matches_xla():
    """Public module shard_map path owns the TPU prefill dispatch."""
    num_tokens, heads, qk_dim, v_dim, slots = 5, 2, 8, 8, 3
    keys = jax.random.split(jax.random.PRNGKey(2), 6)
    query = jax.random.normal(keys[0], (num_tokens, heads, qk_dim), dtype=jnp.float32)
    key = jax.random.normal(keys[1], (num_tokens, heads, qk_dim), dtype=jnp.float32)
    value = jax.random.normal(keys[2], (num_tokens, heads, v_dim), dtype=jnp.float32)
    beta = jax.nn.sigmoid(jax.random.normal(keys[3], (num_tokens, heads), dtype=jnp.float32))
    decay = jax.random.normal(keys[4], (num_tokens, heads), dtype=jnp.float32) * -0.01
    state = jax.random.normal(keys[5], (slots, heads, qk_dim, v_dim), dtype=jnp.float32) * 0.01
    query_start_loc = jnp.array([0, 2, 5], dtype=jnp.int32)
    state_indices = jnp.array([0, 2], dtype=jnp.int32)
    in_specs, out_specs = _module_shard_specs()
    mesh = Mesh(jax.devices("tpu")[:1], ("tp",))

    out_module, state_module = ragged_gdr_module(
        query,
        key,
        value,
        beta,
        decay,
        state,
        query_start_loc,
        state_indices,
        chunk_size=4,
        use_qk_l2norm=True,
        platform="pallas",
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
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
        chunk_size=4,
        use_qk_l2norm=True,
    )

    assert jnp.allclose(out_module, out_xla, atol=5e-2, rtol=0)
    assert jnp.allclose(state_module, state_xla, atol=5e-2, rtol=0)
