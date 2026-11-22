"""Authoritative DSA visibility and effective KDA mesh regressions (accelerator only)."""

import jax
import numpy as np
import pytest
from easydel.operations import OperationMetadata
from easydel.operations.kernels.glm_moe_dsa_indexer import GlmMoeDsaIndexerOp
from easydel.operations.kernels.kda import KernelDeltaAttnOp
from jax import numpy as jnp
from jax.sharding import Mesh

from .test_indexed_dsa_kda_continuation import reference_kda


@pytest.mark.parametrize("explicit_visibility", [False, True])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("mask_kind", ["causal", "boolean", "additive", "min_f32", "min_bf16", "min_f16"])
def test_dsa_authoritative_mask_excludes_every_ineligible_filler(explicit_visibility, rank, mask_kind):
    metadata = OperationMetadata(runtime_dtype=jnp.float32, _stored_mesh=None)
    mask = None
    expected = np.tril(np.ones((3, 3), bool))[None]
    if mask_kind != "causal":
        # The first query is fully masked; key one outranks key two once it
        # becomes visible. Selecting all keys forces masked filler slots.
        allowed = np.array([[False, True, True]])
        expected = expected & allowed[:, None, :]
        if mask_kind == "boolean":
            mask = jnp.asarray(allowed)
        else:
            dtype = {"min_f32": jnp.float32, "min_bf16": jnp.bfloat16, "min_f16": jnp.float16}.get(
                mask_kind, jnp.float32
            )
            blocked = -jnp.inf if mask_kind == "additive" else jnp.finfo(dtype).min
            mask = jnp.where(jnp.asarray(allowed), jnp.asarray(0, dtype), jnp.asarray(blocked, dtype))
        if rank == 3:
            mask = mask[:, None, :]
    extra = {"query_positions": jnp.arange(3)[None]} if explicit_visibility else {}
    output = GlmMoeDsaIndexerOp(metadata).forward_native(
        jnp.ones((1, 3, 1, 1)),
        jnp.array([[[100.0], [3.0], [2.0]]]),
        jnp.ones((1, 3, 1)),
        jnp.arange(3)[None],
        qk_rope_head_dim=0,
        index_topk=3,
        softmax_scale=1.0,
        attention_mask=mask,
        precision=jax.lax.Precision.HIGHEST,
        **extra,
    )
    np.testing.assert_array_equal(output.topk_mask, expected)
    assert output.topk_mask.dtype == jnp.bool_
    raw = np.asarray(output.topk_indices)
    if explicit_visibility:
        np.testing.assert_array_equal((raw >= 0).sum(-1), expected.sum(-1))
        for query in range(3):
            valid = raw[0, query][raw[0, query] >= 0]
            np.testing.assert_array_equal(np.sort(valid), np.flatnonzero(expected[0, query]))
    else:
        # Backward compatibility: only the authoritative boolean mask removes
        # fillers; the documented legacy raw top-k still contains all k keys.
        np.testing.assert_array_equal(np.sort(raw, axis=-1), np.broadcast_to(np.arange(3), (1, 3, 3)))


def test_dsa_finite_additive_bias_still_changes_ranking():
    output = GlmMoeDsaIndexerOp(OperationMetadata(runtime_dtype=jnp.float32, _stored_mesh=None)).forward_native(
        jnp.ones((1, 3, 1, 1)),
        jnp.array([[[2.0], [3.0], [1.0]]]),
        jnp.ones((1, 3, 1)),
        jnp.arange(3)[None],
        qk_rope_head_dim=0,
        index_topk=1,
        softmax_scale=1.0,
        attention_mask=jnp.array([[-4.0, 0.0, 4.0]]),
        precision=jax.lax.Precision.HIGHEST,
    )
    np.testing.assert_array_equal(output.topk_indices[0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(output.topk_mask[0], np.eye(3, dtype=bool))


@pytest.mark.parametrize("mesh_axis", ["tp", "dp", "sp"])
@pytest.mark.parametrize("divisible", [False, True])
def test_dsa_tpu_dispatch_reduced_mesh_keeps_valid_row_placement(mesh_axis, divisible):
    devices = np.asarray(jax.devices())
    mesh_size = len(devices)
    batch = mesh_size if mesh_axis == "dp" and divisible else 1
    seq = 2 * mesh_size if divisible else 2 * mesh_size + 1
    mesh = Mesh(devices, (mesh_axis,))
    metadata = OperationMetadata(runtime_dtype=jnp.float32, _stored_mesh=mesh)
    positions = jnp.broadcast_to(jnp.arange(seq), (batch, seq))
    output = GlmMoeDsaIndexerOp(metadata).forward_tpu(
        jnp.ones((batch, seq, 1, 1), dtype=jnp.float32),
        jnp.broadcast_to(jnp.arange(1, seq + 1, dtype=jnp.float32)[None, :, None], (batch, seq, 1)),
        jnp.ones((batch, seq, 1), dtype=jnp.float32),
        positions,
        qk_rope_head_dim=0,
        index_topk=2,
        softmax_scale=1.0,
        query_positions=positions,
        precision=jax.lax.Precision.HIGHEST,
    )
    # Scores increase strictly with the physical key position. Each row selects
    # its own key then the preceding key, with -1 for the first row's filler.
    expected_indices = np.stack((np.arange(seq), np.arange(seq) - 1), axis=-1)
    expected_mask = np.eye(seq, dtype=bool) | np.eye(seq, k=-1, dtype=bool)
    np.testing.assert_array_equal(output.topk_indices, np.broadcast_to(expected_indices, (batch, seq, 2)))
    np.testing.assert_array_equal(output.topk_mask, np.broadcast_to(expected_mask, (batch, seq, seq)))
    assert output.topk_mask.dtype == jnp.bool_
    # Do not merely avoid a missing-axis exception by replicating everything:
    # the surviving dp member of (fsdp, dp), or sp, must still partition rows.
    spec = tuple(output.topk_indices.sharding.spec)
    spec = spec + (None,) * (3 - len(spec))
    for dim in range(3):
        entry = spec[dim]
        axes = entry if isinstance(entry, tuple) else (() if entry is None else (entry,))
        is_row_axis = (mesh_axis, dim) in (("dp", 0), ("sp", 1))
        expected_axis = (mesh_axis,) if is_row_axis and output.topk_indices.shape[dim] % mesh_size == 0 else ()
        assert axes == expected_axis


@pytest.mark.parametrize("continuation", [False, True])
def test_per_channel_kda_partial_compound_mesh_matches_independent_recurrence(continuation):
    devices = np.asarray(jax.devices())
    batch = len(devices)
    # The default policy names (fsdp, dp) for batch and tp for heads; only dp
    # exists here. Both constraints and local-kernel dispatch must use the
    # effective mesh axes, retaining the surviving dp batch placement.
    mesh = Mesh(devices, ("dp",))
    metadata = OperationMetadata(runtime_dtype=jnp.float32, runtime_softmax_dtype=jnp.float32, _stored_mesh=mesh)
    rng = np.random.default_rng(859)
    q, k, v = (rng.normal(size=(batch, 5, 1, 2)).astype(np.float32) for _ in range(3))
    beta = rng.uniform(0.1, 0.8, (batch, 5, 1)).astype(np.float32)
    decay = rng.uniform(-0.2, -0.02, (batch, 5, 1, 2)).astype(np.float32)
    initial = rng.normal(size=(batch, 1, 2, 2)).astype(np.float32)
    segments = np.broadcast_to([0, 0, -1, 1, 1], (batch, 5)).astype(np.int32)
    expected_out, expected_state = reference_kda(q, k, v, beta, decay, initial, segments, continuation)
    output = KernelDeltaAttnOp(metadata).forward_native(
        *(jnp.asarray(x) for x in (q, k, v, beta, decay)),
        recurrent_state=jnp.asarray(initial),
        segment_ids=jnp.asarray(segments),
        chunk_size=2,
        continue_initial_state=continuation,
    )
    np.testing.assert_allclose(output.attention_outputs, expected_out, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(output.recurrent_state, expected_state, atol=2e-6, rtol=2e-6)
    assert output.attention_outputs.shape == (batch, 5, 1, 2)
    assert output.attention_outputs.dtype == jnp.float32
    assert output.recurrent_state.shape == (batch, 1, 2, 2)
    assert output.recurrent_state.dtype == jnp.float32
