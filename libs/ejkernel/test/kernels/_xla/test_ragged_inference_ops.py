# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XLA kernel tests for packed ragged inference operations.

This module exercises the XLA fallback implementations of the ragged/decode inference kernels used by
EasyDeL eSurge-style serving of hybrid linear-attention models (Qwen3-Next/3.5 family). It covers:

- Registry wiring and signature compatibility for all ragged inference kernels.
- ``compute_schedule_table_v2``: the GDN chunk-scheduling table builder for variable-length packed batches.
- ``ragged_causal_conv1d``: a packed, stateful causal depthwise conv1d over ragged sequences.
- ``ragged_gated_delta_rule_v2``: a packed ragged gated-delta-rule kernel for prefill/decode.
- ``fused_conv_decode``: a single-step (decode) fused causal-conv state update plus SiLU output.
- ``gated_delta_rule_grouped_decode``: a single-step grouped (GQA-style) gated-delta-rule decode.

Each kernel is checked either against a hand-rolled NumPy/JAX reference or for shape/finiteness, on the
default (typically CPU) JAX backend; these tests do not require a TPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.fused_conv_decode import fused_conv_decode
from ejkernel.kernels._xla.gated_delta_rule_grouped_decode import (
    gated_delta_rule_grouped_decode,
)
from ejkernel.kernels._xla.gdn_compute_schedule_v2 import compute_schedule_table_v2
from ejkernel.kernels._xla.ragged_causal_conv1d import ragged_causal_conv1d
from ejkernel.kernels._xla.ragged_gated_delta_rule_v2 import ragged_gated_delta_rule_v2


def test_ragged_inference_kernels_are_registered_and_signature_compatible():
    """Verify every ragged inference kernel is registered for XLA and passes signature validation.

    Iterates the kernel names ``fused_conv_decode``, ``gated_delta_rule_grouped_decode``,
    ``gdn_compute_schedule_v2``, ``ragged_causal_conv1d`` and ``ragged_gated_delta_rule_v2``; for each it
    asserts ``kernel_registry.get(name, platform="xla", backend="any")`` returns a non-``None`` impl and
    that ``kernel_registry.validate_signatures(name)`` reports the registered signatures as compatible.
    """
    for name in (
        "fused_conv_decode",
        "gated_delta_rule_grouped_decode",
        "gdn_compute_schedule_v2",
        "ragged_causal_conv1d",
        "ragged_gated_delta_rule_v2",
    ):
        impl = kernel_registry.get(name, platform="xla", backend="any")
        assert impl is not None
        assert kernel_registry.validate_signatures(name)


def test_gdn_compute_schedule_v2_xla_known_layout():
    """Pin the GDN v2 schedule table output against a hand-computed expected layout.

    Builds a schedule for three requests (token offsets ``[0, 1, 5, 9]``) with ``decode_tokens=1``,
    ``num_valid_seqs=3``, ``max_tokens=9``, ``chunk_size=4``, ``BT=4`` and ``alignment=4``. Asserts the
    returned table has shape ``(9, 23)`` (``11 + 3 * alignment`` columns), that ``total_blocks`` equals 3,
    that the first three rows match the golden ``expected_first_rows`` work descriptors exactly, and that
    all remaining rows are zero-padded.
    """
    query_start_loc = jnp.array([0, 1, 5, 9], dtype=jnp.int32)
    table, total_blocks = compute_schedule_table_v2(
        query_start_loc,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(3, dtype=jnp.int32),
        max_tokens=9,
        chunk_size=4,
        BT=4,
        alignment=4,
    )

    expected_first_rows = jnp.array(
        [
            [1, 0, 1, 3, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            [1, 4, 1, 4, 0, 0, 0, 0, 1, 0, 1, 1, 2, 2, 2, 0, 1, 0, 0, 1, 0, 0, 0],
            [1, 8, 2, 1, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )

    assert table.shape == (9, 23)
    assert int(total_blocks) == 3
    assert jnp.array_equal(table[:3], expected_first_rows)
    assert jnp.array_equal(table[3:], jnp.zeros_like(table[3:]))


def test_ragged_causal_conv1d_xla_matches_zero_state_reference():
    """Verify the ragged causal conv1d matches a hand-rolled zero-initial-state reference.

    Packs three tokens for two requests (``query_start_loc=[0, 1, 3]``, ``distribution=[1, 2, 2]``) with a
    constant ``0.25`` depthwise kernel (``d_conv=3``), zero initial conv state and ``apply_silu=False``.
    Because the conv state starts at zero, each output reduces to a simple weighted sum of the current and
    (within-request) previous tokens: request 0's single token is ``x[0] * 0.25``; request 1's first token
    is ``x[1] * 0.25`` and its second token is ``x[1] * 0.25 + x[2] * 0.25``. Asserts the output and updated
    state keep their input shapes and that the output matches this expected stack.
    """
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
    kernel = jnp.ones((4, 3), dtype=jnp.float32) * 0.25
    conv_state = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    query_start_loc = jnp.array([0, 1, 3], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)

    out, state = ragged_causal_conv1d(
        x,
        conv_state,
        kernel,
        query_start_loc,
        state_indices,
        distribution,
        d_conv=3,
        apply_silu=False,
    )

    expected = jnp.stack(
        [
            x[0] * 0.25,
            x[1] * 0.25,
            x[1] * 0.25 + x[2] * 0.25,
        ]
    )
    assert out.shape == x.shape
    assert state.shape == conv_state.shape
    assert jnp.allclose(out, expected)


def test_ragged_gated_delta_rule_v2_xla_decode_shape_and_finiteness():
    """Smoke-test the ragged gated-delta-rule v2 kernel for correct shapes and finite outputs.

    Constructs a two-token, two-request packed decode batch (``query_start_loc=[0, 1, 2]``,
    ``state_indices=[0, 1]``, ``distribution=[2, 2, 2]``) with a single key/value head, small random fused
    ``mixed_qkv`` (layout ``2 * key_dim + value_dim``), zero gate/decay inputs and zero recurrent state, then
    runs the kernel with ``chunk_size=2``. Asserts the output has shape ``(tokens, n_v * d_v)``, the returned
    state keeps the recurrent-state shape ``(2, n_v, d_k, d_v)``, and both are finite.
    """
    tokens, n_kq, n_v, d_k, d_v = 2, 1, 1, 4, 3
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    mixed_qkv = (
        jax.random.normal(
            jax.random.PRNGKey(0),
            (tokens, 2 * key_dim + value_dim),
            dtype=jnp.float32,
        )
        * 0.05
    )
    b = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    a = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    recurrent_state = jnp.zeros((2, n_v, d_k, d_v), dtype=jnp.float32)
    A_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, 1, 2], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)

    state, out = ragged_gated_delta_rule_v2(
        mixed_qkv,
        b,
        a,
        recurrent_state,
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
    )

    assert out.shape == (tokens, n_v * d_v)
    assert state.shape == recurrent_state.shape
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(jnp.isfinite(state))


def test_fused_conv_decode_xla_matches_reference():
    """Verify the fused single-step conv-decode XLA fallback matches a hand-rolled reference.

    For ``num_slots=2`` cache slots, ``conv_dim=4`` channels and ``d_conv=3`` window, draws a random conv
    state, new tokens and depthwise kernel, then runs ``fused_conv_decode``. The expected behavior is to
    shift the per-slot conv window left by one and append the new token (``concat(state[:, :, 1:],
    new_tokens[..., None])``), then produce ``silu(sum(updated_state * kernel, axis=-1))``. Asserts the
    updated state keeps the conv-state shape, the output has shape ``(num_slots, conv_dim)``, and both match
    the reference via ``jnp.allclose``.
    """
    num_slots, conv_dim, d_conv = 2, 4, 3
    rng = jax.random.PRNGKey(7)
    conv_state = jax.random.normal(rng, (num_slots, conv_dim, d_conv), dtype=jnp.float32)
    new_tokens = jax.random.normal(jax.random.fold_in(rng, 1), (num_slots, conv_dim), dtype=jnp.float32)
    kernel = jax.random.normal(jax.random.fold_in(rng, 2), (conv_dim, d_conv), dtype=jnp.float32)

    updated_state, conv_output = fused_conv_decode(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=jnp.float32,
    )

    expected_state = jnp.concatenate(
        [conv_state[:, :, 1:], new_tokens[:, :, None]],
        axis=-1,
    )
    expected_output = jax.nn.silu(jnp.sum(expected_state * kernel[None, :, :], axis=-1))

    assert updated_state.shape == conv_state.shape
    assert conv_output.shape == (num_slots, conv_dim)
    assert jnp.allclose(updated_state, expected_state)
    assert jnp.allclose(conv_output, expected_output)


def test_gated_delta_rule_grouped_decode_xla_matches_reference():
    """Verify the grouped (GQA-style) gated-delta-rule decode XLA fallback matches a hand-rolled reference.

    Sets up a single decode step with ``num_k_heads=3`` key/query heads each expanded to ``expand_ratio=2``
    value heads (``num_v_heads=6``), ``head_dim=4`` and ``value_dim=5``. Runs
    ``gated_delta_rule_grouped_decode`` over random query/key/value, sigmoid ``beta``, negative-softplus
    ``decay`` and a random recurrent state. Recomputes the expected result directly in the grouped 5D layout:
    scale the query by ``head_dim ** -0.5``, decay the state by ``exp(decay)``, form the key-value memory
    readout, apply the delta-rule update ``(value - kv_mem) * beta``, add the outer-product update, and read
    out with the scaled query, then reshape back to the flat ``(batch, num_v_heads, ...)`` layout. Asserts the
    output and new state have shapes ``(batch, num_v_heads, value_dim)`` and ``(batch, num_v_heads, head_dim,
    value_dim)`` and match the reference to rtol/atol ``1e-5``.
    """
    batch, num_k_heads, expand_ratio, head_dim, value_dim = 2, 3, 2, 4, 5
    num_v_heads = num_k_heads * expand_ratio
    rng = jax.random.PRNGKey(11)

    query = jax.random.normal(rng, (batch, num_k_heads, head_dim), dtype=jnp.float32)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (batch, num_k_heads, head_dim), dtype=jnp.float32)
    value = jax.random.normal(
        jax.random.fold_in(rng, 2),
        (batch, num_k_heads, expand_ratio, value_dim),
        dtype=jnp.float32,
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.fold_in(rng, 3), (batch, num_k_heads, expand_ratio), dtype=jnp.float32)
    )
    decay = -jax.nn.softplus(
        jax.random.normal(jax.random.fold_in(rng, 4), (batch, num_k_heads, expand_ratio), dtype=jnp.float32)
    )
    recurrent_state = jax.random.normal(
        jax.random.fold_in(rng, 5),
        (batch, num_v_heads, head_dim, value_dim),
        dtype=jnp.float32,
    )

    output, new_state = gated_delta_rule_grouped_decode(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
    )

    # Reference computation in grouped 5D layout.
    scale = head_dim**-0.5
    query_s = query * scale
    gs = recurrent_state.reshape(batch, num_k_heads, expand_ratio, head_dim, value_dim)
    gs = gs * jnp.exp(decay)[:, :, :, None, None]
    kv_mem = jnp.einsum("bkehv,bkh->bkev", gs, key)
    delta = (value - kv_mem) * beta[:, :, :, None]
    gs = gs + jnp.einsum("bkh,bkev->bkehv", key, delta)
    expected_output = jnp.einsum("bkehv,bkh->bkev", gs, query_s)
    expected_output = expected_output.reshape(batch, num_v_heads, value_dim)
    expected_state = gs.reshape(batch, num_v_heads, head_dim, value_dim)

    assert output.shape == (batch, num_v_heads, value_dim)
    assert new_state.shape == (batch, num_v_heads, head_dim, value_dim)
    assert jnp.allclose(output, expected_output, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(new_state, expected_state, rtol=1e-5, atol=1e-5)


def test_gated_delta_rule_grouped_decode_xla_handles_none_decay():
    """Verify the grouped gated-delta-rule decode treats ``decay=None`` identically to zero decay.

    Runs ``gated_delta_rule_grouped_decode`` twice over the same random inputs, once passing ``None`` for the
    decay argument and once passing an explicit ``jnp.zeros_like(beta)`` decay, and asserts the two outputs
    and the two resulting states are equal to rtol/atol ``1e-5`` (since ``exp(0) == 1`` leaves the state
    undecayed in both cases).
    """
    batch, num_k_heads, expand_ratio, head_dim, value_dim = 1, 2, 2, 4, 5
    num_v_heads = num_k_heads * expand_ratio
    rng = jax.random.PRNGKey(13)

    query = jax.random.normal(rng, (batch, num_k_heads, head_dim), dtype=jnp.float32)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (batch, num_k_heads, head_dim), dtype=jnp.float32)
    value = jax.random.normal(
        jax.random.fold_in(rng, 2),
        (batch, num_k_heads, expand_ratio, value_dim),
        dtype=jnp.float32,
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.fold_in(rng, 3), (batch, num_k_heads, expand_ratio), dtype=jnp.float32)
    )
    recurrent_state = jax.random.normal(
        jax.random.fold_in(rng, 4),
        (batch, num_v_heads, head_dim, value_dim),
        dtype=jnp.float32,
    )

    output_none, state_none = gated_delta_rule_grouped_decode(
        query,
        key,
        value,
        beta,
        None,
        recurrent_state,
    )
    output_zero, state_zero = gated_delta_rule_grouped_decode(
        query,
        key,
        value,
        beta,
        jnp.zeros_like(beta),
        recurrent_state,
    )

    assert jnp.allclose(output_none, output_zero, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(state_none, state_zero, rtol=1e-5, atol=1e-5)
