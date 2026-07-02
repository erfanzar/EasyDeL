# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Equivalence tests for Qwen3-Next grouped single-step / packed decode kernels (MPMD variant).

This module verifies that the fast grouped Gated-Delta-Rule (GDR) decode and packed
state-update paths used by Qwen3-Next inference produce results that match slower,
straight-line reference implementations.

The tests cover three layers of the Qwen3-Next linear-attention decode stack:

* ``apply_grouped_single_step_gdr`` — the grouped single-token GDR step that shares
  one query/key head across ``expand_ratio`` value heads, checked against
  ``_legacy_single_step`` (which materialises the repeated heads and calls the XLA
  single-step kernel ``_single_step_gdr_fwd``).
* ``_apply_qwen3_next_packed_updates`` / ``_apply_qwen3_next_packed_updates_unified`` —
  the packed multi-request conv1d + GDR state update that consumes a packed token
  buffer described by ``query_start_loc``, checked against ``_reference_packed_updates``
  (a Python for-loop over requests).
* ``_apply_qwen3_next_packed_updates`` dispatch logic (unified vs ragged) driven by explicit config-style
  arguments, and the ``gated_delta_rule_grouped_decode`` Pallas/JAX equivalence on a tensor-parallel mesh.

The ``_make_*`` helpers build deterministic random inputs for each scenario (plain
decode, packed decode-like, mixed prefill/decode, many short prefills, large padded
buckets, partial buckets, and a tensor-parallel grouped layout). ``_make_runtime_mesh``
and ``_make_gdr_op`` construct the SpectraX mesh and ``GatedDeltaRuleOp`` operation used
to drive the kernels under a realistic sharding configuration. TPU/multi-device-only
tests are guarded with ``pytest.mark.skipif``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.gated_delta_rule._xla_impl_fwd import _single_step_gdr_fwd
from ejkernel.modules.operations import gated_delta_rule_grouped_decode
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax import PartitionAxis, PartitionManager, SpxMesh

import easydel.modules.qwen3_next.modeling_qwen3_next as qwen3_next_modeling
from easydel.modules.qwen3_next.modeling_qwen3_next import (
    _apply_qwen3_next_depthwise_conv_sequence,
    _apply_qwen3_next_packed_updates,
    _apply_qwen3_next_packed_updates_unified,
    _finalize_qwen3_next_conv_state_from_combined,
    _preserve_array_sharding,
    apply_grouped_single_step_gdr,
)
from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig
from easydel.operations import OperationMetadata
from easydel.operations.kernels import GatedDeltaRuleOp
from easydel.utils.inference_mode import set_inference_mode


def _make_decode_inputs(dtype=jnp.bfloat16):
    """Build deterministic single-step grouped GDR decode inputs.

    Produces a tiny grouped layout with ``batch=2``, one decode token (``seq_len=1``),
    ``num_k_heads=3`` key/query heads, ``head_k_dim=4`` key/query head dim, ``num_v_heads=6``
    value heads (so ``expand_ratio = 6 // 3 = 2``), and ``head_v_dim=5`` value head dim. All
    arrays are sampled in float32 from a fixed PRNG seed and cast to ``dtype`` for reproducibility.

    Args:
        dtype: Floating dtype the returned arrays are cast to. Defaults to ``jnp.bfloat16``.

    Returns:
        Tuple ``(query, key, value, beta, decay, recurrent_state)`` where ``query`` and ``key``
        have shape ``(2, 1, 3, 4)``, ``value`` has shape ``(2, 1, 6, 5)``, ``beta`` and ``decay``
        have shape ``(2, 1, 6)``, and ``recurrent_state`` has shape ``(2, 6, 4, 5)``, all of ``dtype``.
    """
    rng = jax.random.key(0)
    query = jax.random.normal(rng, (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    value = jax.random.normal(jax.random.fold_in(rng, 2), (2, 1, 6, 5), dtype=jnp.float32).astype(dtype)
    beta = jax.random.normal(jax.random.fold_in(rng, 3), (2, 1, 6), dtype=jnp.float32).astype(dtype)
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (2, 1, 6), dtype=jnp.float32).astype(dtype)
    recurrent_state = jax.random.normal(jax.random.fold_in(rng, 5), (2, 6, 4, 5), dtype=jnp.float32).astype(dtype)
    return query, key, value, beta, decay, recurrent_state


def _legacy_single_step(query, key, value, beta, decay, recurrent_state):
    """Reference single-step GDR via head-repeated XLA kernel ``_single_step_gdr_fwd``.

    Materialises the grouped layout into a dense per-head layout: each key/query head is
    repeated ``expand_ratio = num_v_heads // num_k_heads`` times so every value head has its own
    key/query, then transposes the ``(batch, seq, heads, dim)`` inputs to the
    ``(batch, heads, seq, dim)`` layout expected by ``_single_step_gdr_fwd`` before calling it.
    The output is transposed back to ``(batch, seq, heads, dim)``.

    Args:
        query: Query tensor of shape ``(batch, seq=1, num_k_heads, head_k_dim)``.
        key: Key tensor of shape ``(batch, seq=1, num_k_heads, head_k_dim)``.
        value: Value tensor of shape ``(batch, seq=1, num_v_heads, head_v_dim)``.
        beta: Delta-rule beta gate of shape ``(batch, seq=1, num_v_heads)``.
        decay: Per-head log-decay of shape ``(batch, seq=1, num_v_heads)``, or ``None`` to disable decay.
        recurrent_state: Carried recurrent state of shape ``(batch, num_v_heads, head_k_dim, head_v_dim)``.

    Returns:
        Tuple ``(output, state)`` where ``output`` has shape ``(batch, seq=1, num_v_heads, head_v_dim)``
        and ``state`` is the updated recurrent state of shape ``(batch, num_v_heads, head_k_dim, head_v_dim)``.
    """
    expand_ratio = value.shape[2] // query.shape[2]
    legacy_output, legacy_state = _single_step_gdr_fwd(
        query=jnp.repeat(query, expand_ratio, axis=2).transpose(0, 2, 1, 3),
        key=jnp.repeat(key, expand_ratio, axis=2).transpose(0, 2, 1, 3),
        value=value.transpose(0, 2, 1, 3),
        beta=beta.transpose(0, 2, 1),
        decay=None if decay is None else decay.transpose(0, 2, 1),
        recurrent_state=recurrent_state,
    )
    return legacy_output.transpose(0, 2, 1, 3), legacy_state


def _reference_packed_updates(
    *,
    conv_states,
    recurrent_states,
    conv_input,
    beta,
    decay,
    kernel,
    query_start_loc,
    num_requests,
    key_dim,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    expand_ratio,
    conv_output_dtype,
    gdr_op,
    **_unused,
):
    """Straight-line per-request reference for packed Qwen3-Next conv1d + GDR state updates.

    Iterates over each request slot, slices its packed tokens from ``conv_input`` using
    ``query_start_loc``, runs the depthwise conv1d over the prepended conv-state context, splits
    the conv output into query/key/value, repeats key/query heads when ``expand_ratio > 1``, runs
    the grouped GDR op, and writes the updated conv state, recurrent state, and per-token outputs
    back. Slots beyond ``num_requests`` or with empty token spans are left untouched, which is what
    the fast packed kernel is checked against.

    Args:
        conv_states: Per-slot conv1d state of shape ``(num_slots, conv_dim, d_conv)``.
        recurrent_states: Per-slot GDR recurrent state of shape ``(num_slots, num_v_heads, head_k_dim, head_v_dim)``.
        conv_input: Packed token buffer of shape ``(1, seq_len, conv_dim)`` holding all requests' tokens.
        beta: Packed delta-rule beta gates of shape ``(1, seq_len, num_v_heads)``.
        decay: Packed per-head log-decay of shape ``(1, seq_len, num_v_heads)``.
        kernel: Depthwise conv1d kernel of shape ``(conv_dim, d_conv)``.
        query_start_loc: Cumulative per-request token offsets of shape ``(num_slots + 1,)``; request
            ``slot`` owns tokens ``[query_start_loc[slot], query_start_loc[slot + 1])``.
        num_requests: Scalar count of active requests; slots ``>= num_requests`` are skipped.
        key_dim: Flattened key/query channel width (``num_k_heads * head_k_dim``).
        num_k_heads: Number of key/query heads.
        head_k_dim: Key/query head dimension.
        num_v_heads: Number of value heads.
        head_v_dim: Value head dimension.
        expand_ratio: Number of value heads sharing one key/query head (``num_v_heads // num_k_heads``).
        conv_output_dtype: Dtype the conv1d output is computed in.
        gdr_op: Callable grouped GDR op invoked per request; returns an object exposing
            ``recurrent_state`` and ``attention_outputs``.
        **_unused: Absorbs any extra keyword arguments (e.g. ``ragged_gdr_op``) so the same input
            dict can be passed to this reference and to the kernel under test.

    Returns:
        Tuple ``(updated_conv_states, updated_recurrent_states, token_outputs)`` where
        ``updated_conv_states`` has shape ``(num_slots, conv_dim, d_conv)``, ``updated_recurrent_states``
        has shape ``(num_slots, num_v_heads, head_k_dim, head_v_dim)``, and ``token_outputs`` has shape
        ``(seq_len, num_v_heads, head_v_dim)`` in float32 with zeros for tokens not owned by a request.
    """
    seq_len = conv_input.shape[1]
    d_conv = kernel.shape[1]
    num_slots = min(conv_states.shape[0], query_start_loc.shape[0] - 1)
    request_count = int(np.asarray(num_requests))
    updated_conv_states = conv_states
    updated_recurrent_states = recurrent_states
    token_outputs = jnp.zeros((seq_len, num_v_heads, head_v_dim), dtype=jnp.float32)

    for slot in range(num_slots):
        start = int(np.asarray(query_start_loc[slot]))
        end = int(np.asarray(query_start_loc[slot + 1]))
        length = end - start
        if slot >= request_count or length <= 0:
            continue

        combined_inputs = jnp.concatenate(
            [conv_states[slot].T[None, :, :], conv_input[:, start:end, :]],
            axis=1,
        )
        conv_output = _apply_qwen3_next_depthwise_conv_sequence(
            combined_inputs,
            kernel,
            output_dtype=conv_output_dtype,
        )[:, d_conv:, :]
        query = conv_output[:, :, :key_dim].reshape(1, length, num_k_heads, head_k_dim)
        key = conv_output[:, :, key_dim : key_dim * 2].reshape(1, length, num_k_heads, head_k_dim)
        value = conv_output[:, :, key_dim * 2 :].reshape(1, length, num_v_heads, head_v_dim)
        if expand_ratio > 1:
            query = jnp.repeat(query, expand_ratio, axis=2)
            key = jnp.repeat(key, expand_ratio, axis=2)

        gdr_output = gdr_op(
            query=query,
            key=key,
            value=value,
            beta=beta[:, start:end, :],
            decay=decay[:, start:end, :],
            recurrent_state=recurrent_states[slot : slot + 1],
        )
        updated_conv = _finalize_qwen3_next_conv_state_from_combined(
            combined_inputs,
            jnp.asarray([length + d_conv], dtype=jnp.int32),
            d_conv=d_conv,
            output_dtype=conv_states.dtype,
        )[0]
        updated_conv_states = updated_conv_states.at[slot].set(updated_conv)
        updated_recurrent_states = updated_recurrent_states.at[slot].set(
            gdr_output.recurrent_state[0].astype(updated_recurrent_states.dtype)
        )
        token_outputs = token_outputs.at[start:end].set(gdr_output.attention_outputs[0].astype(token_outputs.dtype))

    return updated_conv_states, updated_recurrent_states, token_outputs


def _make_packed_decode_inputs(dtype=jnp.bfloat16):
    """Build a packed decode-like schedule: 3 active single-token requests across 4 slots.

    Uses ``query_start_loc = [0, 1, 2, 3, 3]`` so the first three slots each own one token and the
    fourth slot is empty, with ``num_requests = 3``. This mirrors a steady-state decode step where
    most slots advance by one token. Sizes: ``num_k_heads=3``, ``head_k_dim=4``, ``num_v_heads=6``,
    ``head_v_dim=5``, ``d_conv=3``, and ``conv_dim = key_dim * 2 + num_v_heads * head_v_dim``.

    Args:
        dtype: Floating dtype the array values are cast to. Defaults to ``jnp.bfloat16``.

    Returns:
        Dict of keyword arguments accepted by both ``_apply_qwen3_next_packed_updates`` and
        ``_reference_packed_updates`` (conv/recurrent states, packed conv input, beta, decay, kernel,
        ``query_start_loc``, ``num_requests``, and the head/dim/expand metadata).
    """
    rng = jax.random.key(42)
    num_slots = 4
    seq_len = 5
    num_requests = jnp.array(3, dtype=jnp.int32)
    num_k_heads = 3
    head_k_dim = 4
    num_v_heads = 6
    head_v_dim = 5
    key_dim = num_k_heads * head_k_dim
    conv_dim = key_dim * 2 + num_v_heads * head_v_dim
    d_conv = 3

    conv_states = jax.random.normal(rng, (num_slots, conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    recurrent_states = jax.random.normal(
        jax.random.fold_in(rng, 1),
        (num_slots, num_v_heads, head_k_dim, head_v_dim),
        dtype=jnp.float32,
    ).astype(dtype)
    conv_input = jax.random.normal(jax.random.fold_in(rng, 2), (1, seq_len, conv_dim), dtype=jnp.float32).astype(dtype)
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.fold_in(rng, 3), (1, seq_len, num_v_heads), dtype=jnp.float32)
    ).astype(dtype)
    decay = (
        -jax.nn.softplus(jax.random.normal(jax.random.fold_in(rng, 4), (1, seq_len, num_v_heads), dtype=jnp.float32))
    ).astype(dtype)
    kernel = jax.random.normal(jax.random.fold_in(rng, 5), (conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    query_start_loc = jnp.array([0, 1, 2, 3, 3], dtype=jnp.int32)

    return {
        "conv_states": conv_states,
        "recurrent_states": recurrent_states,
        "conv_input": conv_input,
        "beta": beta,
        "decay": decay,
        "kernel": kernel,
        "query_start_loc": query_start_loc,
        "num_requests": num_requests,
        "key_dim": key_dim,
        "num_k_heads": num_k_heads,
        "head_k_dim": head_k_dim,
        "num_v_heads": num_v_heads,
        "head_v_dim": head_v_dim,
        "expand_ratio": num_v_heads // num_k_heads,
        "conv_output_dtype": dtype,
    }


def _make_mixed_packed_inputs(dtype=jnp.bfloat16):
    """Build a mixed prefill/decode packed schedule with variable request lengths.

    Uses ``query_start_loc = [0, 1, 4, 6, 6]`` (request token counts 1, 3, 2 then an empty slot)
    with ``num_requests = 3`` over ``seq_len = 6``, exercising the packed kernel with requests of
    differing lengths in one buffer. Head/dim sizes match the other packed builders.

    Args:
        dtype: Floating dtype the array values are cast to. Defaults to ``jnp.bfloat16``.

    Returns:
        Dict of keyword arguments accepted by ``_apply_qwen3_next_packed_updates`` and
        ``_reference_packed_updates`` describing the mixed packed schedule.
    """
    rng = jax.random.key(123)
    num_slots = 4
    seq_len = 6
    num_requests = jnp.array(3, dtype=jnp.int32)
    num_k_heads = 3
    head_k_dim = 4
    num_v_heads = 6
    head_v_dim = 5
    key_dim = num_k_heads * head_k_dim
    conv_dim = key_dim * 2 + num_v_heads * head_v_dim
    d_conv = 3

    conv_states = jax.random.normal(rng, (num_slots, conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    recurrent_states = jax.random.normal(
        jax.random.fold_in(rng, 1),
        (num_slots, num_v_heads, head_k_dim, head_v_dim),
        dtype=jnp.float32,
    ).astype(dtype)
    conv_input = jax.random.normal(jax.random.fold_in(rng, 2), (1, seq_len, conv_dim), dtype=jnp.float32).astype(dtype)
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.fold_in(rng, 3), (1, seq_len, num_v_heads), dtype=jnp.float32)
    ).astype(dtype)
    decay = (
        -jax.nn.softplus(jax.random.normal(jax.random.fold_in(rng, 4), (1, seq_len, num_v_heads), dtype=jnp.float32))
    ).astype(dtype)
    kernel = jax.random.normal(jax.random.fold_in(rng, 5), (conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    query_start_loc = jnp.array([0, 1, 4, 6, 6], dtype=jnp.int32)

    return {
        "conv_states": conv_states,
        "recurrent_states": recurrent_states,
        "conv_input": conv_input,
        "beta": beta,
        "decay": decay,
        "kernel": kernel,
        "query_start_loc": query_start_loc,
        "num_requests": num_requests,
        "key_dim": key_dim,
        "num_k_heads": num_k_heads,
        "head_k_dim": head_k_dim,
        "num_v_heads": num_v_heads,
        "head_v_dim": head_v_dim,
        "expand_ratio": num_v_heads // num_k_heads,
        "conv_output_dtype": dtype,
    }


def _make_many_prefill_packed_inputs(dtype=jnp.bfloat16):
    """Build a packed schedule with many equal-length short prefills (16 slots x 3 tokens).

    Every one of the ``num_slots = 16`` slots owns exactly ``tokens_per_request = 3`` tokens via
    ``query_start_loc = arange(0, seq_len + 1, 3)`` with ``num_requests = 16`` and ``seq_len = 48``.
    Stresses the packed kernel with a large, fully-occupied request grid. Head/dim sizes match the
    other packed builders.

    Args:
        dtype: Floating dtype the array values are cast to. Defaults to ``jnp.bfloat16``.

    Returns:
        Dict of keyword arguments accepted by ``_apply_qwen3_next_packed_updates`` and
        ``_reference_packed_updates`` describing the many-prefill packed schedule.
    """
    rng = jax.random.key(321)
    num_slots = 16
    tokens_per_request = 3
    seq_len = num_slots * tokens_per_request
    num_requests = jnp.array(num_slots, dtype=jnp.int32)
    num_k_heads = 3
    head_k_dim = 4
    num_v_heads = 6
    head_v_dim = 5
    key_dim = num_k_heads * head_k_dim
    conv_dim = key_dim * 2 + num_v_heads * head_v_dim
    d_conv = 3

    conv_states = jax.random.normal(rng, (num_slots, conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    recurrent_states = jax.random.normal(
        jax.random.fold_in(rng, 1),
        (num_slots, num_v_heads, head_k_dim, head_v_dim),
        dtype=jnp.float32,
    ).astype(dtype)
    conv_input = jax.random.normal(jax.random.fold_in(rng, 2), (1, seq_len, conv_dim), dtype=jnp.float32).astype(dtype)
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.fold_in(rng, 3), (1, seq_len, num_v_heads), dtype=jnp.float32)
    ).astype(dtype)
    decay = (
        -jax.nn.softplus(jax.random.normal(jax.random.fold_in(rng, 4), (1, seq_len, num_v_heads), dtype=jnp.float32))
    ).astype(dtype)
    kernel = jax.random.normal(jax.random.fold_in(rng, 5), (conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    query_start_loc = jnp.arange(0, seq_len + 1, tokens_per_request, dtype=jnp.int32)

    return {
        "conv_states": conv_states,
        "recurrent_states": recurrent_states,
        "conv_input": conv_input,
        "beta": beta,
        "decay": decay,
        "kernel": kernel,
        "query_start_loc": query_start_loc,
        "num_requests": num_requests,
        "key_dim": key_dim,
        "num_k_heads": num_k_heads,
        "head_k_dim": head_k_dim,
        "num_v_heads": num_v_heads,
        "head_v_dim": head_v_dim,
        "expand_ratio": num_v_heads // num_k_heads,
        "conv_output_dtype": dtype,
    }


def _make_large_bucket_decode_inputs(dtype=jnp.bfloat16, bucket: int = 512):
    """Build a decode-like packed schedule padded out to a large bucket length.

    Allocates ``conv_input`` / ``beta`` / ``decay`` over ``bucket`` tokens (the padded buffer size)
    but only ``num_slots = 8`` requests of one token each are active via
    ``query_start_loc = arange(num_slots + 1)`` with ``num_requests = 8``; the remaining bucket
    positions are unused padding. Exercises the kernel's handling of large padded buffers with
    sparse occupancy. Head/dim sizes match the other packed builders.

    Args:
        dtype: Floating dtype the array values are cast to. Defaults to ``jnp.bfloat16``.
        bucket: Padded sequence length of the packed token buffer. Defaults to ``512``.

    Returns:
        Dict of keyword arguments accepted by ``_apply_qwen3_next_packed_updates`` and
        ``_reference_packed_updates`` describing the large-bucket decode schedule.
    """
    rng = jax.random.key(777)
    num_slots = 8
    num_requests = jnp.array(8, dtype=jnp.int32)
    num_k_heads = 3
    head_k_dim = 4
    num_v_heads = 6
    head_v_dim = 5
    key_dim = num_k_heads * head_k_dim
    conv_dim = key_dim * 2 + num_v_heads * head_v_dim
    d_conv = 3

    conv_states = jax.random.normal(rng, (num_slots, conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    recurrent_states = jax.random.normal(
        jax.random.fold_in(rng, 1),
        (num_slots, num_v_heads, head_k_dim, head_v_dim),
        dtype=jnp.float32,
    ).astype(dtype)
    conv_input = jax.random.normal(jax.random.fold_in(rng, 2), (1, bucket, conv_dim), dtype=jnp.float32).astype(dtype)
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.fold_in(rng, 3), (1, bucket, num_v_heads), dtype=jnp.float32)
    ).astype(dtype)
    decay = (
        -jax.nn.softplus(jax.random.normal(jax.random.fold_in(rng, 4), (1, bucket, num_v_heads), dtype=jnp.float32))
    ).astype(dtype)
    kernel = jax.random.normal(jax.random.fold_in(rng, 5), (conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    query_start_loc = jnp.arange(num_slots + 1, dtype=jnp.int32)

    return {
        "conv_states": conv_states,
        "recurrent_states": recurrent_states,
        "conv_input": conv_input,
        "beta": beta,
        "decay": decay,
        "kernel": kernel,
        "query_start_loc": query_start_loc,
        "num_requests": num_requests,
        "key_dim": key_dim,
        "num_k_heads": num_k_heads,
        "head_k_dim": head_k_dim,
        "num_v_heads": num_v_heads,
        "head_v_dim": head_v_dim,
        "expand_ratio": num_v_heads // num_k_heads,
        "conv_output_dtype": dtype,
    }


def _make_partial_bucket_prefill_inputs(dtype=jnp.bfloat16, bucket: int = 512, actual_tokens: int = 454):
    """Build a single long prefill that partially fills a large padded bucket.

    A single request owns the first ``actual_tokens`` positions of a ``bucket``-length buffer via
    ``query_start_loc = [0, actual_tokens, actual_tokens, actual_tokens, actual_tokens]`` with
    ``num_requests = 1`` and ``num_slots = 4``; positions ``[actual_tokens, bucket)`` are padding.
    Used to check that the unified (non-ragged) path matches the reference when the bucket is only
    partly used. Head/dim sizes match the other packed builders.

    Args:
        dtype: Floating dtype the array values are cast to. Defaults to ``jnp.bfloat16``.
        bucket: Padded sequence length of the packed token buffer. Defaults to ``512``.
        actual_tokens: Number of real tokens the single request occupies. Defaults to ``454``.

    Returns:
        Dict of keyword arguments accepted by ``_apply_qwen3_next_packed_updates`` and
        ``_reference_packed_updates`` describing the partial-bucket prefill schedule.
    """
    rng = jax.random.key(909)
    num_slots = 4
    num_requests = jnp.array(1, dtype=jnp.int32)
    num_k_heads = 3
    head_k_dim = 4
    num_v_heads = 6
    head_v_dim = 5
    key_dim = num_k_heads * head_k_dim
    conv_dim = key_dim * 2 + num_v_heads * head_v_dim
    d_conv = 3

    conv_states = jax.random.normal(rng, (num_slots, conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    recurrent_states = jax.random.normal(
        jax.random.fold_in(rng, 1),
        (num_slots, num_v_heads, head_k_dim, head_v_dim),
        dtype=jnp.float32,
    ).astype(dtype)
    conv_input = jax.random.normal(jax.random.fold_in(rng, 2), (1, bucket, conv_dim), dtype=jnp.float32).astype(dtype)
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.fold_in(rng, 3), (1, bucket, num_v_heads), dtype=jnp.float32)
    ).astype(dtype)
    decay = (
        -jax.nn.softplus(jax.random.normal(jax.random.fold_in(rng, 4), (1, bucket, num_v_heads), dtype=jnp.float32))
    ).astype(dtype)
    kernel = jax.random.normal(jax.random.fold_in(rng, 5), (conv_dim, d_conv), dtype=jnp.float32).astype(dtype)
    query_start_loc = jnp.array([0, actual_tokens, actual_tokens, actual_tokens, actual_tokens], dtype=jnp.int32)

    return {
        "conv_states": conv_states,
        "recurrent_states": recurrent_states,
        "conv_input": conv_input,
        "beta": beta,
        "decay": decay,
        "kernel": kernel,
        "query_start_loc": query_start_loc,
        "num_requests": num_requests,
        "key_dim": key_dim,
        "num_k_heads": num_k_heads,
        "head_k_dim": head_k_dim,
        "num_v_heads": num_v_heads,
        "head_v_dim": head_v_dim,
        "expand_ratio": num_v_heads // num_k_heads,
        "conv_output_dtype": dtype,
    }


def _make_tp_grouped_decode_inputs(dtype=jnp.bfloat16, batch: int = 8):
    """Build tensor-parallel-shaped grouped single-step decode inputs.

    Produces the "grouped decode" layout consumed by ``gated_delta_rule_grouped_decode`` and
    ``GatedDeltaRuleOp.grouped_gdr_decode_jax``: ``num_k_heads=4`` key/query heads with
    ``head_dim=128``, an ``expand_ratio=4`` so ``num_v_heads = 16``, and ``value_dim=128``. Beta and
    decay carry a per-(head, expand) gate; the recurrent state spans all value heads. Sized so the
    ``num_k_heads`` axis can be sharded across a 4-way tensor-parallel mesh.

    Args:
        dtype: Floating dtype the array values are cast to. Defaults to ``jnp.bfloat16``.
        batch: Number of decode sequences in the batch. Defaults to ``8``.

    Returns:
        Tuple ``(query, key, value, beta, decay, recurrent_state)`` where ``query`` and ``key`` have
        shape ``(batch, 4, 128)``, ``value`` has shape ``(batch, 4, 4, 128)``, ``beta`` and ``decay``
        have shape ``(batch, 4, 4)``, and ``recurrent_state`` has shape ``(batch, 16, 128, 128)``.
    """
    rng = jax.random.key(2026)
    num_k_heads = 4
    expand_ratio = 4
    head_dim = 128
    value_dim = 128
    num_v_heads = num_k_heads * expand_ratio

    query = jax.random.normal(rng, (batch, num_k_heads, head_dim), dtype=jnp.float32).astype(dtype)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (batch, num_k_heads, head_dim), dtype=jnp.float32).astype(dtype)
    value = jax.random.normal(
        jax.random.fold_in(rng, 2),
        (batch, num_k_heads, expand_ratio, value_dim),
        dtype=jnp.float32,
    ).astype(dtype)
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.fold_in(rng, 3), (batch, num_k_heads, expand_ratio), dtype=jnp.float32)
    ).astype(dtype)
    decay = (
        -jax.nn.softplus(
            jax.random.normal(jax.random.fold_in(rng, 4), (batch, num_k_heads, expand_ratio), dtype=jnp.float32)
        )
    ).astype(dtype)
    recurrent_state = jax.random.normal(
        jax.random.fold_in(rng, 5),
        (batch, num_v_heads, head_dim, value_dim),
        dtype=jnp.float32,
    ).astype(dtype)
    return query, key, value, beta, decay, recurrent_state


def _make_runtime_mesh(axis_dims: tuple[int, ...] = (1, 1, -1, 1, 1, 1)) -> SpxMesh:
    """Construct the SpectraX runtime mesh used to drive the GDR kernels under test.

    Builds a throwaway ``Qwen3NextConfig`` with the requested sharding layout on the default JAX
    backend and returns its ``mesh``. The default ``axis_dims`` places all devices on the FSDP axis
    (the third axis, ``-1``); pass ``(1, 1, 1, 1, 4, 1)`` to put 4 devices on the tensor-parallel axis.

    Args:
        axis_dims: Per-axis device counts for ``(pp, dp, fsdp, ep, tp, sp)``; a ``-1`` entry
            absorbs the remaining devices. Defaults to ``(1, 1, -1, 1, 1, 1)``.

    Returns:
        The ``SpxMesh`` constructed by ``Qwen3NextConfig`` for the given layout.
    """
    return Qwen3NextConfig(
        sharding_axis_dims=axis_dims,
        backend=jax.default_backend(),
    ).mesh


def _make_gdr_op(mesh: SpxMesh, runtime_dtype=jnp.bfloat16, axis_dims: tuple[int, ...] = (1, 1, -1, 1, 1, 1)):
    """Construct a ``GatedDeltaRuleOp`` wired to the given mesh and runtime dtype.

    Builds a ``Qwen3NextConfig`` for the requested sharding layout and wraps it in an
    ``OperationMetadata`` (float32 softmax, default backend/platform, the config's partition axis,
    and the supplied mesh) used to instantiate the op. The op is what the packed and grouped-decode
    helpers call to perform the actual GDR computation.

    Args:
        mesh: The ``SpxMesh`` the op should run on (stored on the metadata as ``_stored_mesh``).
        runtime_dtype: Compute dtype for the GDR op. Defaults to ``jnp.bfloat16``.
        axis_dims: Per-axis device counts used to build the backing ``Qwen3NextConfig`` (must be
            consistent with ``mesh``). Defaults to ``(1, 1, -1, 1, 1, 1)``.

    Returns:
        A configured ``GatedDeltaRuleOp`` instance.
    """
    base_config = Qwen3NextConfig(
        sharding_axis_dims=axis_dims,
        backend=jax.default_backend(),
    )
    return GatedDeltaRuleOp(
        OperationMetadata(
            runtime_dtype=runtime_dtype,
            runtime_softmax_dtype=jnp.float32,
            platform=jax.default_backend(),
            backend=jax.default_backend(),
            partition_axis=base_config.partition_axis,
            base_config=base_config,
            _stored_mesh=mesh,
        )
    )


def test_grouped_single_step_gdr_matches_repeated_heads_with_decay():
    """Grouped single-step GDR (with decay) matches the head-repeated legacy kernel.

    Runs both ``apply_grouped_single_step_gdr`` and ``_legacy_single_step`` on the same decode
    inputs with a non-``None`` decay and asserts the output and updated state agree within tolerance,
    and that output/state preserve their respective input dtypes.
    """
    query, key, value, beta, decay, recurrent_state = _make_decode_inputs()

    legacy_output, legacy_state = _legacy_single_step(query, key, value, beta, decay, recurrent_state)
    grouped_output, grouped_state = apply_grouped_single_step_gdr(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        recurrent_state=recurrent_state,
    )

    assert grouped_output.dtype == query.dtype
    assert grouped_state.dtype == recurrent_state.dtype
    assert jnp.allclose(grouped_output.astype(jnp.float32), legacy_output.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(grouped_state.astype(jnp.float32), legacy_state.astype(jnp.float32), rtol=0.02, atol=0.05)


def test_grouped_single_step_gdr_matches_repeated_heads_without_decay():
    """Grouped single-step GDR (decay disabled) matches the head-repeated legacy kernel.

    Same equivalence check as the with-decay variant but passes ``decay=None`` to both paths,
    verifying the no-decay branch agrees on output, updated state, and dtypes.
    """
    query, key, value, beta, _, recurrent_state = _make_decode_inputs()

    legacy_output, legacy_state = _legacy_single_step(query, key, value, beta, None, recurrent_state)
    grouped_output, grouped_state = apply_grouped_single_step_gdr(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=None,
        recurrent_state=recurrent_state,
    )

    assert grouped_output.dtype == query.dtype
    assert grouped_state.dtype == recurrent_state.dtype
    assert jnp.allclose(grouped_output.astype(jnp.float32), legacy_output.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(grouped_state.astype(jnp.float32), legacy_state.astype(jnp.float32), rtol=0.02, atol=0.05)


def test_grouped_gdr_decode_honors_runtime_dtype():
    """Grouped decode honors the op's runtime dtype for the recurrent state.

    Feeds float32 inputs but a ``GatedDeltaRuleOp`` built with ``runtime_dtype=jnp.bfloat16`` and
    asserts the returned output keeps the float32 input dtype while the updated recurrent state is
    cast to the op's bfloat16 runtime dtype.
    """
    query, key, value, beta, decay, recurrent_state = _make_tp_grouped_decode_inputs(dtype=jnp.float32, batch=2)
    mesh = _make_runtime_mesh()

    with mesh:
        gdr_op = _make_gdr_op(mesh, runtime_dtype=jnp.bfloat16)
        grouped_output, grouped_state = apply_grouped_single_step_gdr(
            query=query[:, None, :, :],
            key=key[:, None, :, :],
            value=value.reshape(value.shape[0], 1, -1, value.shape[-1]),
            beta=beta.reshape(beta.shape[0], 1, -1),
            decay=decay.reshape(decay.shape[0], 1, -1),
            recurrent_state=recurrent_state.astype(jnp.float32),
            gdr_op=gdr_op,
        )

    assert grouped_output.dtype == jnp.float32
    assert grouped_state.dtype == jnp.bfloat16


def test_packed_updates_match_reference_loop_for_decode_like_schedule():
    """Packed updates match the per-request reference on a decode-like schedule.

    Runs ``_apply_qwen3_next_packed_updates`` and ``_reference_packed_updates`` on the 3-active-slot
    decode schedule and asserts the conv state, recurrent state, and token outputs all match within
    tolerance. Also asserts the inactive slots (index 3 onward) are left byte-identical to their
    inputs and that outputs for unowned token positions are exactly zero.
    """
    packed_inputs = _make_packed_decode_inputs()
    mesh = _make_runtime_mesh()

    with mesh:
        gdr_op = _make_gdr_op(mesh)
        unified_conv, unified_rec, unified_out = _apply_qwen3_next_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
            ragged_gdr_op=object(),
        )
        ref_conv, ref_rec, ref_out = _reference_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
        )

    assert jnp.allclose(unified_conv.astype(jnp.float32), ref_conv.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(unified_rec.astype(jnp.float32), ref_rec.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(unified_out.astype(jnp.float32), ref_out.astype(jnp.float32), rtol=0.02, atol=0.05)

    assert jnp.allclose(
        unified_conv[3:].astype(jnp.float32),
        packed_inputs["conv_states"][3:].astype(jnp.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert jnp.allclose(
        unified_rec[3:].astype(jnp.float32),
        packed_inputs["recurrent_states"][3:].astype(jnp.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert jnp.allclose(
        unified_out[3:].astype(jnp.float32),
        jnp.zeros_like(unified_out[3:], dtype=jnp.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_packed_updates_match_reference_loop_for_large_bucket_decode_like_schedule():
    """Packed updates match the reference on a sparse decode schedule in a 512-token bucket.

    Drives ``_apply_qwen3_next_packed_updates`` against ``_reference_packed_updates`` with 8
    single-token requests padded into a 512-token bucket and asserts the conv state, recurrent
    state, and token outputs all agree within tolerance.
    """
    packed_inputs = _make_large_bucket_decode_inputs(bucket=512)
    mesh = _make_runtime_mesh()

    with mesh:
        gdr_op = _make_gdr_op(mesh)
        unified_conv, unified_rec, unified_out = _apply_qwen3_next_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
            ragged_gdr_op=object(),
        )
        ref_conv, ref_rec, ref_out = _reference_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
        )

    assert jnp.allclose(unified_conv.astype(jnp.float32), ref_conv.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(unified_rec.astype(jnp.float32), ref_rec.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(unified_out.astype(jnp.float32), ref_out.astype(jnp.float32), rtol=0.02, atol=0.05)


def test_packed_updates_use_unified_when_ragged_disabled_for_partial_prefill_bucket():
    """With ragged GDR disabled, packed updates fall back to the unified path and match it.

    Passes ``use_ragged_gdr=False`` and, on a partial-bucket prefill (454 of 512 tokens used), asserts
    the dispatched ``_apply_qwen3_next_packed_updates`` output equals the directly-called
    ``_apply_qwen3_next_packed_updates_unified`` output, that both match the reference conv/recurrent
    states within tolerance, and that token outputs past the 454th position are exactly zero.
    """
    packed_inputs = _make_partial_bucket_prefill_inputs(bucket=512, actual_tokens=454)
    mesh = _make_runtime_mesh()

    with mesh, set_inference_mode(True):
        gdr_op = _make_gdr_op(mesh)
        dispatched_conv, dispatched_rec, dispatched_out = _apply_qwen3_next_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
            ragged_gdr_op=object(),
            use_ragged_gdr=False,
        )
        unified_conv, unified_rec, unified_out = _apply_qwen3_next_packed_updates_unified(
            **packed_inputs,
            gdr_op=gdr_op,
        )
        ref_conv, ref_rec, _ref_out = _reference_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
        )

    assert jnp.allclose(dispatched_conv.astype(jnp.float32), unified_conv.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(dispatched_rec.astype(jnp.float32), unified_rec.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(dispatched_out.astype(jnp.float32), unified_out.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(unified_conv.astype(jnp.float32), ref_conv.astype(jnp.float32), rtol=0.03, atol=0.06)
    assert jnp.allclose(unified_rec.astype(jnp.float32), ref_rec.astype(jnp.float32), rtol=0.03, atol=0.06)
    assert jnp.allclose(
        dispatched_out[454:].astype(jnp.float32),
        jnp.zeros_like(dispatched_out[454:], dtype=jnp.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_packed_updates_keep_ragged_for_partial_decode_bucket(monkeypatch):
    """With ragged GDR enabled, packed updates dispatch to the ragged path.

    Monkeypatches the unified and ragged packed-update functions
    with distinct sentinel markers (1 vs 2). It then calls ``_apply_qwen3_next_packed_updates`` and
    asserts the result carries the ragged marker (``2``), proving the dispatcher chose the ragged
    implementation for this partial-decode bucket.

    Args:
        monkeypatch: Pytest fixture used to swap the unified/ragged
            packed-update functions for sentinel-returning stubs.
    """
    packed_inputs = _make_large_bucket_decode_inputs(bucket=512)

    unified_marker = (
        jnp.array([1], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
    )
    ragged_marker = (
        jnp.array([2], dtype=jnp.int32),
        jnp.array([2], dtype=jnp.int32),
        jnp.array([2], dtype=jnp.int32),
    )

    monkeypatch.setattr(
        qwen3_next_modeling,
        "_apply_qwen3_next_packed_updates_unified",
        lambda **_: unified_marker,
    )
    monkeypatch.setattr(
        qwen3_next_modeling,
        "_apply_qwen3_next_packed_updates_ragged",
        lambda **_: ragged_marker,
    )

    with set_inference_mode(True):
        dispatched = qwen3_next_modeling._apply_qwen3_next_packed_updates(
            **packed_inputs,
            gdr_op=object(),
            ragged_gdr_op=object(),
        )

    assert int(dispatched[0][0]) == 2


def test_preserve_array_sharding_matches_reference_array():
    """``_preserve_array_sharding`` applies the partition-manager-derived NamedSharding.

    Builds a single-device ``data``-axis mesh and a ``PartitionManager`` whose batch axis maps to
    ``data``, then asserts that the array returned by ``_preserve_array_sharding`` carries the
    expected ``NamedSharding(mesh, PartitionSpec("data", None, None, None))``.
    """
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    partition_axis = PartitionAxis(batch_axis="data", head_axis=None)
    partition_manager = PartitionManager(partition_axis)
    sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None))

    with mesh:
        preserved = _preserve_array_sharding(
            jnp.zeros((2, 3, 4, 5), dtype=jnp.float32),
            partition_manager=partition_manager,
            partition_axis=partition_axis,
        )

    assert preserved.sharding == sharding


def test_packed_prefill_updates_match_reference_loop_for_mixed_schedule():
    """Packed updates (eager and jitted) match the reference on a mixed prefill schedule.

    On the mixed variable-length schedule it asserts that both the eager
    ``_apply_qwen3_next_packed_updates`` and a ``jax.jit``-wrapped invocation of it agree with
    ``_reference_packed_updates`` within tolerance. It further asserts inactive slots (index 3
    onward) keep their input conv/recurrent state byte-for-byte and that token outputs past the
    last owned position (index 6 onward) are exactly zero.
    """
    packed_inputs = _make_mixed_packed_inputs()
    mesh = _make_runtime_mesh()

    with mesh:
        gdr_op = _make_gdr_op(mesh)

        jitted_prefill = jax.jit(
            lambda conv_states, recurrent_states, conv_input, beta, decay, kernel, query_start_loc, num_requests: (
                _apply_qwen3_next_packed_updates(
                    conv_states=conv_states,
                    recurrent_states=recurrent_states,
                    conv_input=conv_input,
                    beta=beta,
                    decay=decay,
                    kernel=kernel,
                    query_start_loc=query_start_loc,
                    num_requests=num_requests,
                    key_dim=packed_inputs["key_dim"],
                    num_k_heads=packed_inputs["num_k_heads"],
                    head_k_dim=packed_inputs["head_k_dim"],
                    num_v_heads=packed_inputs["num_v_heads"],
                    head_v_dim=packed_inputs["head_v_dim"],
                    expand_ratio=packed_inputs["expand_ratio"],
                    conv_output_dtype=packed_inputs["conv_output_dtype"],
                    gdr_op=gdr_op,
                    ragged_gdr_op=object(),
                )
            )
        )

        unified_conv, unified_rec, unified_out = _apply_qwen3_next_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
            ragged_gdr_op=object(),
        )
        jitted_conv, jitted_rec, jitted_out = jitted_prefill(
            packed_inputs["conv_states"],
            packed_inputs["recurrent_states"],
            packed_inputs["conv_input"],
            packed_inputs["beta"],
            packed_inputs["decay"],
            packed_inputs["kernel"],
            packed_inputs["query_start_loc"],
            packed_inputs["num_requests"],
        )
        ref_conv, ref_rec, ref_out = _reference_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
        )

    assert jnp.allclose(unified_conv.astype(jnp.float32), ref_conv.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(unified_rec.astype(jnp.float32), ref_rec.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(unified_out.astype(jnp.float32), ref_out.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(jitted_conv.astype(jnp.float32), ref_conv.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(jitted_rec.astype(jnp.float32), ref_rec.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(jitted_out.astype(jnp.float32), ref_out.astype(jnp.float32), rtol=0.02, atol=0.05)

    assert jnp.allclose(
        unified_conv[3:].astype(jnp.float32),
        packed_inputs["conv_states"][3:].astype(jnp.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert jnp.allclose(
        unified_rec[3:].astype(jnp.float32),
        packed_inputs["recurrent_states"][3:].astype(jnp.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert jnp.allclose(
        unified_out[6:].astype(jnp.float32),
        jnp.zeros_like(unified_out[6:], dtype=jnp.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_packed_prefill_updates_match_reference_loop_for_many_prefills():
    """Packed updates match the reference on a fully-occupied many-prefill grid.

    Runs ``_apply_qwen3_next_packed_updates`` against ``_reference_packed_updates`` on the
    16-slot x 3-token schedule and asserts the conv state, recurrent state, and token outputs all
    agree within tolerance.
    """
    packed_inputs = _make_many_prefill_packed_inputs()
    mesh = _make_runtime_mesh()

    with mesh:
        gdr_op = _make_gdr_op(mesh)
        unified_conv, unified_rec, unified_out = _apply_qwen3_next_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
            ragged_gdr_op=object(),
        )
        ref_conv, ref_rec, ref_out = _reference_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
        )

    assert jnp.allclose(unified_conv.astype(jnp.float32), ref_conv.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(unified_rec.astype(jnp.float32), ref_rec.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(unified_out.astype(jnp.float32), ref_out.astype(jnp.float32), rtol=0.02, atol=0.05)


@pytest.mark.skipif(
    jax.default_backend() != "tpu" or jax.device_count() < 4,
    reason="Requires a 4-device TPU mesh",
)
def test_tp_mesh_helper_uses_tensor_parallel_axis():
    """The GDR op shards the head axis along ``tp`` on a 4-way tensor-parallel TPU mesh.

    Builds a ``(1, 1, 1, 1, 4, 1)`` mesh and op, queries the op's shardings for a dummy decode
    query, and asserts the mesh exposes ``tp == 4`` and that the head axis (index 2) of the query
    sharding is mapped to the ``tp`` axis. Skipped unless running on a TPU with at least 4 devices.
    """
    mesh = _make_runtime_mesh((1, 1, 1, 1, 4, 1))

    with mesh:
        gdr_op = _make_gdr_op(mesh, axis_dims=(1, 1, 1, 1, 4, 1))
        mode = gdr_op.get_mode(query=jnp.zeros((2, 1, 4, 128), dtype=jnp.bfloat16), BTHD=True)
        shardings = gdr_op.metadata.get_shardings(mode, layout="bthd")

    assert gdr_op.metadata.mesh.shape["tp"] == 4
    assert shardings.query[2] == "tp"


@pytest.mark.skipif(
    jax.default_backend() != "tpu" or jax.device_count() < 4,
    reason="Requires a 4-device TPU mesh",
)
def test_grouped_gdr_decode_shard_map_pallas_matches_jax_on_tp_mesh():
    """Pallas shard_map grouped decode matches the pure-JAX reference on a TP mesh.

    On a 4-way tensor-parallel mesh, runs ``gated_delta_rule_grouped_decode`` with
    ``platform="pallas"`` under explicit ``in_specs``/``out_specs`` that shard the head axis across
    ``tp``, then compares its output and updated state against ``GatedDeltaRuleOp.grouped_gdr_decode_jax``
    on the same inputs and asserts they agree within tolerance. Skipped unless running on a TPU with
    at least 4 devices.
    """
    mesh = _make_runtime_mesh((1, 1, 1, 1, 4, 1))
    query, key, value, beta, decay, recurrent_state = _make_tp_grouped_decode_inputs()

    qk_spec = PartitionSpec(None, "tp", None)
    v_spec = PartitionSpec(None, "tp", None, None)
    bd_spec = PartitionSpec(None, "tp", None)
    state_spec = PartitionSpec(None, "tp", None, None)
    out_spec = PartitionSpec(None, "tp", None)

    with mesh:
        _make_gdr_op(mesh, axis_dims=(1, 1, 1, 1, 4, 1))
        pallas_out, pallas_state = gated_delta_rule_grouped_decode(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            recurrent_state=recurrent_state,
            platform="pallas",
            mesh=mesh,
            in_specs=(qk_spec, qk_spec, v_spec, bd_spec, bd_spec, state_spec),
            out_specs=(out_spec, state_spec),
            check_vma=False,
        )
        jax_out, jax_state = GatedDeltaRuleOp.grouped_gdr_decode_jax(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            recurrent_state=recurrent_state,
        )

    assert jnp.allclose(pallas_out.astype(jnp.float32), jax_out.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(pallas_state.astype(jnp.float32), jax_state.astype(jnp.float32), rtol=0.02, atol=0.05)
