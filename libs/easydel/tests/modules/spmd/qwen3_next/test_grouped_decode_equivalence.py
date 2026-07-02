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

"""Equivalence tests for Qwen3-Next grouped single-step gated delta-rule (GDR) decode under SPMD.

This module verifies that the production Qwen3-Next linear-attention decode helpers in
``easydel.modules.qwen3_next.modeling_qwen3_next`` produce the same results as slower, easier-to-audit
reference implementations across the schedules that arise during inference. Specifically it covers:

* ``apply_grouped_single_step_gdr`` (grouped-head single-token recurrent update) versus the legacy
  per-head ``_single_step_gdr_fwd`` kernel that explicitly repeats key/query heads to the value-head count.
* ``_apply_qwen3_next_packed_updates`` / ``_apply_qwen3_next_packed_updates_unified`` (the packed,
  per-request fused conv1d + GDR state update used by the eSurge runner) versus a straight-line Python
  reference loop (``_reference_packed_updates``) that processes each request slot independently.
* The unified-vs-ragged dispatch decision driven by explicit config-style arguments and inference mode,
  and the sharding-preservation helper ``_preserve_array_sharding``.

The helpers prefixed with ``_make_*`` build randomized but reproducible input fixtures for the various
schedules (decode-like one-token-per-request, mixed prefill+decode, many short prefills, large padded
buckets, partial buckets, and tensor-parallel decode). ``_make_runtime_mesh`` / ``_make_gdr_op`` build the
SPMD mesh and the ``GatedDeltaRuleOp`` operation object used by the production code paths.

Tests gated with ``@pytest.mark.skipif`` require a real multi-device TPU mesh and exercise the
tensor-parallel sharding axis and the Pallas grouped-decode kernel; they are skipped on CPU/GPU.
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
    """Build a tiny single-step grouped GDR decode fixture with grouped (expanded) value heads.

    Generates random query/key/value/beta/decay tensors plus an initial recurrent state for a one-token
    decode step where there are more value heads than key/query heads (grouped-query style). Shapes follow
    the BTHD-ish layout expected by ``apply_grouped_single_step_gdr``: batch=2, time=1, with 3 key/query heads
    of dim 4 and 6 value heads of dim 5 (expand ratio 2).

    Args:
        dtype: Output dtype the sampled float32 tensors are cast to (default ``jnp.bfloat16``).

    Returns:
        Tuple of ``(query, key, value, beta, decay, recurrent_state)`` where ``query``/``key`` have shape
        ``(2, 1, 3, 4)``, ``value`` has shape ``(2, 1, 6, 5)``, ``beta``/``decay`` have shape ``(2, 1, 6)``,
        and ``recurrent_state`` has shape ``(2, 6, 4, 5)``, all of dtype ``dtype``.
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
    """Reference single-step GDR update via the legacy per-head ``_single_step_gdr_fwd`` kernel.

    Expands the grouped key/query heads up to the value-head count by repeating along the head axis, then
    transposes the BTHD inputs into the head-major layout the legacy kernel expects, runs the single-step
    forward, and transposes the output back. This is the "ground truth" the grouped helper is compared against.

    Args:
        query: Query tensor of shape ``(batch, time, num_k_heads, head_dim)``.
        key: Key tensor of shape ``(batch, time, num_k_heads, head_dim)``.
        value: Value tensor of shape ``(batch, time, num_v_heads, value_dim)``.
        beta: Per-(value-)head gating tensor of shape ``(batch, time, num_v_heads)``.
        decay: Per-(value-)head decay tensor of shape ``(batch, time, num_v_heads)``, or ``None`` to disable
            decay.
        recurrent_state: Recurrent state of shape ``(batch, num_v_heads, head_dim, value_dim)``.

    Returns:
        Tuple ``(output, recurrent_state)`` where ``output`` has shape
        ``(batch, time, num_v_heads, value_dim)`` (transposed back to BTHD) and ``recurrent_state`` is the
        updated recurrent state returned by the legacy kernel.
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

    Processes each request slot independently in plain Python: it concatenates the carried conv state with
    the slot's slice of packed conv input, runs the depthwise conv1d, splits the conv output into query/key/
    value, repeats key/query heads when ``expand_ratio > 1``, runs the per-slot GDR op, and writes back the
    finalized conv state, updated recurrent state, and per-token outputs. Slots beyond ``num_requests`` or
    with non-positive length are left untouched, mirroring the production helper's masking behavior.

    Args:
        conv_states: Per-slot carried conv state of shape ``(num_slots, conv_dim, d_conv)``.
        recurrent_states: Per-slot GDR recurrent state of shape ``(num_slots, num_v_heads, head_k_dim,
            head_v_dim)``.
        conv_input: Packed conv input of shape ``(1, seq_len, conv_dim)`` holding all requests' tokens.
        beta: Packed per-token gating of shape ``(1, seq_len, num_v_heads)``.
        decay: Packed per-token decay of shape ``(1, seq_len, num_v_heads)``.
        kernel: Depthwise conv1d kernel of shape ``(conv_dim, d_conv)``.
        query_start_loc: Int32 prefix-sum offsets of shape ``(num_slots + 1,)`` delimiting each slot's token
            span ``[query_start_loc[slot], query_start_loc[slot + 1])`` within ``conv_input``.
        num_requests: Scalar int holding the number of active requests; slots ``>= num_requests`` are skipped.
        key_dim: Flattened key width ``num_k_heads * head_k_dim`` used to split the conv output.
        num_k_heads: Number of key/query heads.
        head_k_dim: Per-key-head dimension.
        num_v_heads: Number of value heads.
        head_v_dim: Per-value-head dimension.
        expand_ratio: ``num_v_heads // num_k_heads``; when ``> 1`` key/query heads are repeated to match.
        conv_output_dtype: Dtype the depthwise conv output is produced in.
        gdr_op: Callable gated delta-rule op invoked per slot; must return an object exposing
            ``recurrent_state`` and ``attention_outputs`` arrays.
        **_unused: Extra keyword arguments accepted and ignored so the same fixture dict can be splatted into
            both this reference and the production helper (e.g. ``ragged_gdr_op``).

    Returns:
        Tuple ``(updated_conv_states, updated_recurrent_states, token_outputs)`` where the first two share the
        input shapes/dtypes and ``token_outputs`` has shape ``(seq_len, num_v_heads, head_v_dim)`` in float32,
        zero for tokens belonging to skipped/inactive slots.
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
    """Build a packed-update fixture for a decode-like schedule (one token per active request).

    Creates 4 slots with 3 active requests, each contributing a single token, plus a trailing empty span.
    The returned dict can be splatted directly into ``_apply_qwen3_next_packed_updates`` and
    ``_reference_packed_updates``.

    Args:
        dtype: Output dtype the sampled float32 tensors are cast to (default ``jnp.bfloat16``).

    Returns:
        Dict of keyword arguments for the packed-update helpers. Notable entries: ``conv_states``
        ``(4, conv_dim, 3)``, ``recurrent_states`` ``(4, 6, 4, 5)``, ``conv_input`` ``(1, 5, conv_dim)``,
        ``beta``/``decay`` ``(1, 5, 6)``, ``kernel`` ``(conv_dim, 3)``, ``query_start_loc`` ``[0, 1, 2, 3, 3]``,
        ``num_requests`` scalar 3, plus the scalar dimension descriptors (``key_dim``, ``num_k_heads``,
        ``head_k_dim``, ``num_v_heads``, ``head_v_dim``, ``expand_ratio``, ``conv_output_dtype``), where
        ``conv_dim = key_dim * 2 + num_v_heads * head_v_dim``.
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
    """Build a packed-update fixture for a mixed prefill+decode schedule (varying request lengths).

    Creates 4 slots with 3 active requests whose token spans differ (1, 3, and 2 tokens) so the schedule
    exercises both single-token decode and multi-token prefill request shapes in the same batch.

    Args:
        dtype: Output dtype the sampled float32 tensors are cast to (default ``jnp.bfloat16``).

    Returns:
        Dict of keyword arguments for the packed-update helpers, identical in structure to
        :func:`_make_packed_decode_inputs` but with ``seq_len`` 6 and
        ``query_start_loc`` ``[0, 1, 4, 6, 6]`` (request lengths 1, 3, 2).
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
    """Build a packed-update fixture with many equal-length prefill requests.

    Creates 16 slots, each an active request of 3 tokens (48 packed tokens total), to stress the per-request
    loop / vectorized path with a large number of uniformly sized prefills.

    Args:
        dtype: Output dtype the sampled float32 tensors are cast to (default ``jnp.bfloat16``).

    Returns:
        Dict of keyword arguments for the packed-update helpers, identical in structure to
        :func:`_make_packed_decode_inputs` but with 16 slots, ``seq_len`` 48, ``num_requests`` 16, and
        ``query_start_loc`` = ``[0, 3, 6, ..., 48]`` (evenly spaced every 3 tokens).
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
    """Build a packed-update fixture for a large padded decode bucket (one token per request, padded).

    Creates 8 active single-token requests laid out inside a padded ``bucket``-sized token dimension, so the
    conv input/beta/decay are padded to ``bucket`` even though only the first ``num_slots`` positions carry
    real tokens (``query_start_loc`` = ``[0, 1, 2, ..., 8]``). Used to verify the production helper ignores
    padding correctly.

    Args:
        dtype: Output dtype the sampled float32 tensors are cast to (default ``jnp.bfloat16``).
        bucket: Padded token-dimension length of ``conv_input``/``beta``/``decay`` (default 512).

    Returns:
        Dict of keyword arguments for the packed-update helpers, identical in structure to
        :func:`_make_packed_decode_inputs` but with 8 slots, ``num_requests`` 8, the token dimension padded
        to ``bucket``, and ``query_start_loc`` = ``arange(num_slots + 1)``.
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
    """Build a packed-update fixture for a single prefill that partially fills a padded bucket.

    Creates 4 slots with a single active request that spans ``actual_tokens`` of a ``bucket``-sized padded
    token dimension (``query_start_loc`` = ``[0, actual_tokens, actual_tokens, actual_tokens, actual_tokens]``).
    Used to verify the unified path handles a partially filled bucket and that output past ``actual_tokens``
    stays zero.

    Args:
        dtype: Output dtype the sampled float32 tensors are cast to (default ``jnp.bfloat16``).
        bucket: Padded token-dimension length of ``conv_input``/``beta``/``decay`` (default 512).
        actual_tokens: Number of real tokens for the single active request (default 454).

    Returns:
        Dict of keyword arguments for the packed-update helpers, identical in structure to
        :func:`_make_packed_decode_inputs` but with 4 slots, ``num_requests`` 1, the token dimension padded
        to ``bucket``, and the single request spanning ``[0, actual_tokens)``.
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
    """Build a tensor-parallel-shaped grouped single-step GDR decode fixture.

    Generates inputs sized for a realistic Qwen3-Next decode head layout (4 key heads, expand ratio 4 -> 16
    value heads, head/value dim 128) in the grouped (head, expand) layout expected by the grouped decode op
    and by the TP sharding tests. Unlike :func:`_make_decode_inputs` there is no explicit time axis; the head
    and expand-ratio dimensions are kept separate.

    Args:
        dtype: Output dtype the sampled float32 tensors are cast to (default ``jnp.bfloat16``).
        batch: Batch (number of concurrent decode requests) dimension (default 8).

    Returns:
        Tuple ``(query, key, value, beta, decay, recurrent_state)`` with shapes
        ``query``/``key`` ``(batch, 4, 128)``, ``value`` ``(batch, 4, 4, 128)``,
        ``beta``/``decay`` ``(batch, 4, 4)``, and ``recurrent_state`` ``(batch, 16, 128, 128)``, all of
        dtype ``dtype``.
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
    """Construct the SPMD device mesh used by the packed/grouped GDR helpers under test.

    Builds a throwaway :class:`Qwen3NextConfig` with the requested sharding axis sizes on the default JAX
    backend and returns its derived mesh.

    Args:
        axis_dims: Sharding sizes for the config's six mesh axes (``pp, dp, fsdp, ep, tp, sp``-style ordering);
            ``-1`` means "fill remaining devices". Defaults to ``(1, 1, -1, 1, 1, 1)`` (data/fsdp axis spans
            all devices, tensor-parallel axis size 1).

    Returns:
        The :class:`SpxMesh` produced by the config for the requested axis dimensions.
    """
    return Qwen3NextConfig(
        sharding_axis_dims=axis_dims,
        backend=jax.default_backend(),
    ).mesh


def _make_gdr_op(mesh: SpxMesh, runtime_dtype=jnp.bfloat16, axis_dims: tuple[int, ...] = (1, 1, -1, 1, 1, 1)):
    """Construct a :class:`GatedDeltaRuleOp` bound to the given mesh for use in the tests.

    Builds a :class:`Qwen3NextConfig` to source the partition axis / base config, then wraps everything in an
    :class:`OperationMetadata` and returns the configured ``GatedDeltaRuleOp`` operation object that the
    production decode helpers call.

    Args:
        mesh: The :class:`SpxMesh` the operation is bound to (stored as ``_stored_mesh`` in the metadata).
        runtime_dtype: Compute dtype the operation runs in (default ``jnp.bfloat16``). Distinct from the
            softmax dtype, which is fixed to float32.
        axis_dims: Sharding axis sizes passed to the throwaway config used to derive the partition axis and
            base config (default ``(1, 1, -1, 1, 1, 1)``).

    Returns:
        A :class:`GatedDeltaRuleOp` instance configured with the supplied mesh, dtype, and partition axis on
        the default JAX backend.
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
    """Assert grouped single-step GDR equals the head-repeating legacy kernel when decay is active.

    Runs both :func:`_legacy_single_step` (which repeats key/query heads to value-head count) and
    ``apply_grouped_single_step_gdr`` on the same decode fixture with a non-trivial decay term, and checks the
    grouped path preserves input dtypes for output/state and matches the legacy output and updated recurrent
    state within ``rtol=0.02, atol=0.05`` (bf16 tolerance).
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
    """Assert grouped single-step GDR equals the head-repeating legacy kernel when decay is disabled.

    Same equivalence check as :func:`test_grouped_single_step_gdr_matches_repeated_heads_with_decay` but with
    ``decay=None`` passed to both implementations, verifying the no-decay code path also matches output and
    updated recurrent state within ``rtol=0.02, atol=0.05`` and preserves input dtypes.
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
    """Assert the grouped decode honors the op's runtime dtype for the state while keeping the output dtype.

    Feeds float32 inputs through ``apply_grouped_single_step_gdr`` with a ``GatedDeltaRuleOp`` whose
    ``runtime_dtype`` is bf16, under an active mesh. Verifies the attention output keeps float32 (matching the
    query dtype) while the returned recurrent state is cast to the op's bf16 runtime dtype.
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
    """Assert packed conv1d+GDR updates match the straight-line reference for a decode-like schedule.

    Runs ``_apply_qwen3_next_packed_updates`` and :func:`_reference_packed_updates` on the
    :func:`_make_packed_decode_inputs` fixture (3 active single-token slots out of 4). Checks the updated conv
    states, recurrent states, and token outputs match within ``rtol=0.02, atol=0.05``, and additionally that
    inactive slot 3's conv/recurrent states are left byte-for-byte unchanged and its token output is exactly
    zero.
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
    """Assert packed updates match the reference for a large padded decode bucket.

    Runs ``_apply_qwen3_next_packed_updates`` and :func:`_reference_packed_updates` on the
    :func:`_make_large_bucket_decode_inputs` fixture (8 active single-token slots in a 512-padded bucket) and
    checks updated conv states, recurrent states, and token outputs all match within ``rtol=0.02, atol=0.05``,
    confirming padding past the active tokens does not perturb the result.
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
    """Assert the dispatcher falls back to the unified path when ragged GDR is disabled.

    Passes ``use_ragged_gdr=False`` and, under inference mode, runs the public dispatcher
    ``_apply_qwen3_next_packed_updates`` alongside the unified implementation
    ``_apply_qwen3_next_packed_updates_unified`` and the Python reference on a partial 512-bucket prefill
    (454 real tokens). Verifies the dispatched output equals the unified output (proving it took the unified
    branch), both match the reference within a slightly looser ``rtol=0.03, atol=0.06``, and that token output
    past the 454th position is exactly zero.
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
    """Assert the dispatcher selects the ragged path when ragged GDR is enabled for a decode bucket.

    Monkeypatches both the unified and ragged packed-update implementations
    with sentinel functions returning distinguishable marker tuples (all-1s vs all-2s). Under inference mode it
    calls the dispatcher and asserts the returned marker is the ragged one (value 2), proving the ragged branch
    was taken for a large padded decode bucket.

    Args:
        monkeypatch: Pytest fixture used to replace the unified/ragged
            implementation functions on the modeling module with sentinels.
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
    """Assert ``_preserve_array_sharding`` applies the expected named sharding to an array.

    Builds a single-device mesh with a ``data`` axis, a :class:`PartitionManager` whose batch axis maps to
    ``data`` and whose head axis is unsharded, and passes a 4-D zeros array through ``_preserve_array_sharding``.
    Verifies the resulting array's sharding equals the reference
    ``NamedSharding(mesh, PartitionSpec("data", None, None, None))``.
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
    """Assert packed updates match the reference for a mixed prefill+decode schedule, eager and jitted.

    On the :func:`_make_mixed_packed_inputs` fixture (request lengths 1, 3, 2) this runs three paths: the eager
    ``_apply_qwen3_next_packed_updates``, a ``jax.jit``-wrapped version of the same call, and the Python
    reference. It checks that both the eager and jitted updated conv states, recurrent states, and token
    outputs match the reference within ``rtol=0.02, atol=0.05``. It further asserts inactive slot 3's conv and
    recurrent states are byte-for-byte unchanged and token outputs past the active span (index 6 onward) are
    exactly zero.
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
    """Assert packed updates match the reference for many equal-length prefill requests.

    Runs ``_apply_qwen3_next_packed_updates`` and :func:`_reference_packed_updates` on the
    :func:`_make_many_prefill_packed_inputs` fixture (16 active 3-token requests) and checks the updated conv
    states, recurrent states, and token outputs all match within ``rtol=0.02, atol=0.05``.
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
    """Assert the runtime mesh / GDR op wire the head dimension onto the tensor-parallel axis (TPU-only).

    Skipped unless running on a TPU with at least 4 devices. Builds a mesh with the tensor-parallel axis sized
    4, queries the op's decode mode and BTHD shardings, and asserts the mesh exposes ``tp`` size 4 and that the
    query sharding places the head axis (index 2) on the ``tp`` axis.
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
    """Assert the Pallas grouped-decode kernel matches the JAX reference under TP shard_map (TPU-only).

    Skipped unless running on a TPU with at least 4 devices. On a tensor-parallel mesh (tp=4), runs
    ``gated_delta_rule_grouped_decode`` with ``platform="pallas"`` under explicit ``in_specs``/``out_specs``
    that shard the head axis across ``tp``, and compares against ``GatedDeltaRuleOp.grouped_gdr_decode_jax``.
    Asserts the Pallas output and updated state match the JAX path within ``rtol=0.02, atol=0.05``.
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
