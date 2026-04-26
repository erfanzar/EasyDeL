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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.gated_delta_rule._xla_impl_fwd import _single_step_gdr_fwd
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax import PartitionAxis, PartitionManager

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
    rng = jax.random.key(0)
    query = jax.random.normal(rng, (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    value = jax.random.normal(jax.random.fold_in(rng, 2), (2, 1, 6, 5), dtype=jnp.float32).astype(dtype)
    beta = jax.random.normal(jax.random.fold_in(rng, 3), (2, 1, 6), dtype=jnp.float32).astype(dtype)
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (2, 1, 6), dtype=jnp.float32).astype(dtype)
    recurrent_state = jax.random.normal(jax.random.fold_in(rng, 5), (2, 6, 4, 5), dtype=jnp.float32).astype(dtype)
    return query, key, value, beta, decay, recurrent_state


def _legacy_single_step(query, key, value, beta, decay, recurrent_state):
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
    """Straight-line per-request reference for packed Qwen3-Next state updates."""
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


def _make_runtime_mesh(axis_dims: tuple[int, ...] = (1, 1, -1, 1, 1, 1)) -> Mesh:
    return Qwen3NextConfig(
        sharding_axis_dims=axis_dims,
        backend=jax.default_backend(),
    ).mesh


def _make_gdr_op(mesh: Mesh, runtime_dtype=jnp.bfloat16, axis_dims: tuple[int, ...] = (1, 1, -1, 1, 1, 1)):
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


def test_packed_updates_use_unified_when_ragged_disabled_for_partial_prefill_bucket(monkeypatch):
    packed_inputs = _make_partial_bucket_prefill_inputs(bucket=512, actual_tokens=454)
    mesh = _make_runtime_mesh()

    monkeypatch.setenv("EASYDEL_RAGGED_GDR", "0")

    with mesh, set_inference_mode(True):
        gdr_op = _make_gdr_op(mesh)
        dispatched_conv, dispatched_rec, dispatched_out = _apply_qwen3_next_packed_updates(
            **packed_inputs,
            gdr_op=gdr_op,
            ragged_gdr_op=object(),
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
    packed_inputs = _make_large_bucket_decode_inputs(bucket=512)

    monkeypatch.setenv("EASYDEL_RAGGED_GDR", "1")

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
    mesh = _make_runtime_mesh((1, 1, 1, 1, 4, 1))
    query, key, value, beta, decay, recurrent_state = _make_tp_grouped_decode_inputs()

    with mesh:
        gdr_op = _make_gdr_op(mesh, axis_dims=(1, 1, 1, 1, 4, 1))
        pallas_out, pallas_state = gdr_op.grouped_gdr_decode_shard_map_pallas(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            recurrent_state=recurrent_state,
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
