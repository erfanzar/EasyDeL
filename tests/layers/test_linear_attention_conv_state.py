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
from flax import nnx as nn

from easydel.layers.linear_attention import apply_conv_with_state, apply_mask_to_padding_states


def _make_depthwise_conv(
    *,
    in_features: int = 4,
    kernel_size: int = 4,
    dtype: jnp.dtype = jnp.float32,
) -> nn.Conv:
    return nn.Conv(
        in_features=in_features,
        out_features=in_features,
        kernel_size=(kernel_size,),
        feature_group_count=in_features,
        use_bias=False,
        dtype=dtype,
        param_dtype=dtype,
        rngs=nn.Rngs(0),
    )


def test_apply_conv_with_state_decode_matches_manual_cached_depthwise_conv():
    conv = _make_depthwise_conv()
    x = jax.random.normal(jax.random.key(0), (2, 1, 4), dtype=jnp.float32)
    conv_state = jax.random.normal(jax.random.key(1), (2, 4, 4), dtype=jnp.float32)

    output, new_state = apply_conv_with_state(
        x,
        conv,
        conv_state,
        is_inference=True,
        d_conv=4,
        output_dtype=jnp.float32,
    )

    expected_state = jnp.concatenate((conv_state[:, :, 1:], x[:, 0, :][..., None]), axis=-1)
    kernel = conv.kernel.value.squeeze(1).T
    expected_output = jax.nn.silu(jnp.sum(expected_state * kernel[None, :, :], axis=-1))[:, None, :]

    assert jnp.allclose(new_state, expected_state)
    assert jnp.allclose(output, expected_output)


def test_apply_conv_with_state_prefill_reuses_partial_state_when_requested():
    conv = _make_depthwise_conv()
    x = jax.random.normal(jax.random.key(2), (2, 2, 4), dtype=jnp.float32)
    conv_state = jax.random.normal(jax.random.key(3), (2, 4, 4), dtype=jnp.float32)

    output, new_state = apply_conv_with_state(
        x,
        conv,
        conv_state,
        is_inference=False,
        d_conv=4,
        output_dtype=jnp.float32,
        reuse_partial_state=True,
    )

    expected_output = jax.nn.silu(conv(x))
    expected_state = jnp.concatenate((conv_state[:, :, 2:], x.transpose(0, 2, 1)), axis=-1)

    assert jnp.allclose(output, expected_output)
    assert jnp.allclose(new_state, expected_state)


def test_apply_conv_with_state_prefill_zero_pads_without_state_reuse():
    conv = _make_depthwise_conv()
    x = jax.random.normal(jax.random.key(4), (2, 2, 4), dtype=jnp.float32)
    conv_state = jax.random.normal(jax.random.key(5), (2, 4, 4), dtype=jnp.float32)

    output, new_state = apply_conv_with_state(
        x,
        conv,
        conv_state,
        is_inference=False,
        d_conv=4,
        output_dtype=jnp.float32,
        reuse_partial_state=False,
    )

    expected_output = jax.nn.silu(conv(x))
    expected_state = jnp.pad(x.transpose(0, 2, 1), ((0, 0), (0, 0), (2, 0)))

    assert jnp.allclose(output, expected_output)
    assert jnp.allclose(new_state, expected_state)


def test_apply_conv_with_state_prefill_runs_activation_at_output_dtype():
    conv = _make_depthwise_conv(dtype=jnp.float32)
    x = jax.random.normal(jax.random.key(6), (2, 3, 4), dtype=jnp.float32)

    output, new_state = apply_conv_with_state(
        x,
        conv,
        conv_state=None,
        is_inference=False,
        d_conv=4,
        output_dtype=jnp.bfloat16,
    )

    expected_output = jax.nn.silu(conv(x).astype(jnp.bfloat16)).astype(jnp.bfloat16)
    expected_state = jnp.pad(x.transpose(0, 2, 1), ((0, 0), (0, 0), (1, 0)))

    assert output.dtype == jnp.bfloat16
    assert jnp.allclose(output.astype(jnp.float32), expected_output.astype(jnp.float32))
    assert jnp.allclose(new_state, expected_state)


def test_apply_mask_to_padding_states_zeroes_padded_positions():
    hidden_states = jnp.arange(16, dtype=jnp.float32).reshape(2, 4, 2)
    attention_mask = jnp.asarray([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=jnp.float32)

    masked = apply_mask_to_padding_states(hidden_states, attention_mask)

    assert jnp.array_equal(masked, hidden_states * attention_mask[:, :, None])
