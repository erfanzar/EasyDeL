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
import spectrax as spx
from easydel.layers.linear_attention import apply_conv_with_state, apply_mask_to_padding_states
from spectrax import nn


def _make_depthwise_conv(
    *,
    in_features: int = 4,
    kernel_size: int = 4,
    dtype: jnp.dtype = jnp.float32,
    causal: bool = False,
) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels=in_features,
        out_channels=in_features,
        kernel_size=kernel_size,
        groups=in_features,
        use_bias=False,
        dtype=dtype,
        # Production callers left-pad by kernel_size - 1 (causal, output length == input length).
        padding=((kernel_size - 1, 0),) if causal else "VALID",
        rngs=spx.Rngs(0),
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
    kernel = conv.weight.value.squeeze(1).T
    expected_output = jax.nn.silu(jnp.sum(expected_state * kernel[None, :, :], axis=-1))[:, None, :]

    assert jnp.allclose(new_state, expected_state)
    assert jnp.allclose(output, expected_output)


def test_apply_conv_with_state_decode_padding_token_keeps_window():
    """A cached singleton padding token (segment id -1) keeps the conv window and emits zeros (r10 S08-1)."""
    conv = _make_depthwise_conv()
    x = jax.random.normal(jax.random.key(2), (2, 1, 4), dtype=jnp.float32)
    conv_state = jax.random.normal(jax.random.key(3), (2, 4, 4), dtype=jnp.float32)
    kwargs = dict(is_inference=True, d_conv=4, output_dtype=jnp.float32)

    plain_out, plain_state = apply_conv_with_state(x, conv, conv_state, **kwargs)
    out, state = apply_conv_with_state(
        x, conv, conv_state, segment_ids=jnp.asarray([[-1], [0]], dtype=jnp.int32), **kwargs
    )

    assert jnp.array_equal(out[0], jnp.zeros_like(out[0]))
    assert jnp.array_equal(state[0], conv_state[0])
    assert jnp.allclose(out[1], plain_out[1])
    assert jnp.allclose(state[1], plain_state[1])
    assert not jnp.allclose(plain_state[0], conv_state[0])  # negative control


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

    expected_output = jax.nn.silu(conv(jnp.concatenate((conv_state[:, :, 1:].transpose(0, 2, 1), x), axis=1)))[
        :, -x.shape[1] :, :
    ]
    expected_state = jnp.concatenate((conv_state[:, :, 2:], x.transpose(0, 2, 1)), axis=-1)

    assert jnp.allclose(output, expected_output)
    assert jnp.allclose(new_state, expected_state)


def test_apply_conv_with_state_prefill_uses_cached_context_without_partial_reuse():
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

    expected_output = jax.nn.silu(conv(jnp.concatenate((conv_state[:, :, 1:].transpose(0, 2, 1), x), axis=1)))[
        :, -x.shape[1] :, :
    ]
    expected_state = jnp.concatenate((conv_state[:, :, 2:], x.transpose(0, 2, 1)), axis=-1)

    assert jnp.allclose(output, expected_output)
    assert jnp.allclose(new_state, expected_state)


def test_apply_conv_with_state_split_prefill_matches_unsplit_and_preserves_state():
    conv = _make_depthwise_conv(causal=True)
    x = jax.random.normal(jax.random.key(9), (2, 12, 4), dtype=jnp.float32)
    full, full_state = apply_conv_with_state(x, conv, None, is_inference=False, d_conv=4, output_dtype=jnp.float32)

    for reuse_partial_state in (False, True):
        first, state = apply_conv_with_state(
            x[:, :7],
            conv,
            jnp.zeros((2, 4, 4)),
            is_inference=False,
            d_conv=4,
            output_dtype=jnp.float32,
            reuse_partial_state=reuse_partial_state,
        )
        assert jnp.allclose(first, full[:, :7], atol=1e-5)
        second, final_state = apply_conv_with_state(
            x[:, 7:],
            conv,
            state,
            is_inference=False,
            d_conv=4,
            output_dtype=jnp.float32,
            reuse_partial_state=reuse_partial_state,
        )
        assert jnp.allclose(jnp.concatenate((first, second), axis=1), full, atol=1e-5)
        assert jnp.allclose(final_state, full_state)


def test_apply_conv_with_state_split_short_continuation_matches_unsplit():
    conv = _make_depthwise_conv(causal=True)
    x = jax.random.normal(jax.random.key(10), (2, 6, 4), dtype=jnp.float32)
    full, full_state = apply_conv_with_state(x, conv, None, is_inference=False, d_conv=4, output_dtype=jnp.float32)
    first, state = apply_conv_with_state(
        x[:, :4],
        conv,
        jnp.zeros((2, 4, 4)),
        is_inference=False,
        d_conv=4,
        output_dtype=jnp.float32,
    )
    second, final_state = apply_conv_with_state(
        x[:, 4:], conv, state, is_inference=False, d_conv=4, output_dtype=jnp.float32
    )
    assert jnp.allclose(jnp.concatenate((first, second), axis=1), full, atol=1e-5)
    assert jnp.allclose(final_state, full_state)


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
