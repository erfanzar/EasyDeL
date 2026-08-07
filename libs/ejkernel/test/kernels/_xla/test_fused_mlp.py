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

"""Fused MLP: forward parity, gradient parity, formats, layouts.

References are naive compositions built from scratch in each test — never the
implementation under test — so a broken vjp rule or a scale on the wrong axis
fails instead of cancelling.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import fused_mlp, split_gate_up

K_DIM, I_DIM, TOKENS = 256, 512, 16


def _weights(rng, scale=0.05):
    """Dense float32 gate/up/down weights."""
    return (
        rng.normal(size=(K_DIM, I_DIM)).astype(np.float32) * scale,
        rng.normal(size=(K_DIM, I_DIM)).astype(np.float32) * scale,
        rng.normal(size=(I_DIM, K_DIM)).astype(np.float32) * scale,
    )


def _quantize_channelwise(weight, qdtype):
    """Symmetric per-output-channel quantization."""
    qmax = float(jnp.iinfo(qdtype).max)
    absmax = np.abs(weight).max(axis=0, keepdims=True)
    scale = absmax / qmax
    codes = np.clip(np.round(weight / scale), -qmax, qmax)
    return jnp.asarray(codes).astype(qdtype), jnp.asarray(scale, jnp.float32)


def _naive(x, w_gate, w_up, w_down, activation=jax.nn.silu):
    """Ground-truth MLP in float32."""
    x32 = np.asarray(x, np.float32)
    hidden = np.asarray(activation(jnp.asarray(x32 @ w_gate)), np.float32) * (x32 @ w_up)
    return hidden @ w_down


class TestDenseForwardBackward:
    """bf16 dense format against naive autodiff."""

    def test_forward_matches_naive(self):
        """Forward is the plain composition."""
        rng = np.random.default_rng(0)
        w_gate, w_up, w_down = _weights(rng)
        x = jnp.asarray(rng.normal(size=(TOKENS, K_DIM)), jnp.bfloat16)
        got = np.asarray(
            fused_mlp(x, jnp.asarray(w_gate, jnp.bfloat16), jnp.asarray(w_up, jnp.bfloat16), jnp.asarray(w_down, jnp.bfloat16)),
            np.float32,
        )
        ref = _naive(x, w_gate, w_up, w_down)
        rel = np.abs(got - ref).max() / np.abs(ref).max()
        assert rel < 0.02, rel

    def test_gradients_match_naive_autodiff(self):
        """All four cotangents (dWg, dWu, dWd, dx) track naive autodiff."""
        rng = np.random.default_rng(1)
        w_gate, w_up, w_down = (jnp.asarray(w, jnp.bfloat16) for w in _weights(rng))
        x = jnp.asarray(rng.normal(size=(TOKENS, K_DIM)), jnp.bfloat16)

        def naive_loss(params, x):
            wg, wu, wd = params
            hidden = jax.nn.silu((x @ wg).astype(jnp.float32)) * (x @ wu).astype(jnp.float32)
            return jnp.sum((hidden.astype(x.dtype) @ wd).astype(jnp.float32) ** 2)

        def fused_loss(params, x):
            wg, wu, wd = params
            return jnp.sum(fused_mlp(x, wg, wu, wd).astype(jnp.float32) ** 2)

        naive_grads = jax.grad(naive_loss)((w_gate, w_up, w_down), x)
        fused_grads = jax.grad(fused_loss)((w_gate, w_up, w_down), x)
        for name, ref, got in zip(("dWg", "dWu", "dWd"), naive_grads, fused_grads, strict=True):
            rel = float(
                jnp.abs(ref.astype(jnp.float32) - got.astype(jnp.float32)).max()
                / (jnp.abs(ref.astype(jnp.float32)).max() + 1e-9)
            )
            assert rel < 0.03, f"{name} diverged: {rel}"

        dx_ref = jax.grad(lambda x: naive_loss((w_gate, w_up, w_down), x))(x)
        dx_got = jax.grad(lambda x: fused_loss((w_gate, w_up, w_down), x))(x)
        rel = float(
            jnp.abs(dx_ref.astype(jnp.float32) - dx_got.astype(jnp.float32)).max()
            / jnp.abs(dx_ref.astype(jnp.float32)).max()
        )
        assert rel < 0.03, f"dx diverged: {rel}"


class TestQuantizedFormats:
    """Channelwise int8/int4 weights: forward parity and frozen-weight grads."""

    @pytest.mark.parametrize("qdtype", [jnp.int8, jnp.int4])
    def test_forward_matches_dequantized_naive(self, qdtype):
        """Quantized forward equals the naive product of dequantized weights."""
        rng = np.random.default_rng(2)
        dense = _weights(rng)
        quantized = [_quantize_channelwise(w, qdtype) for w in dense]
        x = jnp.asarray(rng.normal(size=(TOKENS, K_DIM)), jnp.bfloat16)

        got = np.asarray(
            fused_mlp(
                x,
                quantized[0][0],
                quantized[1][0],
                quantized[2][0],
                gate_scale=quantized[0][1],
                up_scale=quantized[1][1],
                down_scale=quantized[2][1],
            ),
            np.float32,
        )
        dequant = [np.asarray(c, np.float32) * np.asarray(s) for c, s in quantized]
        ref = _naive(x, *dequant)
        rel = np.abs(got - ref).max() / np.abs(ref).max()
        assert rel < 0.02, rel

    def test_dx_matches_dequantized_autodiff_and_weights_are_frozen(self):
        """dx flows exactly through dequantized weights; codes get zero grads."""
        rng = np.random.default_rng(3)
        dense = _weights(rng)
        quantized = [_quantize_channelwise(w, jnp.int8) for w in dense]
        x = jnp.asarray(rng.normal(size=(TOKENS, K_DIM)), jnp.bfloat16)
        dequant = [jnp.asarray(np.asarray(c, np.float32) * np.asarray(s), jnp.bfloat16) for c, s in quantized]

        def quant_loss(x):
            return jnp.sum(
                fused_mlp(
                    x,
                    quantized[0][0],
                    quantized[1][0],
                    quantized[2][0],
                    gate_scale=quantized[0][1],
                    up_scale=quantized[1][1],
                    down_scale=quantized[2][1],
                ).astype(jnp.float32)
                ** 2
            )

        def dense_loss(x):
            return jnp.sum(fused_mlp(x, *dequant).astype(jnp.float32) ** 2)

        dx_quant = jax.grad(quant_loss)(x)
        dx_dense = jax.grad(dense_loss)(x)
        rel = float(
            jnp.abs(dx_quant.astype(jnp.float32) - dx_dense.astype(jnp.float32)).max()
            / jnp.abs(dx_dense.astype(jnp.float32)).max()
        )
        assert rel < 0.03, f"frozen-weight dx diverged: {rel}"


class TestLayouts:
    """Fused gate_up layouts are normalized before kernels see them."""

    def test_concat_layout_matches_separate(self):
        """A [K, 2I] concat weight equals separate gate/up."""
        rng = np.random.default_rng(4)
        w_gate, w_up, w_down = (jnp.asarray(w, jnp.bfloat16) for w in _weights(rng))
        x = jnp.asarray(rng.normal(size=(TOKENS, K_DIM)), jnp.bfloat16)
        fused = jnp.concatenate([w_gate, w_up], axis=1)
        np.testing.assert_array_equal(
            np.asarray(fused_mlp(x, gate_up=fused, w_down=w_down)),
            np.asarray(fused_mlp(x, w_gate, w_up, w_down)),
        )

    def test_interleaved_layout_round_trips(self):
        """TP-interleaved segments split back to the original gate/up."""
        rng = np.random.default_rng(5)
        w_gate, w_up, _ = _weights(rng)
        segments = 4
        seg_i = I_DIM // segments
        pieces = []
        for s in range(segments):
            pieces.append(w_gate[:, s * seg_i : (s + 1) * seg_i])
            pieces.append(w_up[:, s * seg_i : (s + 1) * seg_i])
        interleaved = jnp.asarray(np.concatenate(pieces, axis=1))
        gate, up = split_gate_up(interleaved, layout="interleaved", segments=segments)
        np.testing.assert_allclose(np.asarray(gate), w_gate, rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(up), w_up, rtol=0, atol=0)

    def test_conflicting_weight_arguments_raise(self):
        """gate_up and separate gate/up are mutually exclusive."""
        rng = np.random.default_rng(6)
        w_gate, w_up, w_down = (jnp.asarray(w, jnp.bfloat16) for w in _weights(rng))
        x = jnp.ones((4, K_DIM), jnp.bfloat16)
        with pytest.raises(ValueError, match="not both"):
            fused_mlp(x, w_gate, w_up, w_down, gate_up=jnp.concatenate([w_gate, w_up], axis=1))


class TestActivations:
    """Every table activation runs and matches its naive composition."""

    @pytest.mark.parametrize("name,fn", [
        ("silu", jax.nn.silu),
        ("gelu", jax.nn.gelu),
        ("gelu_tanh", lambda v: jax.nn.gelu(v, approximate=True)),
        ("relu", jax.nn.relu),
        ("sigmoid", jax.nn.sigmoid),
    ])
    def test_activation_matches_naive(self, name, fn):
        """Forward under each activation equals the naive composition."""
        rng = np.random.default_rng(7)
        w_gate, w_up, w_down = _weights(rng)
        x = jnp.asarray(rng.normal(size=(TOKENS, K_DIM)), jnp.bfloat16)
        got = np.asarray(
            fused_mlp(
                x,
                jnp.asarray(w_gate, jnp.bfloat16),
                jnp.asarray(w_up, jnp.bfloat16),
                jnp.asarray(w_down, jnp.bfloat16),
                activation=name,
            ),
            np.float32,
        )
        ref = _naive(x, w_gate, w_up, w_down, activation=fn)
        rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9)
        assert rel < 0.03, f"{name}: {rel}"
