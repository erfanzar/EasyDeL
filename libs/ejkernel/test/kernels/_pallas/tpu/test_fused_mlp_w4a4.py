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

"""On-device parity for the fused W4A4 MLP kernel.

The reference replicates the kernel's exact quantized semantics tile by tile
in NumPy — int4 input codes, per-channel weight scales, per-(token, tile)
hidden re-quantization — so the assertion is kernel-exactness (~bf16
rounding), not a loose accuracy bound.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.fused_mlp import fused_mlp_w4a4_pallas
from ejkernel.kernels._pallas.tpu.quantized_matmul._packed_gemv import pack_int4_adjacent
from ejkernel.modules import fused_mlp

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu",
    reason="int4 MXU feed requires TPU (CPU backend rejects int4 dot_general)",
)

K_DIM, I_DIM, TOKENS, TILE_I = 1024, 2048, 8, 512


def _quantize(rng, k_dim, n_dim, scale_factor=0.05):
    """Per-channel int4 quantization of a gaussian weight."""
    weight = rng.normal(size=(k_dim, n_dim)).astype(np.float32) * scale_factor
    scale = np.abs(weight).max(axis=0, keepdims=True) / 7.0
    codes = np.clip(np.round(weight / scale), -7, 7).astype(np.int32)
    return codes, scale


def _reference(x_codes, x_scale, gate, up, down, tile_i):
    """Tile-wise NumPy replication of the kernel's quantized semantics."""
    gate_codes, gate_scale = gate
    up_codes, up_scale = up
    down_codes, down_scale = down
    x_real = x_codes * x_scale  # [m, K]

    acc = np.zeros((x_codes.shape[0], down_codes.shape[1]), np.float32)
    for start in range(0, gate_codes.shape[1], tile_i):
        end = start + tile_i
        a = (x_real @ gate_codes[:, start:end]) * gate_scale[:, start:end]
        b = (x_real @ up_codes[:, start:end]) * up_scale[:, start:end]
        hidden = np.asarray(jax.nn.silu(jnp.asarray(a)), np.float32) * b
        habs = np.abs(hidden).max(axis=1, keepdims=True)
        h_scale = habs / 7.0
        h4 = np.clip(np.round(np.divide(hidden, h_scale, where=h_scale > 0)), -7, 7)
        acc += (h4 @ down_codes[start:end, :]) * h_scale
    return acc * down_scale


def test_kernel_matches_tilewise_reference():
    """The fused kernel is exact against its own quantization semantics."""
    rng = np.random.default_rng(0)
    gate = _quantize(rng, K_DIM, I_DIM)
    up = _quantize(rng, K_DIM, I_DIM)
    down = _quantize(rng, I_DIM, K_DIM)

    x_float = rng.normal(size=(TOKENS, K_DIM)).astype(np.float32)
    x_scale = np.abs(x_float).max(axis=1, keepdims=True) / 7.0
    x_codes = np.clip(np.round(x_float / x_scale), -7, 7)

    got = np.asarray(
        fused_mlp_w4a4_pallas(
            jnp.asarray(x_codes, jnp.int4),
            pack_int4_adjacent(jnp.asarray(gate[0])),
            pack_int4_adjacent(jnp.asarray(up[0])),
            pack_int4_adjacent(jnp.asarray(down[0])),
            jnp.asarray(gate[1]),
            jnp.asarray(up[1]),
            jnp.asarray(down[1]),
            jnp.asarray(x_scale, jnp.float32),
            activation="silu",
            tile_i=TILE_I,
        ),
        np.float32,
    )
    ref = _reference(x_codes, x_scale, gate, up, down, TILE_I)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 0.02, f"fused W4A4 kernel diverged from tile-wise reference: relerr={rel}"


def test_public_op_dispatches_to_packed_kernel():
    """The packed path is exact wiring around the kernel, and really engages.

    Asserted as: public-op output == direct kernel call with the same
    input-quantization recipe (bit-exact), and != the XLA path (which does
    not quantize activations) — accuracy itself is covered by the tile-wise
    parity test above.
    """
    rng = np.random.default_rng(1)
    gate = _quantize(rng, K_DIM, I_DIM)
    up = _quantize(rng, K_DIM, I_DIM)
    down = _quantize(rng, I_DIM, K_DIM)
    x = jnp.asarray(rng.normal(size=(TOKENS, K_DIM)), jnp.bfloat16)

    packed = tuple(pack_int4_adjacent(jnp.asarray(codes)) for codes, _ in (gate, up, down))
    common = dict(
        gate_scale=jnp.asarray(gate[1]),
        up_scale=jnp.asarray(up[1]),
        down_scale=jnp.asarray(down[1]),
    )
    weights = (jnp.asarray(gate[0], jnp.int4), jnp.asarray(up[0], jnp.int4), jnp.asarray(down[0], jnp.int4))

    via_op = np.asarray(
        fused_mlp(x, *weights, **common, packed_weights=packed, packed_tile_i=TILE_I), np.float32
    )

    x_abs = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=1, keepdims=True)
    x_scale = x_abs / 7.0
    x4 = jnp.clip(jnp.round(x.astype(jnp.float32) / jnp.where(x_scale == 0, 1, x_scale)), -7, 7).astype(jnp.int4)
    direct = np.asarray(
        fused_mlp_w4a4_pallas(
            x4, *packed, jnp.asarray(gate[1]), jnp.asarray(up[1]), jnp.asarray(down[1]),
            x_scale, activation="silu", tile_i=TILE_I,
        ).astype(x.dtype),
        np.float32,
    )
    np.testing.assert_array_equal(via_op, direct)

    xla_out = np.asarray(fused_mlp(x, *weights, **common), np.float32)
    assert np.abs(via_op - xla_out).max() > 0, "packed path did not engage (identical to XLA path)"
    assert np.isfinite(via_op).all()
