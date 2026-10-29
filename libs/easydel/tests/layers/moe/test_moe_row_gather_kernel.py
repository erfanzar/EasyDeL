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

"""``moe_row_gather_kernel``: the fused-MoE dispatch/combine row gathers through the Pallas TPU kernels.

Pins, on TPU:

* ``permute`` / ``unpermute`` with ``row_gather_kernel=True`` route every token copy to the right expert row and
  combine it with the right weight (float64 reference), give the default path's values and gradients, and lower to
  the kernels;
* shapes the kernels cannot take fall back to exactly the default result;
* the fused MoE block with ``moe_row_gather_kernel`` matches the flag-off block in outputs and input gradients
  (fsdp-only, and tp with the deferred TP reduce, where the combine sees full-width rows).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.moe._communication_utils import permute, unpermute

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU row-gather kernels")

BF16, F32 = jnp.bfloat16, jnp.float32
T, K, H, E = 1024, 8, 2048, 16
AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _inputs(hidden=H, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((1, T, hidden)), F32).astype(BF16)
    logits = jnp.asarray(rng.standard_normal((T, E)), F32)
    ct = jnp.asarray(rng.standard_normal((T, hidden)), F32)
    return x, logits, ct


def _moe(x, logits, flag):
    """permute -> an expert-dependent row op -> unpermute (the fused path's dispatch/combine contract)."""
    xs, sorted_sel, weights, _gs, sorted_experts = permute(
        x, logits, num_experts_per_tok=K, num_experts=E, dtype=BF16, row_gather_kernel=flag
    )
    factor = 1.0 + sorted_experts.astype(F32)[:, None] / E
    y = (xs.astype(F32) * factor).astype(BF16)
    out = unpermute(y, sorted_sel, weights, 1, T, num_experts_per_tok=K, dtype=BF16, row_gather_kernel=flag)
    return out[0]


def _loss(flag, ct):
    return lambda x, logits: jnp.sum(_moe(x, logits, flag).astype(F32) * ct)


def _rel(a, b):
    a, b = np.asarray(jnp.asarray(a, F32), np.float64), np.asarray(jnp.asarray(b, F32), np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def test_kernel_dispatch_combine_matches_float64_routing():
    x, logits, _ = _inputs()
    out = np.asarray(jax.jit(lambda a, b: _moe(a, b, True))(x, logits).astype(F32), np.float64)
    weights, experts = jax.lax.top_k(logits, K)
    xf = np.asarray(x[0].astype(F32), np.float64)
    ref = np.zeros((T, H), np.float64)
    for j in range(K):
        factor = 1.0 + np.asarray(experts[:, j], np.float64)[:, None] / E
        routed = np.asarray(jnp.asarray(xf * factor, F32).astype(BF16).astype(F32), np.float64)
        ref += np.asarray(weights[:, j], np.float64)[:, None] * routed
    # fp32 accumulation, one bf16 rounding of the result.
    assert np.all(np.abs(out - ref) <= 2.0**-8 * np.abs(ref) + 1e-4)


def test_kernel_path_matches_default_values_and_grads():
    x, logits, ct = _inputs()
    out_def = jax.jit(lambda a, b: _moe(a, b, False))(x, logits)
    out_ker = jax.jit(lambda a, b: _moe(a, b, True))(x, logits)
    assert _rel(out_ker, out_def) < 4e-3
    (gx_def, gl_def) = jax.jit(jax.grad(_loss(False, ct), argnums=(0, 1)))(x, logits)
    (gx_ker, gl_ker) = jax.jit(jax.grad(_loss(True, ct), argnums=(0, 1)))(x, logits)
    assert np.isfinite(np.asarray(gx_ker, np.float32)).all() and np.isfinite(np.asarray(gl_ker)).all()
    assert _rel(gx_ker, gx_def) < 5e-3, _rel(gx_ker, gx_def)
    assert _rel(gl_ker, gl_def) < 5e-3, _rel(gl_ker, gl_def)


def test_kernel_path_lowers_to_the_pallas_kernels():
    x, logits, ct = _inputs()

    def custom_calls(flag):
        fn = jax.jit(jax.value_and_grad(_loss(flag, ct), argnums=(0, 1)))
        return fn.lower(x, logits).compile().as_text().count("tpu_custom_call")

    # fwd: pack + gather, pack + gather-sum; bwd: pack + gather-sum, pack + scaled gather with row dot.
    assert custom_calls(True) - custom_calls(False) >= 8


def test_unsupported_shapes_fall_back_to_the_default_result():
    x, logits, ct = _inputs(hidden=384)
    out_def = jax.jit(lambda a, b: _moe(a, b, False))(x, logits)
    out_ker = jax.jit(lambda a, b: _moe(a, b, True))(x, logits)
    np.testing.assert_array_equal(np.asarray(out_ker, np.float32), np.asarray(out_def, np.float32))
    g_def = jax.jit(jax.grad(_loss(False, ct)))(x, logits)
    g_ker = jax.jit(jax.grad(_loss(True, ct)))(x, logits)
    np.testing.assert_array_equal(np.asarray(g_ker, np.float32), np.asarray(g_def, np.float32))


# --------------------------------------------------------------------------------------------- fused MoE block
def _build(*, dims, flag: bool, defer: bool, chunk: int):
    from easydel.modules.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseBlock
    from easydel.modules.qwen3_moe.qwen3_moe_configuration import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        hidden_size=H,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        moe_method="fused_moe",
        moe_force_xla_gmm=True,
        sharding_axis_dims=dims,
        sharding_axis_names=AXIS_NAMES,
        scan_layers=False,
    )
    config.add_basic_configurations(
        fsdp_is_ep_bound=False,
        sp_is_ep_bound=False,
        moe_chunk_size=chunk,
        moe_defer_tp_reduce=defer,
        moe_row_gather_kernel=flag,
    )
    return Qwen3MoeSparseBlock(config=config, dtype=BF16, param_dtype=F32, rngs=spx.Rngs(7))


def _block_out_and_grad(block, x):
    def loss(xin):
        out, _ = block(xin)
        return jnp.sum(out.astype(F32) ** 2), out

    with block.config.mesh:
        (_, out), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(x)
        calls = jax.jit(jax.grad(lambda xin: loss(xin)[0])).lower(x).compile().as_text().count("tpu_custom_call")
    return out, grad, calls


@pytest.mark.parametrize(
    "dims,defer,shape",
    [((1, 1, 4, 1, 1, 1), False, (8, 512, H)), ((1, 1, 2, 1, 2, 1), True, (8, 256, H))],
    ids=["fsdp4", "fsdp2_tp2_defer"],
)
def test_fused_block_matches_flag_off(dims, defer, shape):
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices")
    x = jax.random.normal(jax.random.PRNGKey(0), shape, F32).astype(BF16)
    out_off, grad_off, calls_off = _block_out_and_grad(_build(dims=dims, flag=False, defer=defer, chunk=1024), x)
    out_on, grad_on, calls_on = _block_out_and_grad(_build(dims=dims, flag=True, defer=defer, chunk=1024), x)
    assert calls_on > calls_off, (calls_on, calls_off)
    assert np.isfinite(np.asarray(out_on, np.float32)).all() and np.isfinite(np.asarray(grad_on, np.float32)).all()
    assert _rel(out_on, out_off) < 1e-2, _rel(out_on, out_off)
    assert _rel(grad_on, grad_off) < 2e-2, _rel(grad_on, grad_off)
