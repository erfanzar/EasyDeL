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

"""``moe_defer_tp_reduce``: the chunked fused-MoE TP reduction after the top-k combine.

The default chunked dispatch ``psum_scatter``s the down-projection output over TP on the top-k-expanded
``[chunk * k, H]`` rows and then combines the k expert rows per token. The combine is linear, so reducing the
combined ``[chunk, H]`` output instead is the same value with ``1 / k`` of the reduce-scatter bytes. These tests pin:

* outputs and input gradients equal the default path (only the summation order differs),
* the deferred path reduce-scatters ``[chunk, H]`` rows and never the expanded ``[chunk * k, H]`` ones,
* the flag is inert (bit-identical) where it cannot apply (tp == 1).
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
HIDDEN, TOPK, CHUNK = 32, 2, 4


def _build(*, dims, defer: bool, chunk: int = CHUNK):
    """Bias-free Qwen3-MoE fused block with chunked dispatch and fsdp kept off the expert axis (ep == 1)."""
    from easydel.modules.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseBlock
    from easydel.modules.qwen3_moe.qwen3_moe_configuration import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_experts=8,
        num_experts_per_tok=TOPK,
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
    )
    return Qwen3MoeSparseBlock(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(7))


def _tp2_dims():
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices for fsdp2 x tp2")
    return (1, 1, 2, 1, 2, 1)


def _x():
    return jax.random.normal(jax.random.PRNGKey(0), (8, 4, HIDDEN), dtype=jnp.float32)


def _out_and_grad(block, x):
    def loss(xin):
        out, _ = block(xin)
        return jnp.sum(out * out), out

    with jax.default_matmul_precision("highest"), block.config.mesh:
        (_, out), grad = jax.value_and_grad(loss, has_aux=True)(x)
    return np.asarray(out, np.float64), np.asarray(grad, np.float64)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def test_deferred_reduce_matches_default_outputs_and_grads():
    dims = _tp2_dims()
    x = _x()
    out_ref, grad_ref = _out_and_grad(_build(dims=dims, defer=False), x)
    out_def, grad_def = _out_and_grad(_build(dims=dims, defer=True), x)
    assert np.isfinite(out_def).all() and np.isfinite(grad_def).all()
    assert _rel(out_def, out_ref) < 1e-5, _rel(out_def, out_ref)
    assert _rel(grad_def, grad_ref) < 1e-5, _rel(grad_def, grad_ref)


def _reduce_scatter_operand_rows(block, x):
    """Leading dims of every reduce-scatter operand in the lowered forward."""
    with block.config.mesh:
        text = jax.jit(lambda xin: block(xin)[0]).lower(x).as_text()
    # The operand type follows the op's reduction region, so match across lines up to the region's close.
    pattern = r"stablehlo\.reduce_scatter.*?\}\)\s*:\s*\(tensor<([0-9x]+)xf32>\)"
    return {int(m.group(1).split("x")[0]) for m in re.finditer(pattern, text, flags=re.S)}


def test_deferred_reduce_scatters_combined_rows():
    dims = _tp2_dims()
    x = _x()
    default_rows = _reduce_scatter_operand_rows(_build(dims=dims, defer=False), x)
    deferred_rows = _reduce_scatter_operand_rows(_build(dims=dims, defer=True), x)
    assert CHUNK * TOPK in default_rows, f"default path should reduce [chunk*k, H] rows, saw {default_rows}"
    assert CHUNK in deferred_rows, f"deferred path should reduce [chunk, H] rows, saw {deferred_rows}"
    assert CHUNK * TOPK not in deferred_rows, f"deferred path still reduces expanded rows: {deferred_rows}"


def test_flag_is_inert_without_tensor_parallelism():
    if jax.device_count() < 2:
        pytest.skip("needs 2 devices")
    dims = (1, 1, 2, 1, 1, 1)
    x = _x()
    out_ref, grad_ref = _out_and_grad(_build(dims=dims, defer=False), x)
    out_def, grad_def = _out_and_grad(_build(dims=dims, defer=True), x)
    np.testing.assert_array_equal(out_def, out_ref)
    np.testing.assert_array_equal(grad_def, grad_ref)
