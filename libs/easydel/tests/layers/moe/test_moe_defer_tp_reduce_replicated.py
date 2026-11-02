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

"""``moe_defer_tp_reduce`` under inference mode: the single-pass replicated-expert dispatch.

With ``moe_chunk_size=None`` and ``set_inference_mode()``, ``moe_defer_tp_reduce=True`` makes the fused MoE sort each
expert shard's local (token, expert) pairs once, run the experts without the per-row TP reduction and reduce the
combined ``[tokens, H]`` partial once. On four TPU devices these tests pin:

* outputs equal the default dispatch on ep x tp meshes, in the regular (TP-scattered) and the decode-replicated
  layout: float32 activations with dense or W8A8 experts to 1e-5, and the served precision (bfloat16 activations,
  W8A8 experts) to within three bfloat16 roundings, the one numerical change being that the combined partial is
  rounded to bfloat16 for the TP reduce-scatter instead of every top-k-expanded expert row,
* outputs match an independent float64 NumPy reference of GLM-5's routed experts,
* each equivalence case really ran the deferred dispatch: its lowering differs from the default dispatch's (a gate
  fallback lowers the identical program, as the outside-inference-mode case pins), it never reduce-scatters the
  top-k-expanded ``[tokens * k, H]`` rows, it reduce-scatters ``[tokens, H]`` rows in the TP-scattered layout and it
  drops the boundary all-gather in the decode layout,
* with row windows (``moe_local_ffn_window_rows`` forced, or enough routed rows that the automatic window is shorter
  than ``tokens * k``) the dispatch computes only each shard's local rows, window by window, and stays dropless:
  several windows per shard, a padded last window and routing that sends every route to one expert shard all match
  the one-buffer deferred dispatch, the default dispatch and the float64 reference,
* outside inference mode (training) the knobs leave the lowered program and the outputs identical.
"""

from __future__ import annotations

import contextlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

AXES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
HIDDEN, INTER, EXPERTS, TOPK, TOKENS = 256, 256, 16, 4, 32
MESHES = {"ep2tp2": (1, 1, 1, 2, 2, 1), "ep4tp1": (1, 1, 1, 4, 1, 1), "ep1tp4": (1, 1, 1, 1, 4, 1)}
LAYOUTS = pytest.mark.parametrize("decode", [False, True], ids=["prefill_layout", "decode_layout"])
# bfloat16 activations: the deferred dispatch rounds each token's combined float32 partial to bfloat16 once for the
# TP reduce-scatter where the default rounds every expert row after its reduction, so the outputs differ by about one
# bfloat16 rounding (2**-8 relative); allow three.
BF16_REL_TOL = 3 * 2**-8
# Row windows. At TOKENS * TOPK = 128 rows the automatic window covers every row (no windows), so the window tests
# force one: 48 rows gives 3 windows over the 128 sorted rows (the last one padded); an ep1 / ep2 / ep4 shard holds
# 128 / about 64 / about 32 local rows and runs 3 / about 2 / about 1 of them. SKEW_TOKENS * TOPK = 4096 rows makes the
# automatic window 2048 (ep4) / 3072 (ep2) rows: a shard that receives every route needs a second window.
FORCED_WINDOW_ROWS = 48
SKEW_TOKENS = 1024
_REDUCE_SCATTER = re.compile(r"stablehlo\.reduce_scatter.*?\}\)\s*:\s*\(tensor<((?:\d+x)+)(\w+)>\)", re.S)


def _skip_without_four_devices():
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices")


def _build(dims, *, quantize: bool = False, dtype=jnp.float32):
    """GLM-5-Next routed-MoE block (no shared expert) in ``dtype`` on ``dims``; optionally W8A8 experts."""
    import easydel as ed
    from easydel.modules.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    config = ed.Glm5NextTextConfig(
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        qk_rope_head_dim=0,
        index_topk=8,
        index_kpool=4,
        moe_intermediate_size=INTER,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOPK,
        n_shared_experts=0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        swiglu_limit=10.0,
    )
    config.add_basic_configurations(sharding_axis_dims=dims, sharding_axis_names=AXES, moe_method="fused_moe")
    block = Glm5NextTextMoE(config=config, dtype=dtype, param_dtype=dtype, precision=None, rngs=spx.Rngs(3))
    for projection in (block.experts.gate_proj, block.experts.up_proj, block.experts.down_proj):
        projection.weight.value = projection.weight.value * 5
    if quantize:
        from easydel.layers.quantization import EasyQuantizer, QuantizationConfig

        quantizer = EasyQuantizer(
            quantization_config=QuantizationConfig(dtype="w8a8", pattern=r"experts\.(?:gate_proj|up_proj|down_proj)$")
        )
        block = quantizer.apply_quantization(block)
        assert block.experts.down_proj.quant_kernel.value.dtype == jnp.int8
    return block


def _x(dtype=jnp.float32, tokens: int = TOKENS):
    return jax.random.normal(jax.random.PRNGKey(0), (1, tokens, HIDDEN), dtype=jnp.float32).astype(dtype)


def _forward(block, x, *, defer: bool, inference: bool = True, decode: bool = False, window_rows=None):
    """Jitted block output (float64) and the lowered StableHLO text, with the knobs set at trace time."""
    from easydel.infra.sharding import decode_mode_specs
    from easydel.utils.inference_mode import set_inference_mode

    graphdef, state = spx.export(block)

    def fwd(st, h):
        module = spx.bind(graphdef, st)
        module.config.moe_defer_tp_reduce = defer
        module.config.moe_local_ffn_window_rows = window_rows
        return module(h)[0]

    with contextlib.ExitStack() as stack:
        stack.enter_context(jax.default_matmul_precision("highest"))
        stack.enter_context(block.config.mesh)
        if inference:
            stack.enter_context(set_inference_mode())
        stack.enter_context(decode_mode_specs(decode))
        out = jax.jit(fwd)(state, x)
        text = jax.jit(fwd).lower(state, x).as_text()
    return np.asarray(jax.device_get(out), np.float32).astype(np.float64), text


def _reference(block, x):
    """Independent float64 GLM-5 routed experts: sigmoid scores, top-k, renormalize x2.5, clamped SwiGLU."""
    xs = np.asarray(x, np.float64).reshape(-1, HIDDEN)
    router = np.asarray(block.gate.weight.value, np.float64)
    bias = np.asarray(block.gate.e_score_correction_bias.value, np.float64)
    scores = 1.0 / (1.0 + np.exp(-(xs @ router)))
    chosen = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :TOPK]
    weights = np.take_along_axis(scores, chosen, axis=-1)
    weights = weights / (weights.sum(-1, keepdims=True) + 1e-20) * 2.5
    gate = np.asarray(block.experts.gate_proj.weight.value, np.float64)
    up = np.asarray(block.experts.up_proj.weight.value, np.float64)
    down = np.asarray(block.experts.down_proj.weight.value, np.float64)
    out = np.zeros_like(xs)
    for t in range(xs.shape[0]):
        for j in range(TOPK):
            e = chosen[t, j]
            g = np.minimum(xs[t] @ gate[e], 10.0)
            u = np.clip(xs[t] @ up[e], -10.0, 10.0)
            out[t] += weights[t, j] * (((g / (1.0 + np.exp(-g))) * u) @ down[e])
    return out.reshape(np.shape(x))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _reduce_scatter_operands(text: str) -> set[tuple[int, str]]:
    """``(leading dim, element type)`` of every reduce-scatter operand in a lowered forward."""
    return {(int(dims.split("x")[0]), dtype) for dims, dtype in _REDUCE_SCATTER.findall(text)}


def _assert_deferred_lowering(
    text_default: str, text_deferred: str, *, mesh: str, decode: bool, tokens: int = TOKENS
) -> None:
    """Fail unless the deferred dispatch ran: a gate fallback (warn once) lowers the default program unchanged."""
    tp = MESHES[mesh][4]
    assert text_deferred != text_default, "moe_defer_tp_reduce fell back to the default dispatch"
    deferred_rows = {rows for rows, _ in _reduce_scatter_operands(text_deferred)}
    default_rows = {rows for rows, _ in _reduce_scatter_operands(text_default)}
    assert tokens * TOPK not in deferred_rows, f"deferred dispatch reduce-scatters expanded rows: {deferred_rows}"
    if decode:
        # One all-reduce over (ep, tp) replaces the TP reduce-scatter and the decode boundary all-gather.
        assert not deferred_rows, f"decode-layout deferred dispatch still reduce-scatters: {deferred_rows}"
        assert "stablehlo.all_gather" not in text_deferred, "the decode boundary all-gather survived"
        if tp > 1:
            assert "stablehlo.all_gather" in text_default, "the default decode dispatch has no boundary all-gather"
    elif tp > 1:
        assert deferred_rows == {tokens}, f"deferred dispatch should reduce-scatter [tokens, H] rows: {deferred_rows}"
        assert tokens * TOPK in default_rows, f"default dispatch should reduce-scatter [tokens*k, H]: {default_rows}"


@LAYOUTS
@pytest.mark.parametrize("mesh", sorted(MESHES))
def test_deferred_dispatch_matches_default_and_reference(mesh, decode):
    _skip_without_four_devices()
    block = _build(MESHES[mesh])
    x = _x()
    reference = _reference(block, x)
    out_default, text_default = _forward(block, x, defer=False, decode=decode)
    out_deferred, text_deferred = _forward(block, x, defer=True, decode=decode)
    _assert_deferred_lowering(text_default, text_deferred, mesh=mesh, decode=decode)
    assert np.isfinite(out_deferred).all()
    assert _rel(out_default, reference) < 1e-4, _rel(out_default, reference)
    assert _rel(out_deferred, reference) < 1e-4, _rel(out_deferred, reference)
    assert _rel(out_deferred, out_default) < 1e-5, _rel(out_deferred, out_default)


@LAYOUTS
@pytest.mark.parametrize("mesh", ["ep2tp2", "ep4tp1"])
def test_deferred_dispatch_matches_default_w8a8(mesh, decode):
    _skip_without_four_devices()
    block = _build(MESHES[mesh], quantize=True)
    x = _x()
    out_default, text_default = _forward(block, x, defer=False, decode=decode)
    out_deferred, text_deferred = _forward(block, x, defer=True, decode=decode)
    _assert_deferred_lowering(text_default, text_deferred, mesh=mesh, decode=decode)
    assert np.isfinite(out_deferred).all() and float(np.max(np.abs(out_default))) > 0
    assert _rel(out_deferred, out_default) < 1e-5, _rel(out_deferred, out_default)


@LAYOUTS
def test_deferred_dispatch_matches_default_bf16_w8a8(decode):
    """Served precision (bfloat16 activations, W8A8 experts) on ep2tp2, with both the TP and the expert reduction."""
    _skip_without_four_devices()
    block = _build(MESHES["ep2tp2"], quantize=True, dtype=jnp.bfloat16)
    x = _x(jnp.bfloat16)
    out_default, text_default = _forward(block, x, defer=False, decode=decode)
    out_deferred, text_deferred = _forward(block, x, defer=True, decode=decode)
    _assert_deferred_lowering(text_default, text_deferred, mesh="ep2tp2", decode=decode)
    if not decode:
        # The combined partial is reduce-scattered over TP in bfloat16, as the default does with its expanded rows.
        deferred_ops = _reduce_scatter_operands(text_deferred)
        assert deferred_ops == {(TOKENS, "bf16")}, deferred_ops
        assert (TOKENS * TOPK, "bf16") in _reduce_scatter_operands(text_default)
    assert np.isfinite(out_deferred).all() and float(np.max(np.abs(out_default))) > 0
    rel = _rel(out_deferred, out_default)
    assert rel < BF16_REL_TOL, rel


def test_deferred_dispatch_reduce_scatters_combined_rows():
    _skip_without_four_devices()
    block = _build(MESHES["ep2tp2"])
    x = _x()
    _, text_default = _forward(block, x, defer=False)
    _, text_deferred = _forward(block, x, defer=True)
    default_rows = {rows for rows, _ in _reduce_scatter_operands(text_default)}
    deferred_rows = {rows for rows, _ in _reduce_scatter_operands(text_deferred)}
    assert TOKENS * TOPK in default_rows, f"default dispatch should reduce [tokens*k, H] rows, saw {default_rows}"
    assert TOKENS in deferred_rows, f"deferred dispatch should reduce [tokens, H] rows, saw {deferred_rows}"
    assert TOKENS * TOPK not in deferred_rows, f"deferred dispatch still reduces expanded rows: {deferred_rows}"


def test_knob_is_inert_outside_inference_mode():
    _skip_without_four_devices()
    block = _build(MESHES["ep2tp2"])
    x = _x()
    out_default, text_default = _forward(block, x, defer=False, inference=False)
    out_knob, text_knob = _forward(block, x, defer=True, inference=False)
    out_window, text_window = _forward(block, x, defer=True, inference=False, window_rows=FORCED_WINDOW_ROWS)
    # A gate fallback lowers exactly the default program, which is what makes the lowering inequality asserted by
    # every equivalence test above proof that the deferred dispatch ran. The row-window size cannot reach training.
    assert text_knob == text_default
    assert text_window == text_default
    np.testing.assert_array_equal(out_knob, out_default)
    np.testing.assert_array_equal(out_window, out_default)


def _while_loops(text: str) -> int:
    return text.count("stablehlo.while")


@LAYOUTS
@pytest.mark.parametrize("mesh", sorted(MESHES))
def test_windowed_dispatch_matches_one_buffer_and_reference(mesh, decode):
    """Forced row windows (several per shard, the last one padded) give the one-buffer deferred result."""
    _skip_without_four_devices()
    block = _build(MESHES[mesh])
    x = _x()
    reference = _reference(block, x)
    _, text_default = _forward(block, x, defer=False, decode=decode)
    out_full, text_full = _forward(block, x, defer=True, decode=decode)
    out_window, text_window = _forward(block, x, defer=True, decode=decode, window_rows=FORCED_WINDOW_ROWS)
    _assert_deferred_lowering(text_default, text_window, mesh=mesh, decode=decode)
    assert text_window != text_full, "the forced row window left the one-buffer deferred program unchanged"
    assert _while_loops(text_window) > _while_loops(text_full), "no window loop in the windowed program"
    assert np.isfinite(out_window).all()
    assert _rel(out_window, reference) < 1e-4, _rel(out_window, reference)
    # Only the fp32 summation order across windows differs from the one-buffer combine.
    assert _rel(out_window, out_full) < 1e-5, _rel(out_window, out_full)


@pytest.mark.parametrize("mesh", ["ep2tp2", "ep4tp1"])
def test_windowed_dispatch_is_dropless_under_skewed_routing(mesh):
    """Every route lands on expert shard 0, twice its automatic window: the second window must run."""
    _skip_without_four_devices()
    ep = MESHES[mesh][3]
    block = _build(MESHES[mesh])
    bias = np.zeros((EXPERTS,), np.float32)
    # Sigmoid scores lie in (0, 1), so a +10 selection bias makes shard 0's experts win every top-k slot; the combine
    # weights come from the unbiased scores (and the float64 reference applies the bias the same way).
    bias[: EXPERTS // ep] = 10.0
    block.gate.e_score_correction_bias.value = jnp.asarray(bias, jnp.float32)
    x = _x(tokens=SKEW_TOKENS)
    reference = _reference(block, x)
    out_default, text_default = _forward(block, x, defer=False)
    out_full, text_full = _forward(block, x, defer=True, window_rows=SKEW_TOKENS * TOPK)
    out_window, text_window = _forward(block, x, defer=True)
    _assert_deferred_lowering(text_default, text_window, mesh=mesh, decode=False, tokens=SKEW_TOKENS)
    assert _while_loops(text_window) > _while_loops(text_full), "the automatic window did not split the rows"
    assert np.isfinite(out_window).all() and float(np.max(np.abs(out_default))) > 0
    assert _rel(out_default, reference) < 1e-4, _rel(out_default, reference)
    assert _rel(out_window, reference) < 1e-4, _rel(out_window, reference)
    assert _rel(out_window, out_full) < 1e-5, _rel(out_window, out_full)
    assert _rel(out_window, out_default) < 1e-5, _rel(out_window, out_default)


@LAYOUTS
@pytest.mark.parametrize("mesh", ["ep2tp2", "ep4tp1"])
def test_windowed_dispatch_bf16_w8a8(mesh, decode):
    """Served precision (bfloat16 activations, W8A8 experts) with forced row windows."""
    _skip_without_four_devices()
    block = _build(MESHES[mesh], quantize=True, dtype=jnp.bfloat16)
    x = _x(jnp.bfloat16)
    out_default, text_default = _forward(block, x, defer=False, decode=decode)
    out_full, _ = _forward(block, x, defer=True, decode=decode)
    out_window, text_window = _forward(block, x, defer=True, decode=decode, window_rows=FORCED_WINDOW_ROWS)
    _assert_deferred_lowering(text_default, text_window, mesh=mesh, decode=decode)
    assert np.isfinite(out_window).all() and float(np.max(np.abs(out_default))) > 0
    assert _rel(out_window, out_default) < BF16_REL_TOL, _rel(out_window, out_default)
    assert _rel(out_window, out_full) < BF16_REL_TOL, _rel(out_window, out_full)
