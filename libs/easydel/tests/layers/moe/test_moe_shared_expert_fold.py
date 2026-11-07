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

"""The W8A8 shared expert inside the inference-mode deferred MoE dispatch (``moe_call(..., shared_expert=...)``).

Under ``set_inference_mode()`` with the deferred dispatch active (``moe_defer_tp_reduce``), ``BaseMoeModule.moe_call``
runs a W8A8 gated-MLP shared expert inside the fused dispatch's ``shard_map``: every expert shard runs it on its share
of the token rows with this TP rank's weight shards, and its float32 partial joins the routed partial before the
dispatch's one reduction. On four TPU devices (GLM-5-Next MoE block, bfloat16 activations, W8A8 routed and shared
experts) these tests pin:

* outputs stay within the W8A8 error of an independent float64 reference of the unquantized block, are no further from
  it than the shared expert added after ``moe_call`` (the composition before the fold), and stay close to that
  composition. The two differ because the folded down projection quantizes each TP rank's slice of a hidden row on its
  own, and the shared and routed partials are summed in float32 instead of being rounded to bfloat16 separately,
* the folded program runs the collectives of the routed experts alone: the shared expert adds none (no int32 or int8
  collective is left),
* every token is counted by exactly one expert shard when the token count does not divide over the expert axis,
  including fewer tokens than expert shards,
* outside inference mode, and for a shared expert the fold does not accept (dense bfloat16), ``moe_call`` lowers the
  program of adding the shared expert after it,
* ``Glm5NextTextMoE`` hands its shared experts to ``moe_call``.
"""

from __future__ import annotations

import collections
import contextlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

AXES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
HIDDEN, INTER, EXPERTS, TOPK, TOKENS = 256, 256, 16, 4, 32
LIMIT, SCALING = 10.0, 2.5
MESHES = {"ep2tp2": (1, 1, 1, 2, 2, 1), "ep4tp1": (1, 1, 1, 4, 1, 1), "ep1tp4": (1, 1, 1, 1, 4, 1)}
LAYOUTS = pytest.mark.parametrize("decode", [False, True], ids=["prefill_layout", "decode_layout"])
# W8A8 against the unquantized float64 block: per-channel int8 weights and per-row int8 activations at three
# projections each for the routed and the shared experts, plus bfloat16 roundings.
REF_TOL = 6e-2
TOKEN_REF_TOL = 0.15
# Folded against added after ``moe_call``: the TP-local quantization of the shared down projection's input and the
# float32 sum of the shared and routed partials (about one activation quantization step, a few bfloat16 roundings).
FOLD_TOL = 4e-2
_INSTR = re.compile(r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*(.+?)\s+([\w\-]+)\(")
_LAYOUT = re.compile(r"\{[^}]*\}")
_COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "ragged-all-to-all", "collective-permute")


def _skip_without_four_devices():
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices")


def _build(dims, *, quantize_shared: bool = True):
    """GLM-5-Next MoE block with one shared expert, bfloat16, W8A8 routed experts (and shared expert by default).

    Returns:
        ``(block, dense)``: the quantized block and float64 copies of its weights before quantization.
    """
    import easydel as ed
    from easydel.infra.base_module import _parameter_init_sharding_context
    from easydel.layers.quantization import EasyQuantizer, QuantizationConfig
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
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=SCALING,
        swiglu_limit=LIMIT,
    )
    config.add_basic_configurations(sharding_axis_dims=dims, sharding_axis_names=AXES, moe_method="fused_moe")
    with _parameter_init_sharding_context(config):
        block = Glm5NextTextMoE(
            config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, precision=None, rngs=spx.Rngs(3)
        )
    shared = block.shared_experts
    projections = {
        "gate": block.experts.gate_proj,
        "up": block.experts.up_proj,
        "down": block.experts.down_proj,
        "shared_gate_up": shared.gate_up_proj,
        "shared_down": shared.down_proj,
    }
    for projection in projections.values():
        projection.weight.value = projection.weight.value * 5
    dense = {name: np.asarray(proj.weight.value, np.float64) for name, proj in projections.items()}
    dense["router"] = np.asarray(block.gate.weight.value, np.float64)
    dense["bias"] = np.asarray(block.gate.e_score_correction_bias.value, np.float64)
    pattern = (
        r".*experts\.(?:gate_proj|up_proj|down_proj|gate_up_proj)$"
        if quantize_shared
        else r"experts\.(?:gate_proj|up_proj|down_proj)$"
    )
    quantizer = EasyQuantizer(quantization_config=QuantizationConfig(dtype="w8a8", pattern=pattern))
    with _parameter_init_sharding_context(config):
        block = quantizer.apply_quantization(block)
    assert block.experts.down_proj.quant_kernel.value.dtype == jnp.int8
    if quantize_shared:
        assert block.shared_experts.gate_up_proj.quant_kernel.value.dtype == jnp.int8
        assert block.shared_experts.down_proj.quant_kernel.value.dtype == jnp.int8
    return block, dense


def _x(tokens: int):
    return jax.random.normal(jax.random.PRNGKey(0), (1, tokens, HIDDEN), dtype=jnp.float32).astype(jnp.bfloat16)


def _run(block, x, *, shared: str, decode: bool = False, inference: bool = True):
    """Block output (float64) with the lowered StableHLO and the compiled HLO text of its program.

    ``shared``: ``"moe_call"`` hands the shared expert to ``moe_call``; ``"after"`` adds it after ``moe_call`` (the
    composition before the fold); ``"none"`` runs the routed experts alone; ``"module"`` runs the block's forward.
    The deferred dispatch is requested (``moe_defer_tp_reduce=True``); it runs under inference mode.
    """
    from easydel.infra.sharding import decode_mode_specs
    from easydel.layers import clamped_swiglu
    from easydel.utils.inference_mode import set_inference_mode

    graphdef, state = spx.export(block)

    def fwd(st, h):
        module = spx.bind(graphdef, st)
        module.config.moe_defer_tp_reduce = True
        if shared == "module":
            return module(h)[0]
        hooks = module.moe_hooks.replace(select_hook=module._select_hook())

        def ffn_activation(gate, up):
            return clamped_swiglu(gate, up, limit=LIMIT, act_fn=module.experts.act_fn)

        out, _ = module.moe_call(
            hidden_state=h,
            gate_layer=module.gate,
            expert_layer=module.experts,
            wi_kernel=module.experts.gate_proj.kernel_view(),
            wu_kernel=module.experts.up_proj.kernel_view(),
            wd_kernel=module.experts.down_proj.kernel_view(),
            act_fn=module.experts.act_fn,
            ffn_activation=ffn_activation,
            hooks=hooks,
            shared_expert=module.shared_experts if shared == "moe_call" else None,
        )
        if shared == "after":
            out = out + module.shared_experts(h)
        return out

    with contextlib.ExitStack() as stack:
        stack.enter_context(jax.default_matmul_precision("highest"))
        stack.enter_context(block.config.mesh)
        if inference:
            stack.enter_context(set_inference_mode())
        stack.enter_context(decode_mode_specs(decode))
        lowered = jax.jit(fwd).lower(state, x)
        compiled = lowered.compile()
        out = compiled(state, x)
    return np.asarray(jax.device_get(out), np.float32).astype(np.float64), lowered.as_text(), compiled.as_text()


def _silu(v):
    return v / (1.0 + np.exp(-v))


def _reference(dense, x, *, tp: int):
    """Independent float64 GLM-5 MoE block on the unquantized weights: routed experts plus the shared expert.

    The shared expert's fused gate/up columns are TP-interleaved (one ``[gate | up]`` segment per TP rank).
    """
    xs = np.asarray(x, np.float64).reshape(-1, HIDDEN)
    tokens = xs.shape[0]
    scores = 1.0 / (1.0 + np.exp(-(xs @ dense["router"])))
    chosen = np.argsort(-(scores + dense["bias"]), axis=-1, kind="stable")[:, :TOPK]
    weights = np.take_along_axis(scores, chosen, axis=-1)
    weights = weights / (weights.sum(-1, keepdims=True) + 1e-20) * SCALING
    out = np.zeros_like(xs)
    for t in range(tokens):
        for j in range(TOPK):
            e = chosen[t, j]
            g = np.minimum(xs[t] @ dense["gate"][e], LIMIT)
            u = np.clip(xs[t] @ dense["up"][e], -LIMIT, LIMIT)
            out[t] += weights[t, j] * ((_silu(g) * u) @ dense["down"][e])
    inter = dense["shared_down"].shape[0]
    gate_up = (xs @ dense["shared_gate_up"]).reshape(tokens, tp, 2, inter // tp)
    g = np.minimum(gate_up[:, :, 0].reshape(tokens, inter), LIMIT)
    u = np.clip(gate_up[:, :, 1].reshape(tokens, inter), -LIMIT, LIMIT)
    out += (_silu(g) * u) @ dense["shared_down"]
    return out.reshape(np.shape(x))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _max_token_rel(a, b):
    a2, b2 = a.reshape(-1, HIDDEN), b.reshape(-1, HIDDEN)
    return float(np.max(np.linalg.norm(a2 - b2, axis=-1) / np.maximum(np.linalg.norm(b2, axis=-1), 1e-30)))


def _collectives(compiled_text: str) -> list[tuple[str, str]]:
    """``(opcode, result shape)`` of every collective in a compiled program (async ``-start`` halves folded in)."""
    found = []
    for line in compiled_text.splitlines():
        m = _INSTR.match(line)
        if not m:
            continue
        shape, opcode = m.groups()
        base = opcode[: -len("-start")] if opcode.endswith("-start") else opcode
        if base in _COLLECTIVES:
            found.append((base, _LAYOUT.sub("", shape)))
    return found


def _assert_close_to_reference(folded, after, reference):
    assert np.isfinite(folded).all()
    err_folded, err_after = _rel(folded, reference), _rel(after, reference)
    assert err_folded < REF_TOL, f"folded vs float64 reference: {err_folded:.3e}"
    assert err_after < REF_TOL, f"added after moe_call vs float64 reference: {err_after:.3e}"
    assert err_folded <= 1.25 * err_after + 2e-3, f"folded {err_folded:.3e} vs added after {err_after:.3e}"
    token_err = _max_token_rel(folded, reference)
    assert token_err < TOKEN_REF_TOL, f"worst token vs float64 reference: {token_err:.3e}"
    fold_vs_after = _rel(folded, after)
    assert fold_vs_after < FOLD_TOL, f"folded vs added after moe_call: {fold_vs_after:.3e}"


@LAYOUTS
@pytest.mark.parametrize("mesh", sorted(MESHES))
def test_folded_shared_expert_matches_reference_without_collectives(mesh, decode):
    _skip_without_four_devices()
    block, dense = _build(MESHES[mesh])
    x = _x(TOKENS)
    folded, _, compiled_folded = _run(block, x, shared="moe_call", decode=decode)
    after, _, compiled_after = _run(block, x, shared="after", decode=decode)
    _, _, compiled_routed = _run(block, x, shared="none", decode=decode)
    assert compiled_folded != compiled_after, "the shared expert was not folded into the dispatch"
    folded_collectives = _collectives(compiled_folded)
    kinds_folded = collections.Counter(op for op, _ in folded_collectives)
    kinds_routed = collections.Counter(op for op, _ in _collectives(compiled_routed))
    assert kinds_folded == kinds_routed, f"folded {sorted(folded_collectives)} vs routed alone {kinds_routed}"
    integer = [entry for entry in folded_collectives if "s32[" in entry[1] or "s8[" in entry[1]]
    assert not integer, f"integer collectives left in the folded program: {integer}"
    _assert_close_to_reference(folded, after, _reference(dense, x, tp=MESHES[mesh][4]))


@pytest.mark.parametrize(
    ("mesh", "tokens", "decode"),
    [("ep4tp1", 31, False), ("ep2tp2", 31, False), ("ep4tp1", 3, True)],
    ids=["ep4_31_tokens", "ep2tp2_31_tokens", "ep4_3_tokens"],
)
def test_folded_shared_expert_counts_every_token_once(mesh, tokens, decode):
    _skip_without_four_devices()
    block, dense = _build(MESHES[mesh])
    x = _x(tokens)
    folded, _, compiled_folded = _run(block, x, shared="moe_call", decode=decode)
    after, _, compiled_after = _run(block, x, shared="after", decode=decode)
    assert compiled_folded != compiled_after, "the shared expert was not folded into the dispatch"
    _assert_close_to_reference(folded, after, _reference(dense, x, tp=MESHES[mesh][4]))


def test_shared_expert_argument_is_added_after_the_dispatch_outside_inference_mode():
    _skip_without_four_devices()
    block, _ = _build(MESHES["ep2tp2"])
    x = _x(TOKENS)
    out_arg, text_arg, _ = _run(block, x, shared="moe_call", inference=False)
    out_after, text_after, _ = _run(block, x, shared="after", inference=False)
    assert text_arg == text_after
    np.testing.assert_array_equal(out_arg, out_after)


def test_unfoldable_shared_expert_is_added_after_the_dispatch():
    _skip_without_four_devices()
    block, _ = _build(MESHES["ep2tp2"], quantize_shared=False)
    x = _x(TOKENS)
    out_arg, text_arg, _ = _run(block, x, shared="moe_call")
    out_after, text_after, _ = _run(block, x, shared="after")
    assert text_arg == text_after
    np.testing.assert_array_equal(out_arg, out_after)


@LAYOUTS
def test_glm5_next_moe_hands_its_shared_experts_to_moe_call(decode):
    _skip_without_four_devices()
    block, _ = _build(MESHES["ep2tp2"])
    x = _x(TOKENS)
    out_module = _run(block, x, shared="module", decode=decode)[0]
    out_folded = _run(block, x, shared="moe_call", decode=decode)[0]
    np.testing.assert_array_equal(out_module, out_folded)
