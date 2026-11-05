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

"""Inference-mode W8A8 routed experts at decode and measured prefill rows run ejkernel's Pallas A8 whole-slab entry.

``_channelwise_grouped_platform`` selects the Pallas channelwise entry for A8 activations with int8 codes at up to
``_A8_PALLAS_MAX_ROWS`` dispatch rows under inference mode on TPU v5p, when one 128-aligned ``(K, N)`` expert slab
fits ``_A8_PALLAS_MAX_SLAB_BYTES``; on ``_A8_PALLAS_PREFILL_FAMILIES`` up to ``_A8_PALLAS_PREFILL_MAX_ROWS`` rows.
Pinned here:

* the selection: Pallas only under inference mode, on v5p, for bfloat16 rows, int8 codes and a fitting 128-aligned
  slab, at most 1024 rows or, on the measured prefill families, at most 81,920 rows; XLA for training,
  ``moe_force_xla_gmm``, other chips and other shapes;
* a W8A8 GLM-5-Next routed-MoE block at the served precision (bfloat16 activations) under inference mode, in the
  decode and the TP-scattered layout on ep2 x tp2 and ep4 x tp1: the automatic selection compiles the gate, up and
  down projections as Pallas grouped matmuls and no XLA ragged dot (``moe_force_xla_gmm`` the reverse), and the
  outputs are bit-identical;
* outside inference mode (training) the block compiles no Pallas grouped matmul, with or without
  ``moe_force_xla_gmm``, and the outputs are identical.
"""

from __future__ import annotations

import contextlib
import re
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.moe import _moe_module as mm
from jax.experimental.pallas import tpu

AXES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
HIDDEN, INTER, EXPERTS, TOPK, TOKENS = 256, 256, 16, 4, 32
MESHES = {"ep2tp2": (1, 1, 1, 2, 2, 1), "ep4tp1": (1, 1, 1, 4, 1, 1)}
# Both expert matmuls compile to TPU custom calls; their outputs tell them apart: XLA's int8 ragged dot writes the raw
# int32 dot (``s32[rows, n]``), ejkernel's A8 slab kernel the scaled float32 rows (``f32[rows, n]``). At these shapes
# the block runs no other Pallas kernel (its top-k combine stays on XLA below ``_LIVE_COMBINE_MIN_LOCAL_ROWS``).
_EXPERT_CUSTOM_CALL = re.compile(
    r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*(s32|f32)\[\d+,\d+\](?:\{[^}]*\})?\s+custom-call\("
    r".*custom_call_target=\"tpu_custom_call\"",
    re.M,
)


@pytest.mark.parametrize(
    "shape,inference,force,chip,expected",
    [
        ((72, 256, 4096, 512), True, False, "v5p", "pallas"),
        ((72, 256, 512, 4096), True, False, "v5p", "pallas"),
        ((72, 1, 4096, 512), True, False, "v5p", "pallas"),
        ((72, 1024, 4096, 512), True, False, "v5p", "pallas"),
        ((288, 256, 4096, 128), True, False, "v5p", "pallas"),
        ((72, 1025, 4096, 512), True, False, "v5p", "pallas"),
        ((72, 5120, 512, 4096), True, False, "v5p", "pallas"),
        ((72, 81920, 4096, 512), True, False, "v5p", "pallas"),
        ((72, 81921, 4096, 512), True, False, "v5p", "xla"),
        ((288, 1025, 4096, 128), True, False, "v5p", "xla"),
        ((72, 5120, 4096, 512), False, False, "v5p", "xla"),
        ((72, 5120, 4096, 512), True, True, "v5p", "xla"),
        ((128, 5120, 640, 2560), True, False, "v5p", "xla"),
        ((72, 256, 4096, 512), False, False, "v5p", "xla"),
        ((72, 256, 4096, 512), True, True, "v5p", "xla"),
        ((72, 256, 4096, 512), True, False, "v6e", "xla"),
        ((128, 80, 2560, 1280), True, False, "v5p", "xla"),
        ((72, 256, 4096, 640), True, False, "v5p", "xla"),
        ((16, 64, 256, 192), True, False, "v5p", "xla"),
    ],
)
def test_a8_selection(monkeypatch, shape, inference, force, chip, expected):
    from easydel.utils.inference_mode import set_inference_mode

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(
        tpu, "get_tpu_info", lambda: types.SimpleNamespace(chip_version=types.SimpleNamespace(value=chip))
    )
    e, m, k, n = shape
    x = jax.ShapeDtypeStruct((m, k), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((e, k, n), jnp.int8)
    with set_inference_mode(inference):
        assert mm._channelwise_grouped_platform(x, w, 8, force_xla=force) == expected


def test_a8_selection_keeps_xla_for_other_operands(monkeypatch):
    from easydel.utils.inference_mode import set_inference_mode

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(
        tpu, "get_tpu_info", lambda: types.SimpleNamespace(chip_version=types.SimpleNamespace(value="v5p"))
    )
    with set_inference_mode():
        f32_rows = jax.ShapeDtypeStruct((256, 4096), jnp.float32)
        int8 = jax.ShapeDtypeStruct((72, 4096, 512), jnp.int8)
        int4 = jax.ShapeDtypeStruct((72, 4096, 512), jnp.int4)
        bf16_rows = jax.ShapeDtypeStruct((256, 4096), jnp.bfloat16)
        assert mm._channelwise_grouped_platform(f32_rows, int8, 8) == "xla"
        assert mm._channelwise_grouped_platform(bf16_rows, int4, 8) == "xla"
        monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
        assert mm._channelwise_grouped_platform(bf16_rows, int8, 8) == "xla"


def _build(dims):
    """W8A8 GLM-5-Next routed-MoE block (no shared expert) in bfloat16 on ``dims``."""
    import easydel as ed
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
        n_shared_experts=0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        swiglu_limit=10.0,
    )
    config.add_basic_configurations(sharding_axis_dims=dims, sharding_axis_names=AXES, moe_method="fused_moe")
    block = Glm5NextTextMoE(
        config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, precision=None, rngs=spx.Rngs(3)
    )
    for projection in (block.experts.gate_proj, block.experts.up_proj, block.experts.down_proj):
        projection.weight.value = projection.weight.value * 5
    quantizer = EasyQuantizer(
        quantization_config=QuantizationConfig(dtype="w8a8", pattern=r"experts\.(?:gate_proj|up_proj|down_proj)$")
    )
    block = quantizer.apply_quantization(block)
    assert block.experts.down_proj.quant_kernel.value.dtype == jnp.int8
    return block


def _forward(block, x, *, force_xla: bool, inference: bool, decode: bool):
    """Output (float32 host array) and compiled HLO text, with ``moe_force_xla_gmm`` set at trace time."""
    from easydel.infra.sharding import decode_mode_specs
    from easydel.utils.inference_mode import set_inference_mode

    graphdef, state = spx.export(block)

    def fwd(st, h):
        module = spx.bind(graphdef, st)
        module.config.moe_force_xla_gmm = force_xla
        return module(h)[0]

    with contextlib.ExitStack() as stack:
        stack.enter_context(block.config.mesh)
        if inference:
            stack.enter_context(set_inference_mode())
        stack.enter_context(decode_mode_specs(decode))
        compiled = jax.jit(fwd).lower(state, x).compile()
        out = compiled(state, x)
    return np.asarray(jax.device_get(out), np.float32), compiled.as_text()


def _expert_calls(text: str) -> tuple[int, int]:
    """(A8 slab kernels, XLA int8 ragged dots) in a compiled program, told apart by their output dtype."""
    kinds = _EXPERT_CUSTOM_CALL.findall(text)
    return kinds.count("f32"), kinds.count("s32")


def _skip_unless_v5p_with_four_devices():
    if jax.default_backend() != "tpu" or jax.device_count() < 4:
        pytest.skip("needs 4 TPU devices")
    if tpu.get_tpu_info().chip_version.value != "v5p":
        pytest.skip("the A8 slab selection is measured on v5p only")


@pytest.mark.parametrize("decode", [False, True], ids=["prefill_layout", "decode_layout"])
@pytest.mark.parametrize("mesh", sorted(MESHES))
def test_w8a8_block_runs_the_slab_kernels_bit_identically(mesh, decode):
    _skip_unless_v5p_with_four_devices()
    block = _build(MESHES[mesh])
    x = jax.random.normal(jax.random.PRNGKey(0), (1, TOKENS, HIDDEN), jnp.float32).astype(jnp.bfloat16)
    out_auto, text_auto = _forward(block, x, force_xla=False, inference=True, decode=decode)
    out_xla, text_xla = _forward(block, x, force_xla=True, inference=True, decode=decode)
    pallas_auto, ragged_auto = _expert_calls(text_auto)
    pallas_xla, ragged_xla = _expert_calls(text_xla)
    # Gate, up and down: Pallas grouped matmuls and no XLA ragged dot; moe_force_xla_gmm gives the reverse.
    assert pallas_auto >= 3 and ragged_auto == 0, (pallas_auto, ragged_auto)
    assert pallas_xla == 0 and ragged_xla >= 3, (pallas_xla, ragged_xla)
    assert np.isfinite(out_auto).all() and float(np.max(np.abs(out_xla))) > 0
    np.testing.assert_array_equal(out_auto, out_xla)


def test_w8a8_block_training_program_is_unchanged():
    _skip_unless_v5p_with_four_devices()
    block = _build(MESHES["ep2tp2"])
    x = jax.random.normal(jax.random.PRNGKey(0), (1, TOKENS, HIDDEN), jnp.float32).astype(jnp.bfloat16)
    out_auto, text_auto = _forward(block, x, force_xla=False, inference=False, decode=False)
    out_xla, text_xla = _forward(block, x, force_xla=True, inference=False, decode=False)
    assert _expert_calls(text_auto)[0] == 0 and _expert_calls(text_xla)[0] == 0
    assert _expert_calls(text_auto)[1] >= 3, _expert_calls(text_auto)
    np.testing.assert_array_equal(out_auto, out_xla)
