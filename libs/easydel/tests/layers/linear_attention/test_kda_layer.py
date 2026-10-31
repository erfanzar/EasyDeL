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

"""The shared Kernel Delta Attention layer (``KernelDeltaAttention``) used by Kimi-Linear and GLM-5-Next.

Per-family numerics are pinned by the packed / cached / HF-parity suites; these tests pin what the shared layer adds:
the base-config KDA settings reach every family, the cache the generation mixin builds has the layer's shapes,
parameters keep their storage dtypes, legacy nested forget-gate saves load, and the settings are validated at both
config entry points.
"""

from __future__ import annotations

import collections
import types

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import KDACacheView
from easydel.layers.linear_attention import KernelDeltaAttention, kda_layer_spec
from easydel.modules.glm5_next.modeling_glm5_next import (
    Glm5NextForCausalLM,
    Glm5NextLinearAttention,
    Glm5NextTextModel,
)
from easydel.modules.kimi_linear.modeling_kimi_linear import KimiDeltaAttention

HIDDEN, HEADS, HEAD_DIM, D_CONV = 32, 2, 8, 4


def _kimi_config(**overrides):
    return ed.KimiLinearConfig(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2,
        linear_attn_config={
            "kda_layers": [1], "full_attn_layers": [], "num_heads": HEADS, "head_dim": HEAD_DIM,
            "short_conv_kernel_size": D_CONV,
        },
        sharding_axis_dims=(1, 1, 1, 1, 1, 1), **overrides,
    )  # fmt: skip


def _glm_config(**overrides):
    return ed.Glm5NextTextConfig(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, n_routed_experts=4, num_experts_per_tok=2, kv_lora_rank=8,
        q_lora_rank=8, qk_rope_head_dim=0, qk_nope_head_dim=8, v_head_dim=8, index_topk=4, index_head_dim=4,
        index_n_heads=2, index_kpool=2, layer_types=["linear_attention"], mlp_layer_types=["dense"],
        linear_head_dim=HEAD_DIM, linear_num_heads=HEADS, linear_conv_kernel_dim=D_CONV, hc_mult=2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1), **overrides,
    )  # fmt: skip


FAMILIES = {
    "kimi_linear": (_kimi_config, ed.KimiLinearForCausalLM, KimiDeltaAttention),
    "glm5_next": (_glm_config, ed.Glm5NextForCausalLM, Glm5NextLinearAttention),
}


def _model(family, **overrides):
    make_config, model_cls, _ = FAMILIES[family]
    config = make_config(**overrides)
    with config.mesh:
        model = model_cls(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    layer = model.model.layers[0].self_attn
    # Non-trivial decay rates so the recurrence (and the gate) is exercised.
    layer.A_log.value = jnp.asarray(np.random.default_rng(1).normal(size=layer.A_log.value.shape) * 0.5, jnp.float32)
    return model, layer


def _hidden(batch, seq):
    return jnp.asarray(np.random.default_rng(0).normal(size=(batch, seq, HIDDEN)), jnp.float32)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _primitive_counts(fn, *args):
    counts = collections.Counter()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            counts.update([eqn.primitive.name])
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else (param,):
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns"):
                        walk(sub)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return counts


def _parameter_dtypes(layer):
    dtypes = {}
    for path, var in spx.iter_variables(layer):
        if getattr(var, "kind", None) == "parameters":
            name = path if isinstance(path, str) else ".".join(map(str, path))
            dtypes[name] = var.value.dtype
    return dtypes


def test_kda_chunk_impl_reaches_kimi_linear():
    """Kimi-Linear honours the base-config ``kda_chunk_impl``: the lowered program changes and the output matches."""
    h = _hidden(2, 128)
    _, scan = _model("kimi_linear")
    _, fused = _model("kimi_linear", kda_chunk_impl="kda_chunk")
    assert fused.config.kda_chunk_impl == "kda_chunk"
    with scan.config.mesh, jax.default_matmul_precision("highest"):
        want = scan(h).attention_output
        got = fused(h).attention_output
        assert _primitive_counts(lambda x: fused(x).attention_output, h) != _primitive_counts(
            lambda x: scan(x).attention_output, h
        )
    assert _rel(got, want) < 1e-3


@pytest.mark.parametrize("family", list(FAMILIES))
def test_generation_cache_config_fits_the_layer(family):
    """The mixin's KDA cache has the layer's shapes, and prefill / decode / append through it match one full pass."""
    model, layer = _model(family)
    assert isinstance(layer, KernelDeltaAttention)
    cache_config = model.create_kda_cache_config(batch_size=2)
    assert cache_config.key_dim == layer.key_dim
    assert cache_config.value_dim == layer.value_dim
    assert cache_config.d_conv == layer.d_conv
    assert tuple(cache_config.recurrent_state_shape) == (layer.num_heads, layer.head_k_dim, layer.head_v_dim)
    h = _hidden(2, 24)
    with model.config.mesh, jax.default_matmul_precision("highest"):
        view = KDACacheView.init(cache_config, layer_index=0, dtype=jnp.float32)
        full = layer(h).attention_output
        prefix = layer(h[:, :16], None, view)
        step = layer(h[:, 16:17], None, prefix.cache_view)
        rest = layer(h[:, 17:], None, step.cache_view)
    pieces = jnp.concatenate([prefix.attention_output, step.attention_output, rest.attention_output], axis=1)
    np.testing.assert_allclose(np.asarray(pieces), np.asarray(full), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_parameters_keep_their_storage_dtype(family):
    """Every parameter is stored in ``param_dtype`` except the gate's ``A_log`` / ``dt_bias``, which stay float32
    like Hugging Face's strict-fp32 gate."""
    make_config, _, layer_cls = FAMILIES[family]
    config = make_config()
    with config.mesh:
        layer = layer_cls(config, 0, dtype=jnp.float32, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
        dtypes = _parameter_dtypes(layer)
    assert dtypes.pop("A_log") == jnp.float32
    assert dtypes.pop("dt_bias") == jnp.float32
    assert set(dtypes.values()) == {jnp.dtype(jnp.bfloat16)}, dtypes


@pytest.mark.cpu_ok
def test_layer_spec_reads_moonshot_fla_and_glm_keys():
    def spec(linear_attn_config, **attrs):
        return kda_layer_spec(types.SimpleNamespace(linear_attn_config=linear_attn_config, **attrs))

    moonshot = spec({"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 3})
    assert tuple(moonshot) == (4, 16, 16, 3, 16, 64)
    fla = spec(
        {"num_heads": 2, "head_k_dim": 8, "head_v_dim": 12, "d_conv": 5, "gate_low_rank_dim": 6, "chunk_size": 32}
    )
    assert tuple(fla) == (2, 8, 12, 5, 6, 32)
    assert (fla.key_dim, fla.value_dim) == (16, 24)
    # GLM-5-Next's synthesized dict has no ``head_dim``: the gate bottleneck is the head width.
    glm = spec({"num_heads": 64, "head_k_dim": 128, "head_v_dim": 128, "d_conv": 4})
    assert tuple(glm) == (64, 128, 128, 4, 128, 64)
    assert spec(None, num_attention_heads=3).num_heads == 3
    with pytest.raises(ValueError, match="head count"):
        spec({}, num_attention_heads=0)


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "field,bad,good", [("kda_chunk_impl", "twophase", "kda_chunk"), ("kda_chunk_precision", "bf16", "default")]
)
def test_kda_settings_are_validated_by_both_config_entry_points(field, bad, good):
    with pytest.raises(ValueError, match=field):
        ed.LlamaConfig(hidden_size=64, num_attention_heads=4, **{field: bad})
    config = ed.LlamaConfig(hidden_size=64, num_attention_heads=4)
    with pytest.raises(ValueError, match=field):
        config.add_basic_configurations(**{field: bad})
    config.add_basic_configurations(**{field: good})
    assert getattr(config, field) == good


@pytest.mark.cpu_ok
@pytest.mark.parametrize("model_cls", [Glm5NextForCausalLM, Glm5NextTextModel])
def test_legacy_nested_forget_gate_saves_load_into_the_flat_layer(model_cls):
    """GLM-5-Next's load hook flattens a legacy nested ``self_attn.forget_gate`` and keeps flat states unchanged."""
    prefix = ("parameters", "model", "layers", 0, "self_attn")
    gate = [("A_log",), ("dt_bias",), ("f_a_proj", "weight"), ("f_b_proj", "weight")]
    other = (*prefix, "q_proj", "weight")
    expected = {(*prefix, *leaf): None for leaf in gate} | {other: None}
    legacy = {(*prefix, "forget_gate", *leaf): np.full((1,), i) for i, leaf in enumerate(gate)} | {other: np.zeros(1)}

    state = model_cls._upgrade_legacy_native_state(dict(legacy), expected)
    assert set(state) == set(expected)
    for i, leaf in enumerate(gate):
        assert state[(*prefix, *leaf)][0] == i
    current = {key: np.zeros(1) for key in expected}
    assert model_cls._upgrade_legacy_native_state(dict(current), expected).keys() == current.keys()
