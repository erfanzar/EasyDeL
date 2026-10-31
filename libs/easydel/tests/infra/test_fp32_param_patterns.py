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

"""``fp32_param_patterns``: leaves a model keeps in float32 under a low-precision ``param_dtype``.

Every path that materializes parameters must keep the pinned leaves float32 (and bit-exact against the
source) while casting the rest to ``param_dtype``: construction, ``lazy_init``, HF conversion (full and
streaming torch loaders), and the native TensorStore save -> load roundtrip. Models without patterns are
unaffected.
"""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.utils.parameters_transformation import fp32_param_pattern_matches

PINNED = ("input_layernorm.weight",)
AXIS_DIMS = (1, 1, 1, 1, -1, 1)
NUM_LAYERS = 2


def _variables(model) -> dict[str, jax.Array]:
    return {path: var.value for path, var in spx.iter_variables(model) if var.kind == "parameters"}


def _split(model):
    leaves = _variables(model)
    pinned = {k: v for k, v in leaves.items() if fp32_param_pattern_matches(k, PINNED)}
    others = {k: v for k, v in leaves.items() if k not in pinned and jnp.issubdtype(v.dtype, jnp.floating)}
    return pinned, others


def _assert_layout(model, reference: dict[str, np.ndarray] | None = None):
    pinned, others = _split(model)
    assert len(pinned) == NUM_LAYERS, sorted(pinned)
    for key, value in pinned.items():
        assert value.dtype == jnp.float32, key
    assert others and all(v.dtype == jnp.bfloat16 for v in others.values()), {
        k: v.dtype for k, v in others.items() if v.dtype != jnp.bfloat16
    }
    if reference is not None:
        for key, value in pinned.items():
            layer = next(int(part) for part in key.split(".") if part.isdigit())
            np.testing.assert_array_equal(np.asarray(jax.device_get(value)), reference[layer])


@pytest.fixture
def pinned_llama(monkeypatch):
    monkeypatch.setattr(ed.LlamaForCausalLM, "fp32_param_patterns", PINNED)


def _tiny_config():
    return ed.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sharding_axis_dims=AXIS_DIMS,
    )


def _tiny_hf_llama(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).to(torch.float32)
    with torch.no_grad():
        for layer in model.model.layers:
            # 1 + 1e-4 * noise is not representable in bfloat16, so a downcast is visible.
            layer.input_layernorm.weight.copy_(1.0 + 1e-4 * torch.randn_like(layer.input_layernorm.weight))
    path = tmp_path / "hf"
    model.save_pretrained(path, safe_serialization=True)
    reference = [layer.input_layernorm.weight.detach().numpy().copy() for layer in model.model.layers]
    return path, reference


def _from_pretrained(path, dtype=jnp.bfloat16, **kwargs):
    return ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(path),
        dtype=dtype,
        param_dtype=dtype,
        sharding_axis_dims=AXIS_DIMS,
        auto_shard_model=True,
        verbose=False,
        **kwargs,
    )


def test_matcher_uses_whole_dotted_segments():
    assert fp32_param_pattern_matches("model.layers.0.mlp.router.weight", ("router.weight",))
    assert fp32_param_pattern_matches(("parameters", "layers", 3, "A_log"), ("A_log",))
    assert fp32_param_pattern_matches("model/layers/0/mixer/dt_bias", ("dt_bias",))
    assert not fp32_param_pattern_matches("model.layers.0.routers.weight", ("router.weight",))
    assert not fp32_param_pattern_matches("model.layers.0.router.weight_scale", ("router.weight",))
    assert not fp32_param_pattern_matches("model.layers.0.router.weight", ())


def test_construction_and_lazy_init_pin_leaves(pinned_llama):
    model = ed.LlamaForCausalLM(config=_tiny_config(), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    _assert_layout(model)
    lazy = ed.LlamaForCausalLM.lazy_init(
        config=_tiny_config(), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0)
    )
    _assert_layout(lazy)
    input_ids = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
    with model.config.mesh:
        logits = model(input_ids=input_ids).logits
    assert bool(jnp.all(jnp.isfinite(logits)))


def test_no_patterns_keeps_every_leaf_at_param_dtype():
    model = ed.LlamaForCausalLM(config=_tiny_config(), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    floats = [v for v in _variables(model).values() if jnp.issubdtype(v.dtype, jnp.floating)]
    assert floats and all(v.dtype == jnp.bfloat16 for v in floats)


@pytest.mark.parametrize("mode", ["full", "streaming"])
def test_hf_conversion_keeps_pinned_leaves_fp32(tmp_path, pinned_llama, mode):
    path, reference = _tiny_hf_llama(tmp_path)
    model = _from_pretrained(path, torch_load_mode=mode)
    _assert_layout(model, reference)


def test_hf_conversion_without_patterns_rounds_every_leaf(tmp_path):
    path, _ = _tiny_hf_llama(tmp_path)
    model = _from_pretrained(path)
    floats = [v for v in _variables(model).values() if jnp.issubdtype(v.dtype, jnp.floating)]
    assert floats and all(v.dtype == jnp.bfloat16 for v in floats)


@pytest.mark.parametrize("float_dtype", [None, jnp.bfloat16])
def test_native_save_load_roundtrip_keeps_pinned_leaves_fp32(tmp_path, pinned_llama, float_dtype):
    path, reference = _tiny_hf_llama(tmp_path)
    model = _from_pretrained(path)
    model.save_pretrained(str(tmp_path / "native"), float_dtype=float_dtype)
    del model
    loaded = _from_pretrained(tmp_path / "native")
    _assert_layout(loaded, reference)


def test_native_load_chunked_keeps_pinned_leaves_fp32(tmp_path, pinned_llama):
    path, reference = _tiny_hf_llama(tmp_path)
    _from_pretrained(path, dtype=jnp.float32).save_pretrained(str(tmp_path / "native"))
    loaded = _from_pretrained(tmp_path / "native", checkpoint_load_chunk_size=1)
    _assert_layout(loaded, reference)
