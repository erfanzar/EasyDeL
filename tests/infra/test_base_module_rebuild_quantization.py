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

import jax.numpy as jnp
import spectrax as spx

import easydel as ed
from easydel.layers.quantization import QuantizationConfig, QuantizationType


def _build_tiny_qwen3():
    config = ed.Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        quantization_config=QuantizationConfig(dtype=QuantizationType.INT8, group_size=64),
    )
    _, model_cls = ed.get_modules_by_type("qwen3", ed.TaskType.CAUSAL_LM)
    return model_cls.lazy_init(
        config=config,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=None,
        rngs=spx.Rngs(0),
    )


def test_new_graphdef_keeps_dense_models_dense_when_quantization_config_is_only_in_config():
    model = _build_tiny_qwen3()
    assert not model.is_quantized

    graphdef = model.new_graphdef()
    rebuilt = model.merge_module(graphdef, model.graphstate, model.graphother)

    assert not rebuilt.is_quantized


def test_new_graphdef_preserves_quantized_models():
    model = _build_tiny_qwen3()
    model = model.quantize(quantization_config=model.config.quantization_config, verbose=False)
    assert model.is_quantized

    graphdef = model.new_graphdef()
    rebuilt = model.merge_module(graphdef, model.graphstate, model.graphother)

    assert rebuilt.is_quantized
