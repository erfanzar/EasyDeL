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

"""Quantize-on-load from a native tensorstore checkpoint.

Modern spx checkpoints key their leaves as ``(collection, *path)`` (e.g.
``("parameters", "model", ..., "weight")``) while module paths carry no
collection segment. ``from_pretrained(..., quantization_config=...,
apply_quantization=True)`` must still locate and prepack every matching
kernel; a key mismatch here previously left every quantized leaf abstract
and failed materialization.
"""

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.linears._linear_quantized import ParallelLinearQuantized
from easydel.layers.quantization import QuantizationConfig
from easydel.utils import set_inference_mode
from easydel.utils.traversals import iter_module_search


def _tiny_llama_config() -> "ed.LlamaConfig":
    config = ed.LlamaConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=256,
        attn_mechanism="vanilla",
    )
    config.sharding_axis_dims = (1, 1, 1, 1, -1, 1)
    partition_axis = ed.PartitionAxis()
    partition_axis.hidden_state_axis = None
    config.partition_axis = partition_axis
    config.use_qmm_best_config = False  # keep CPU test free of autotune
    return config


@pytest.mark.parametrize("quant_dtype", ["int8"])
def test_quantize_on_load_from_tensorstore(tmp_path, quant_dtype):
    config = _tiny_llama_config()
    with set_inference_mode():
        model = ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=spx.Rngs(0),
        )
        model = model.shard_model()

    ids = jax.random.randint(jax.random.PRNGKey(1), (1, 16), 0, config.vocab_size)
    with model.mesh:
        ref = jax.device_get(model(input_ids=ids).logits.astype(jnp.float32))

    save_dir = tmp_path / "tiny_llama"
    model.save_pretrained(str(save_dir))

    qcfg = QuantizationConfig(dtype=quant_dtype, group_size=16)
    with set_inference_mode():
        loaded = ed.LlamaForCausalLM.from_pretrained(
            str(save_dir),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            sharding_axis_dims=(1, 1, 1, 1, -1, 1),
            partition_axis=config.partition_axis,
            auto_shard_model=True,
            quantization_config=qcfg,
            apply_quantization=True,
            config_kwargs={"use_qmm_best_config": False},
            verbose=False,
        )

    quantized_modules = [m for _, m in iter_module_search(loaded, spx.Module) if isinstance(m, ParallelLinearQuantized)]
    assert quantized_modules, "quantize-on-load converted no modules"
    for module in quantized_modules:
        kernel = getattr(module.quant_kernel, "value", None)
        assert kernel is not None and not isinstance(kernel, jax.ShapeDtypeStruct), (
            "quantized kernel left abstract by quantize-on-load"
        )

    with loaded.mesh:
        out = jax.device_get(loaded(input_ids=ids).logits.astype(jnp.float32))

    assert np.isfinite(out).all(), "quantized forward produced non-finite logits"
    # Reference is the bf16 model; int8 g16 weight quantization must keep
    # greedy decisions essentially intact on a tiny model.
    agreement = (out.argmax(-1) == ref.argmax(-1)).mean()
    assert agreement > 0.9, f"greedy agreement vs bf16 reference too low: {agreement}"
