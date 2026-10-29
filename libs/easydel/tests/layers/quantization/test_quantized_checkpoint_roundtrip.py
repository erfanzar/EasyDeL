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

"""Quantize, save, reload, dequantize — the full lifecycle of a quantized model.

Every step of this path was broken, each failure masking the next:

1. ``save_pretrained`` raised ``TypeError`` because ``QuantizationConfig`` had
   no JSON form, so no quantized model could be written at all.
2. ``ParallelLinearQuantized`` retained ``rngs`` as module state, leaving
   abstract PRNG leaves that the checkpoint writer could not shard.
3. The tensorstore reader cast **every** leaf to the model's ``param_dtype``,
   so ``uint32`` packed codes and ``uint8`` scales came back as ``bfloat16``
   — a checkpoint the writer itself could not read.
4. ``dequantize_modules`` called ``from_quantized(config=...)``, which is not
   that method's signature, so restoring full precision always raised.
5. The same call left ``quantization_config`` populated on a model that was no
   longer quantized.

The test asserts the two properties that matter — a reload is *bit-exact*
against what was saved, and dequantization restores dense layers — because
weaker assertions (no exception, right shapes) passed while the weights were
being silently destroyed.
"""

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.linears import ParallelLinear
from easydel.layers.linears._linear_quantized import ParallelLinearQuantized
from easydel.layers.quantization import EasyQuantizer, QuantizationConfig
from easydel.utils import set_inference_mode
from easydel.utils.traversals import iter_module_search


def _tiny_config():
    """Build a small Llama config whose dims suit group-32 quantization."""
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
    # Shard on fsdp rather than tp: this test is about the checkpoint
    # lifecycle, and column-parallel quantized matmul has its own
    # unrelated local-K constraint at tp>1 that would mask it.
    config.sharding_axis_dims = (1, 1, -1, 1, 1, 1)
    partition_axis = ed.PartitionAxis()
    partition_axis.hidden_state_axis = None
    config.partition_axis = partition_axis
    config.use_qmm_best_config = False
    return config


def _count(model, kind) -> int:
    """Count submodules of a given type.

    Args:
        model: Module tree to walk.
        kind: Type to count.

    Returns:
        Number of matching submodules.
    """
    return sum(1 for _, module in iter_module_search(model, spx.Module) if isinstance(module, kind))


def _leaves(model) -> dict[str, jax.Array]:
    """Flatten a model's state into dotted-path leaves.

    Args:
        model: Module to export.

    Returns:
        Mapping of dotted parameter path to array.
    """
    _, state = spx.export(model)
    return {
        ".".join(str(getattr(part, "key", part)) for part in path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]
    }


@pytest.fixture(scope="module")
def quantized_model():
    """Build a tiny model and quantize it to MXFP4."""
    config = _tiny_config()
    with set_inference_mode():
        model = ed.LlamaForCausalLM(
            config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0)
        ).shard_model()
        model = EasyQuantizer(QuantizationConfig(dtype="mxfp4", group_size=32)).apply_quantization(model)
    return config, model


class TestQuantizationConfigSerialization:
    """The config must survive the JSON boundary ``save_pretrained`` crosses."""

    def test_round_trips_through_a_dict(self):
        """``to_dict``/``from_dict`` preserve every field."""
        original = QuantizationConfig(dtype="mxfp4", group_size=32)
        restored = QuantizationConfig.from_dict(original.to_dict())
        assert restored.dtype is original.dtype
        assert restored.group_size == original.group_size

    def test_model_config_is_json_serializable_when_quantized(self, quantized_model):
        """A quantized model's config must serialize; it previously raised."""
        _, model = quantized_model
        assert "mxfp4" in model.config.to_json_string()


class TestQuantizedStateShape:
    """Quantized modules must not leak non-serializable state."""

    def test_no_abstract_leaves_remain(self, quantized_model):
        """Retained ``rngs`` left abstract leaves the writer could not shard."""
        _, model = quantized_model
        abstract = [path for path, leaf in _leaves(model).items() if isinstance(leaf, jax.ShapeDtypeStruct)]
        assert not abstract, f"abstract leaves would break save_pretrained: {abstract[:4]}"

    def test_packed_weights_use_integer_dtypes(self, quantized_model):
        """Packed codes and scales are integers, not floats."""
        _, model = quantized_model
        leaves = _leaves(model)
        kernels = [v for k, v in leaves.items() if k.endswith("quant_kernel")]
        scales = [v for k, v in leaves.items() if k.endswith("quant_scales")]
        assert kernels and scales
        assert all(k.dtype == jnp.uint32 for k in kernels)
        assert all(s.dtype == jnp.uint8 for s in scales)


class TestFullLifecycle:
    """Quantize, save, reload, dequantize."""

    def test_reload_is_bit_exact_and_dequantize_restores_dense(self, tmp_path, quantized_model):
        """The whole path, asserted on values rather than on absence of errors."""
        config, model = quantized_model
        quantized_count = _count(model, ParallelLinearQuantized)
        assert quantized_count > 0

        ids = jax.random.randint(jax.random.PRNGKey(1), (1, 16), 0, config.vocab_size)
        with model.mesh:
            quantized_logits = jax.device_get(model(input_ids=ids).logits.astype(jnp.float32))
        assert np.isfinite(quantized_logits).all()

        save_dir = tmp_path / "mxfp4_model"
        model.save_pretrained(str(save_dir))

        with set_inference_mode():
            reloaded = ed.LlamaForCausalLM.from_pretrained(
                str(save_dir),
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
                sharding_axis_dims=(1, 1, -1, 1, 1, 1),
                partition_axis=config.partition_axis,
                auto_shard_model=True,
                config_kwargs={"use_qmm_best_config": False},
                verbose=False,
            )

        # The checkpoint declares its own quantization, so the structure is
        # rebuilt quantized without the caller asking for it.
        assert _count(reloaded, ParallelLinearQuantized) == quantized_count

        reloaded_leaves = _leaves(reloaded)
        assert all(v.dtype == jnp.uint32 for k, v in reloaded_leaves.items() if k.endswith("quant_kernel"))
        assert all(v.dtype == jnp.uint8 for k, v in reloaded_leaves.items() if k.endswith("quant_scales"))

        with reloaded.mesh:
            reloaded_logits = jax.device_get(reloaded(input_ids=ids).logits.astype(jnp.float32))
        np.testing.assert_array_equal(reloaded_logits, quantized_logits)

        with set_inference_mode():
            dense = EasyQuantizer(QuantizationConfig(dtype="mxfp4", group_size=32)).dequantize_modules(reloaded)

        assert _count(dense, ParallelLinearQuantized) == 0
        assert _count(dense, ParallelLinear) >= quantized_count
        assert dense.config.quantization_config is None

        with dense.mesh:
            dense_logits = jax.device_get(dense(input_ids=ids).logits.astype(jnp.float32))
        # Dequantizing recovers the values the quantized model was computing
        # with, so the outputs must agree closely.
        assert np.abs(dense_logits - quantized_logits).max() < 1e-2
