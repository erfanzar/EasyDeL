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

"""DFlashModel builds DFlash-specific layers while keeping the DSpark parameter layout (run on TPU/GPU)."""

import numpy as np
import spectrax as spx
from easydel.modules.dflash import DFlashAttention, DFlashConfig, DFlashDecoderLayer, DFlashModel
from easydel.modules.dspark.modeling_dspark import DSparkDecoderLayer
from jax import numpy as jnp


def _config():
    return DFlashConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        target_hidden_size=64,
        target_layer_ids=(0, 1),
        block_size=3,
        num_anchors=4,
        mask_token_id=127,
    )


def test_dflash_model_instantiates_dflash_layers():
    model = DFlashModel(config=_config(), rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    assert len(model.layers) == 2
    for layer in model.layers:
        assert isinstance(layer, DFlashDecoderLayer)
        assert isinstance(layer.self_attn, DFlashAttention)
        # DSpark-compatible parameter layout is preserved.
        assert isinstance(layer, DSparkDecoderLayer)
        for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
            assert hasattr(layer, name)

    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    context = jnp.ones((1, 3, 128), dtype=jnp.float32)
    out = model(input_ids=input_ids, target_hidden_states=context)
    assert out.logits.shape == (1, 3, 128)
    assert out.hidden_states.shape == (1, 3, 64)
    assert out.confidence_logits is None
    assert np.isfinite(np.asarray(out.logits)).all()
