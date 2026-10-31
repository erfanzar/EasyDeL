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

"""Accelerator regression tests for ``easydel.modules._base`` features and Arctic (array computation)."""

import easydel as ed
import jax
import numpy as np
import spectrax as spx
from easydel.modules._base import MultiModalMergeFeature, SequenceLengthPoolingFeature
from jax import numpy as jnp


def test_max_pooling_ignores_padding_positions():
    hidden = jnp.array(
        [
            [[1.0, 5.0], [2.0, 1.0], [100.0, 100.0]],
            [[3.0, 3.0], [4.0, 0.0], [5.0, 9.0]],
        ]
    )
    mask = jnp.array([[1, 1, 0], [1, 1, 1]])
    pool = SequenceLengthPoolingFeature(strategy="max", pad_token_id=0)
    out = np.asarray(pool.pool(hidden, attention_mask=mask))
    np.testing.assert_allclose(out, np.array([[2.0, 5.0], [5.0, 9.0]]))
    # Without a mask every position participates.
    np.testing.assert_allclose(np.asarray(pool.pool(hidden))[0], np.array([100.0, 100.0]))


def test_max_pooling_all_padding_row_is_zero():
    hidden = jnp.array([[[-3.0, -4.0], [-1.0, -2.0]]])
    out = SequenceLengthPoolingFeature(strategy="max", pad_token_id=0).pool(
        hidden, attention_mask=jnp.zeros((1, 2), dtype=jnp.int32)
    )
    np.testing.assert_allclose(np.asarray(out), np.zeros((1, 2)))


def test_multimodal_merge_feature_handles_batches_and_empty_features():
    merge = MultiModalMergeFeature(strategy="placeholder", image_token_id=9)
    input_ids = jnp.array([[1, 9, 9, 2], [9, 3, 9, 4]])
    embeds = jnp.zeros((2, 4, 3))
    feats = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) + 1.0
    out = np.asarray(merge.merge(input_ids, embeds, feats))
    np.testing.assert_allclose(out[0, 1], np.asarray(feats[0]))
    np.testing.assert_allclose(out[0, 2], np.asarray(feats[1]))
    np.testing.assert_allclose(out[1, 0], np.asarray(feats[2]))
    np.testing.assert_allclose(out[1, 2], np.asarray(feats[3]))
    np.testing.assert_allclose(out[0, 0], 0.0)
    # No features and no placeholders: a no-op, not an out-of-bounds gather.
    no_img = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    out2 = merge.merge(no_img, embeds + 1.0, jnp.zeros((0, 3)))
    np.testing.assert_allclose(np.asarray(out2), np.ones((2, 4, 3)))


def _arctic_aux_loss(coef: float) -> jax.Array:
    config = ed.ArcticConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_layer_frequency=1,
        router_aux_loss_coef=coef,
        output_router_logits=True,
    )
    config.moe_force_xla_gmm = True
    model = ed.ArcticForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8) % 100
    out = model(input_ids=ids, attention_mask=jnp.ones_like(ids), output_router_logits=True)
    return out.aux_loss


def test_arctic_aux_loss_is_scaled_by_router_aux_loss_coef():
    base = float(_arctic_aux_loss(1.0))
    scaled = float(_arctic_aux_loss(0.25))
    assert base > 0.0
    np.testing.assert_allclose(scaled, 0.25 * base, rtol=1e-4)
