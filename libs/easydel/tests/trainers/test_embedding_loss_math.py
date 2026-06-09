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

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from easydel.trainers.embedding_trainer._fn import _embedding_loss_values
from easydel.trainers.embedding_trainer.embedding_config import EmbeddingConfig
from easydel.trainers.embedding_trainer.embedding_trainer import EmbeddingTrainer


class _DummyEmbeddingModule:
    def __call__(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(embeddings=input_ids.astype(jnp.float32))


class _DummyPoolingFeature:
    def __init__(self, *, strategy, pad_token_id):
        self.strategy = strategy
        self.pad_token_id = pad_token_id


def test_embedding_triplet_matryoshka_without_negatives_keeps_zero_loss():
    batch = {
        "query_input_ids": jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32),
        "query_attention_mask": jnp.ones((2, 3), dtype=jnp.int32),
        "positive_input_ids": jnp.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], dtype=jnp.float32),
        "positive_attention_mask": jnp.ones((2, 3), dtype=jnp.int32),
    }

    loss, metrics = _embedding_loss_values(
        module=_DummyEmbeddingModule(),
        batch=batch,
        loss_type="triplet",
        temperature=0.05,
        margin=0.2,
        normalize=True,
        matryoshka_dims=[2, 3],
    )

    np.testing.assert_allclose(np.asarray(loss), np.asarray(0.0), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(metrics["fraction_active_triplets"]), np.asarray(0.0), rtol=0, atol=0)


def test_embedding_pooling_strategy_validates_known_values():
    assert EmbeddingConfig(pooling_strategy="mean").pooling_strategy == "mean"

    with pytest.raises(ValueError, match="pooling_strategy"):
        EmbeddingConfig(pooling_strategy="bad")


def test_embedding_pooling_strategy_updates_model_feature_only_when_set():
    model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=7),
        _pooling_feature=_DummyPoolingFeature(strategy="last", pad_token_id=7),
    )

    EmbeddingTrainer._apply_pooling_strategy(model, "mean")

    assert model._pooling_feature.strategy == "mean"
    assert model._pooling_feature.pad_token_id == 7


def test_embedding_pooling_strategy_requires_pooling_capable_model():
    with pytest.raises(ValueError, match="no embedding model"):
        EmbeddingTrainer._apply_pooling_strategy(None, "mean")

    with pytest.raises(ValueError, match="pooling feature"):
        EmbeddingTrainer._apply_pooling_strategy(SimpleNamespace(config=SimpleNamespace()), "mean")
