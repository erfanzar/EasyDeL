"""CPU-only forward-protocol regressions using a NumPy backbone and head.

No parameters are initialized and no JAX array operations run. The real task
forward, output class, and pooling implementations are exercised together.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.infra.modeling_outputs import MoeModelOutput
from easydel.modules._base import _features
from easydel.modules._base._features import SequenceLengthPoolingFeature
from easydel.modules.qwen3_moe import Qwen3MoeForSequenceClassification

pytestmark = pytest.mark.cpu_ok


def _model(monkeypatch):
    monkeypatch.setattr(_features, "jnp", np)
    # Bypass numerical parameter initialization, not the public forward path.
    model = object.__new__(Qwen3MoeForSequenceClassification)
    attributes = {
        "config": SimpleNamespace(pad_token_id=0),
        "_base_model_name": "model",
        "_score_head_name": "score",
        "_pooling_feature": SequenceLengthPoolingFeature(strategy="last", pad_token_id=0),
        "_router_aux_loss_feature": None,
        "model": lambda **kwargs: MoeModelOutput(last_hidden_state=kwargs["inputs_embeds"]),
        "score": lambda hidden: hidden[..., :2],
    }
    for name, value in attributes.items():
        object.__setattr__(model, name, value)
    return model


@pytest.mark.parametrize("batch_size", [1, 2])
def test_embeddings_without_ids_or_mask_pool_the_last_position(monkeypatch, batch_size):
    model = _model(monkeypatch)
    embeds = np.arange(batch_size * 4 * 3).reshape(batch_size, 4, 3)

    output = model.forward(inputs_embeds=embeds)

    np.testing.assert_array_equal(output.logits, embeds[:, -1, :2])


def test_attention_mask_still_selects_the_rightmost_valid_embedding(monkeypatch):
    model = _model(monkeypatch)
    embeds = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    # Right padding and left padding plus a trailing masked position.
    mask = np.array([[1, 1, 0, 0], [0, 1, 1, 0]])

    output = model.forward(inputs_embeds=embeds, attention_mask=mask)

    np.testing.assert_array_equal(output.logits, np.stack([embeds[0, 1, :2], embeds[1, 2, :2]]))


def test_token_ids_retain_padding_aware_pooling(monkeypatch):
    model = _model(monkeypatch)
    embeds = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    ids = np.array([[3, 4, 0, 0], [5, 6, 7, 0]])

    output = model.forward(input_ids=ids, inputs_embeds=embeds)

    np.testing.assert_array_equal(output.logits, np.stack([embeds[0, 1, :2], embeds[1, 2, :2]]))


def test_head_skipping_remains_available_for_unmasked_embeddings(monkeypatch):
    model = _model(monkeypatch)
    embeds = np.arange(12).reshape(1, 4, 3)

    output = model.forward(inputs_embeds=embeds, apply_lm_head=False)

    assert output.logits is None
