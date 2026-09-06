# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Vision-language families share one multimodal-merge implementation.

Four families carried their own copy of the cumsum-gather merge that splices
vision embeddings into the text stream at placeholder positions. Three matched
``BaseVisionLanguageModule.merge_multimodal_embeddings`` exactly; the fourth,
``qwen3_omni_moe``, had drifted into a shape bug that only manifested above
batch size 1:

``jnp.cumsum`` with no ``axis`` flattens, so ``update_values`` came out
``(batch * seq, hidden)`` while the ``jnp.where`` condition stayed
``(batch, seq, 1)``. At batch 1 the two broadcast together by coincidence; at
batch 2 and above the merge raised ``ValueError``.

All four now delegate to the shared implementation. These tests pin both the
delegation and the batch behaviour that the drifted copy got wrong.
"""

import importlib

import numpy as np
import pytest
from easydel.modules._base.vision_language_module import BaseVisionLanguageModule
from jax import numpy as jnp

FAMILIES = ("qwen2_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_omni_moe")
PLACEHOLDER = 7


def _merge_fn(family: str):
    module = importlib.import_module(f"easydel.modules.{family}.modeling_{family}")
    return module.merge_multimodal_embeddings


def _inputs(batch: int, seq: int = 9, hidden: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    input_ids = jnp.asarray(rng.integers(0, 10, (batch, seq)), jnp.int32)
    embeds = jnp.asarray(rng.standard_normal((batch, seq, hidden)), jnp.float32)
    count = int(jnp.sum(input_ids == PLACEHOLDER))
    vision = jnp.asarray(rng.standard_normal((count, hidden)), jnp.float32)
    return input_ids, embeds, vision


@pytest.mark.parametrize("family", FAMILIES)
def test_family_delegates_to_shared_merge(family):
    """No family may carry its own copy of the merge again."""
    assert _merge_fn(family) is BaseVisionLanguageModule.merge_multimodal_embeddings


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_merge_works_above_batch_one(family, batch):
    """Regression: the drifted copy raised for every batch above 1."""
    input_ids, embeds, vision = _inputs(batch)
    merged = _merge_fn(family)(input_ids, embeds, vision, PLACEHOLDER)
    assert merged.shape == embeds.shape


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_merge_places_vision_rows_in_order(batch):
    """Placeholder slots take vision rows left-to-right; others are untouched."""
    input_ids, embeds, vision = _inputs(batch)
    merged = BaseVisionLanguageModule.merge_multimodal_embeddings(input_ids, embeds, vision, PLACEHOLDER)

    mask = np.asarray(input_ids == PLACEHOLDER)
    merged_np, embeds_np, vision_np = np.asarray(merged), np.asarray(embeds), np.asarray(vision)

    # Non-placeholder positions keep their original text embedding.
    assert np.array_equal(merged_np[~mask], embeds_np[~mask])

    # Placeholder positions consume vision rows in flattened scan order.
    taken = merged_np[mask]
    assert np.array_equal(taken, vision_np[: taken.shape[0]])


def test_merge_accepts_multiple_placeholder_ids():
    """The list form (image + video ids) selects both token kinds."""
    input_ids, embeds, _ = _inputs(2)
    ids = [7, 8]
    count = int(jnp.sum(jnp.isin(input_ids, jnp.asarray(ids))))
    rng = np.random.default_rng(1)
    vision = jnp.asarray(rng.standard_normal((count, embeds.shape[-1])), jnp.float32)

    merged = BaseVisionLanguageModule.merge_multimodal_embeddings(input_ids, embeds, vision, ids)

    mask = np.asarray(jnp.isin(input_ids, jnp.asarray(ids)))
    assert merged.shape == embeds.shape
    assert np.array_equal(np.asarray(merged)[~mask], np.asarray(embeds)[~mask])


def test_merge_is_a_noop_without_placeholders():
    """With no placeholder tokens the text embeddings pass through unchanged."""
    rng = np.random.default_rng(3)
    input_ids = jnp.zeros((3, 5), jnp.int32)  # no PLACEHOLDER anywhere
    embeds = jnp.asarray(rng.standard_normal((3, 5, 4)), jnp.float32)
    vision = jnp.empty((0, 4), jnp.float32)

    merged = BaseVisionLanguageModule.merge_multimodal_embeddings(input_ids, embeds, vision, PLACEHOLDER)
    assert jnp.array_equal(merged, embeds)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_zero_image_batch_is_a_true_noop(batch):
    """An all-text batch hands in a literal ``(0, hidden)`` array, not ``None``.

    The bucket collator emits ``image_embeds=(0, hidden)`` for a text-only
    batch, and the forward gate is ``if image_embeds is not None`` — so the
    empty array *enters* the merge and must pass straight through.

    This is what forces the pad row to be built at an explicit ``(1, hidden)``
    shape. The ``zeros_like(multimodal_embeddings[0:1])`` spelling collapses to
    ``(0, hidden)`` on an empty input, leaving ``flattened_padded`` with no
    index-0 slot and turning the cumsum gather into an out-of-bounds read
    (``TypeError: Slice size at index 0 in gather op is out of range``).
    """
    rng = np.random.default_rng(batch)
    hidden = 4
    input_ids = jnp.zeros((batch, 5), jnp.int32)  # no placeholders: all text
    embeds = jnp.asarray(rng.standard_normal((batch, 5, hidden)), jnp.float32)
    empty = jnp.zeros((0, hidden), jnp.float32)

    merged = BaseVisionLanguageModule.merge_multimodal_embeddings(input_ids, embeds, empty, PLACEHOLDER)

    assert merged.shape == embeds.shape
    assert jnp.array_equal(merged, embeds), "empty image_embeds altered the text embeddings"


@pytest.mark.parametrize("family", FAMILIES)
def test_families_survive_the_zero_image_batch(family):
    """Every family routed onto the shared merge must handle the empty case.

    These four previously carried their own copy which built the pad row
    explicitly; routing them onto a base that did not would have regressed the
    text-only lane silently.
    """
    hidden = 4
    rng = np.random.default_rng(0)
    input_ids = jnp.zeros((2, 5), jnp.int32)
    embeds = jnp.asarray(rng.standard_normal((2, 5, hidden)), jnp.float32)
    empty = jnp.zeros((0, hidden), jnp.float32)

    merged = _merge_fn(family)(input_ids, embeds, empty, PLACEHOLDER)
    assert jnp.array_equal(merged, embeds)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


@pytest.mark.parametrize("feature_count", [1, 3])
def test_merge_rejects_placeholder_feature_count_mismatch(feature_count):
    ids = jnp.array([[1, 9, 9, 2]], jnp.int32)
    text = jnp.zeros((1, 4, 8), jnp.float32)
    visual = jnp.zeros((feature_count, 8), jnp.float32)
    with pytest.raises(ValueError, match=rf"2 placeholder.*{feature_count} multimodal"):
        BaseVisionLanguageModule.merge_multimodal_embeddings(ids, text, visual, 9)
