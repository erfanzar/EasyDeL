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

import jax
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


def _prefix_merge_inputs(placeholder_count, capacity=6):
    """Build nonzero padding plus an independent flattened NumPy scatter oracle."""
    ids = np.zeros((2, 5), dtype=np.int32)
    positions = np.array([1, 4, 5, 7, 8, 9])[:placeholder_count]
    ids.reshape(-1)[positions] = PLACEHOLDER
    text = np.arange(40, dtype=np.float32).reshape(2, 5, 4) / 8
    features = np.full((capacity, 4), -1234.5, dtype=np.float32)
    features[:placeholder_count] = np.arange(placeholder_count * 4, dtype=np.float32).reshape(-1, 4) + 100
    expected = text.copy()
    expected.reshape(-1, 4)[positions] = features[:placeholder_count]
    return ids, text, features, expected


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit-dynamic-length"])
def test_explicit_valid_prefix_matches_numpy_scatter(compiled):
    """Only genuine prefix rows replace placeholders; nonzero tail rows never do."""

    def merge(ids, text, features, valid_length):
        return BaseVisionLanguageModule.merge_multimodal_embeddings(
            ids,
            text,
            features,
            PLACEHOLDER,
            multimodal_embeddings_valid_length=valid_length,
        )

    run = jax.jit(merge) if compiled else merge
    # Same capacity and shapes, changing data-dependent lengths (including no
    # images) through the same jitted callable. Preprocessing owns validation
    # of dynamic metadata, so only valid length/count pairs enter this path.
    for count in (3, 1, 0, 6):
        ids, text, features, expected = _prefix_merge_inputs(count)
        merged = run(jnp.asarray(ids), jnp.asarray(text), jnp.asarray(features), jnp.asarray(count, dtype=jnp.int32))
        assert merged.shape == text.shape
        assert merged.dtype == text.dtype
        np.testing.assert_array_equal(np.asarray(merged), expected)


@pytest.mark.parametrize("valid_length", [2, np.int32(2), np.int64(2), np.asarray(2, dtype=np.int32)])
def test_explicit_valid_prefix_accepts_concrete_integer_scalars(valid_length):
    ids, text, features, expected = _prefix_merge_inputs(2)
    merged = BaseVisionLanguageModule.merge_multimodal_embeddings(
        jnp.asarray(ids),
        jnp.asarray(text),
        jnp.asarray(features),
        PLACEHOLDER,
        multimodal_embeddings_valid_length=valid_length,
    )
    np.testing.assert_array_equal(np.asarray(merged), expected)


@pytest.mark.parametrize("valid_length", [0, 1, 3, 6])
def test_valid_prefix_rejects_genuine_count_mismatch_despite_spare_capacity(valid_length):
    """Capacity is not evidence of real features: two placeholders need n == 2."""
    ids, text, features, _ = _prefix_merge_inputs(2)
    with pytest.raises(ValueError, match=rf"2 placeholder.*{valid_length} multimodal"):
        BaseVisionLanguageModule.merge_multimodal_embeddings(
            jnp.asarray(ids),
            jnp.asarray(text),
            jnp.asarray(features),
            PLACEHOLDER,
            multimodal_embeddings_valid_length=jnp.asarray(valid_length, dtype=jnp.int32),
        )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "traced-inputs-concrete-metadata"])
@pytest.mark.parametrize(
    "valid_length, message",
    [
        (-1, "between 0 and capacity 6"),
        (7, "between 0 and capacity 6"),
        (np.uint64(2**63), "between 0 and capacity 6"),
        ([2], "scalar integer"),
        (np.asarray([[2]], dtype=np.int32), "scalar integer"),
        (2.0, "scalar integer"),
        (np.float32(2), "scalar integer"),
        (True, "scalar integer"),
        (np.bool_(False), "scalar integer"),
        (2 + 0j, "scalar integer"),
        ("2", "scalar integer"),
    ],
)
def test_valid_prefix_rejects_invalid_concrete_metadata(compiled, valid_length, message):
    """A traced placeholder count must not suppress concrete metadata checks."""
    ids, text, features, _ = _prefix_merge_inputs(2)

    def merge(ids, text, features):
        return BaseVisionLanguageModule.merge_multimodal_embeddings(
            ids,
            text,
            features,
            PLACEHOLDER,
            multimodal_embeddings_valid_length=valid_length,
        )

    run = jax.jit(merge) if compiled else merge
    with pytest.raises(ValueError, match=message):
        run(jnp.asarray(ids), jnp.asarray(text), jnp.asarray(features))


@pytest.mark.parametrize(
    "valid_length",
    [np.asarray([2], dtype=np.int32), np.asarray(2, dtype=np.float32), np.asarray(True)],
)
def test_valid_prefix_rejects_invalid_traced_metadata_shape_or_dtype(valid_length):
    ids, text, features, _ = _prefix_merge_inputs(2)

    @jax.jit
    def merge(ids, text, features, length):
        return BaseVisionLanguageModule.merge_multimodal_embeddings(
            ids,
            text,
            features,
            PLACEHOLDER,
            multimodal_embeddings_valid_length=length,
        )

    with pytest.raises(ValueError, match="scalar integer"):
        merge(jnp.asarray(ids), jnp.asarray(text), jnp.asarray(features), jnp.asarray(valid_length))


def test_padded_no_image_features_still_require_explicit_zero_length():
    """Opting out of prefix metadata retains the old exact-count contract."""
    ids, text, features, _ = _prefix_merge_inputs(0)
    with pytest.raises(ValueError, match=r"0 placeholder.*6 multimodal"):
        BaseVisionLanguageModule.merge_multimodal_embeddings(
            jnp.asarray(ids), jnp.asarray(text), jnp.asarray(features), PLACEHOLDER
        )
