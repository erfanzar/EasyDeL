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

import jax
import jax.numpy as jnp

from easydel.utils.checkpoint_compat import (
    adapt_legacy_checkpoint_collections,
    legacy_checkpoint_key_aliases,
    materialize_tied_lm_head_from_embeddings,
    rename_legacy_checkpoint_leaves,
)


def test_legacy_checkpoint_key_aliases_cover_optimizer_key_format_changes():
    assert "tx.1.mu.lm_head.kernel.value" in legacy_checkpoint_key_aliases("tx.1.mu.parameters.lm_head.weight")
    assert "tx.1.mu.model.language_model.embed_tokens.embedding.value" in legacy_checkpoint_key_aliases(
        "tx.1.mu.parameters.model.language_model.embed_tokens.weight"
    )
    assert "tx.1.mu.model.language_model.layers.0.linear_attn.A_log.value" in legacy_checkpoint_key_aliases(
        "tx.1.mu.parameters.model.language_model.layers.0.linear_attn.A_log"
    )


def test_adapt_legacy_checkpoint_collections_wraps_bare_model_keys():
    state = {("model", "embed_tokens", "weight"): jnp.ones((2, 2))}
    required = {("parameters", "model", "embed_tokens", "weight")}

    adapted = adapt_legacy_checkpoint_collections(state, required)

    assert ("parameters", "model", "embed_tokens", "weight") in adapted
    assert jnp.allclose(
        adapted[("parameters", "model", "embed_tokens", "weight")], state[("model", "embed_tokens", "weight")]
    )


def test_adapt_legacy_checkpoint_collections_renames_params_collection():
    state = {("params", "lm_head", "kernel"): jnp.ones((2, 2))}
    required = {("parameters", "lm_head", "kernel")}

    adapted = adapt_legacy_checkpoint_collections(state, required)

    assert ("parameters", "lm_head", "kernel") in adapted


def test_rename_legacy_checkpoint_leaves_maps_old_suffixes_to_weight():
    state = {
        ("parameters", "dense", "kernel"): jnp.ones((2, 2)),
        ("parameters", "embed", "embedding"): jnp.ones((3, 2)),
        ("parameters", "norm", "scale"): jnp.ones((2,)),
    }

    renamed = rename_legacy_checkpoint_leaves(state)

    assert ("parameters", "dense", "weight") in renamed
    assert ("parameters", "embed", "weight") in renamed
    assert ("parameters", "norm", "weight") in renamed


def test_materialize_tied_lm_head_transposes_token_embedding():
    source_key = ("parameters", "model", "language_model", "embed_tokens", "weight")
    target_key = ("parameters", "lm_head", "weight")
    embedding = jnp.arange(12, dtype=jnp.float32).reshape(6, 2)
    state = {source_key: embedding}
    required = {target_key: jax.ShapeDtypeStruct((2, 6), jnp.float32)}

    materialized = materialize_tied_lm_head_from_embeddings(
        state,
        required,
        tie_word_embeddings=True,
    )

    assert target_key in materialized
    assert jnp.allclose(materialized[target_key], embedding.T)


def test_materialize_tied_lm_head_accepts_already_matching_shape():
    source_key = ("parameters", "model", "language_model", "embed_tokens", "weight")
    target_key = ("parameters", "lm_head", "weight")
    embedding = jnp.arange(12, dtype=jnp.float32).reshape(2, 6)
    state = {source_key: embedding}
    required = {target_key: jax.ShapeDtypeStruct((2, 6), jnp.float32)}

    materialized = materialize_tied_lm_head_from_embeddings(
        state,
        required,
        tie_word_embeddings=True,
    )

    assert jnp.allclose(materialized[target_key], embedding)


def test_materialize_tied_lm_head_prefers_text_embedding_over_vision_embedding():
    vision_key = ("parameters", "vision_tower", "patch_embedding", "weight")
    text_key = ("parameters", "model", "language_model", "embed_tokens", "weight")
    target_key = ("parameters", "lm_head", "weight")
    vision_embedding = jnp.full((6, 2), -1.0)
    text_embedding = jnp.arange(12, dtype=jnp.float32).reshape(6, 2)
    state = {vision_key: vision_embedding, text_key: text_embedding}
    required = {target_key: jax.ShapeDtypeStruct((2, 6), jnp.float32)}

    materialized = materialize_tied_lm_head_from_embeddings(
        state,
        required,
        tie_word_embeddings=True,
    )

    assert jnp.allclose(materialized[target_key], text_embedding.T)


def test_materialize_tied_lm_head_does_nothing_when_untied_or_existing():
    source_key = ("parameters", "model", "language_model", "embed_tokens", "weight")
    target_key = ("parameters", "lm_head", "weight")
    embedding = jnp.arange(12, dtype=jnp.float32).reshape(6, 2)
    existing = jnp.zeros((2, 6), dtype=jnp.float32)
    required = {target_key: jax.ShapeDtypeStruct((2, 6), jnp.float32)}

    untied = materialize_tied_lm_head_from_embeddings(
        {source_key: embedding},
        required,
        tie_word_embeddings=False,
    )
    existing_result = materialize_tied_lm_head_from_embeddings(
        {source_key: embedding, target_key: existing},
        required,
        tie_word_embeddings=True,
    )

    assert target_key not in untied
    assert jnp.allclose(existing_result[target_key], existing)
