# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import spectrax as spx
from easydel.layers.embeddings import NGramEmbed


def _config(**overrides):
    values = {
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "vocab_size": 100,
        "ngram_vocab_size_base": 11,
        "make_ngram_vocab_size_divisible_by": 4,
        "split_ngram_parts": 2,
        "seed": 0,
        "eos_token_id": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ngram_table_uses_configured_feature_sharding_axis():
    module = NGramEmbed(
        _config(ngram_sharding_axis="ep"),
        embedding_dim=8,
        rngs=spx.Rngs(0),
    )

    assert module.shards[0].sharding.axis_names == (None, "ep")


def test_packed_segments_do_not_form_cross_document_ngrams():
    from jax import numpy as jnp

    module = NGramEmbed(_config(), embedding_dim=8, rngs=spx.Rngs(0))
    ids = jnp.array([[1, 2, 3, 4]], jnp.int32)
    segments = jnp.array([[0, 0, 1, 1]], jnp.int32)
    packed = module.hash_ids(ids, segment_ids=segments)
    first = module.hash_ids(ids[:, :2])
    second = module.hash_ids(ids[:, 2:])
    assert jnp.array_equal(packed[:, :2], first)
    assert jnp.array_equal(packed[:, 2:], second)


def test_cached_ngram_context_preserves_segment_membership():
    from jax import numpy as jnp

    module = NGramEmbed(_config(), embedding_dim=8, rngs=spx.Rngs(0))
    context = jnp.array([[1, 2]], jnp.int32)
    token = jnp.array([[3]], jnp.int32)
    segment = jnp.array([[7]], jnp.int32)
    context_segments = jnp.array([[7, 7]], jnp.int32)
    cached = module.hash_ids(
        jnp.concatenate([context, token], axis=1),
        segment_ids=jnp.concatenate([context_segments, segment], axis=1),
    )[:, -1:]
    rows = module.hash_ids(
        jnp.concatenate([context, token], axis=1),
        segment_ids=jnp.concatenate([context_segments, segment], axis=1),
    )[:, -1:]
    assert jnp.array_equal(cached, rows)
    got = module(
        token,
        context=context,
        segment_ids=segment,
        context_segment_ids=context_segments,
    )
    assert jnp.array_equal(got, module.lookup(rows))
