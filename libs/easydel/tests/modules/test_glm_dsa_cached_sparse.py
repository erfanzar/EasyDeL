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

"""Public cached/packed GLM sparse attention parity (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import TransformerCacheView, TransformerMetadata
from easydel.modules.glm5_next.modeling_glm5_next import Glm5NextForCausalLM, Glm5NextTextModel
from easydel.modules.glm5_next.modeling_glm5_next_mtp import Glm5NextMTPHead
from easydel.modules.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaAttention,
    GlmMoeDsaForCausalLM,
    GlmMoeDsaModel,
)
from ejkernel.types import MaskInfo
from jax import lax
from jax import numpy as jnp
from spectrax.common_types import MODE_PREFILL

from .test_glm_dsa_forward_contracts import _config, _layer


def _model(family, config, causal_lm=False):
    model_type = (
        (Glm5NextForCausalLM if causal_lm else Glm5NextTextModel)
        if family == "glm5_next"
        else (GlmMoeDsaForCausalLM if causal_lm else GlmMoeDsaModel)
    )
    return model_type(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )


def _view(model, batch=1, capacity=16):
    config = model.create_transformer_cache_config(batch, capacity)
    return TransformerCacheView.init(
        config,
        layer_index=0,
        mesh=model.config.mesh,
        dtype=jnp.float32,
        runtime_sharding_resolver=model.config.runtime_sharding_resolver,
    )


def _hidden(batch, length):
    values = np.arange(batch * length * 16, dtype=np.float32).reshape(batch, length, 16)
    return jnp.asarray(np.sin(values * 0.37) + np.cos(values * 0.19))


def _frequencies(family):
    if family == "glm5_next":
        return None
    # A nontrivial analytic rotary table, independently of the model's table builder.
    angle = np.arange(32, dtype=np.float32)[:, None] * 0.23
    return jnp.asarray(np.concatenate([np.cos(angle), np.sin(angle)], axis=-1))


@pytest.mark.parametrize("family", ["glm5_next", "glm_moe_dsa"])
@pytest.mark.parametrize("causal_lm", [False, True])
def test_public_cache_builder_preallocates_indexer_features(family, causal_lm):
    config = _config(family, dsa_sparse_training=False)
    with config.mesh:
        model = _model(family, config, causal_lm=causal_lm)
        cache = model.init_cache(batch_size=1, max_length=16)
    view = cache.views[0]
    width = 2 * config.index_head_dim + 1 if family == "glm5_next" else config.index_head_dim
    assert view.read_auxiliary().shape == (1, 16, width)
    assert view.read_auxiliary().dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(view.read_auxiliary()), 0)


@pytest.mark.parametrize("family", ["glm5_next", "glm_moe_dsa"])
def test_cached_sparse_prefill_multitoken_continuation_decode_and_reset(family):
    config = _config(family, index_topk=2, dsa_sparse_training=False)
    with config.mesh, jax.default_matmul_precision("highest"):
        model = _model(family, config)
        layer = _layer(family, config)
        layer.eval()
        view = _view(model)
        initial_structure = jax.tree.structure(view)
        hidden = _hidden(1, 10)
        positions = jnp.arange(10, dtype=jnp.int32)[None]
        frequencies = _frequencies(family)
        full = layer(
            hidden,
            MaskInfo.from_segments(jnp.zeros((1, 10), jnp.int32)),
            positions,
            MODE_PREFILL,
            output_attentions=True,
            frequencies=frequencies,
        )
        # Positional JIT arguments avoid pjit/in_shardings keyword-argument restrictions.
        step = jax.jit(
            lambda cache, x, pos, mask: layer(
                x, mask, pos, MODE_PREFILL, cache, output_attentions=True, frequencies=frequencies
            )
        )
        begin = 0
        for size in (6, 3, 1):
            result = step(
                view,
                hidden[:, begin : begin + size],
                positions[:, begin : begin + size],
                MaskInfo.from_segments(jnp.zeros((1, size), jnp.int32)),
            )
            np.testing.assert_allclose(
                np.asarray(result.attention_output),
                np.asarray(full.attention_output[:, begin : begin + size]),
                rtol=2e-5,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                np.asarray(result.attention_weight[..., :10]),
                np.asarray(full.attention_weight[:, :, begin : begin + size]),
                rtol=2e-5,
                atol=2e-6,
            )
            assert np.all(np.asarray(result.attention_weight[..., 10:]) == 0)
            view = result.cache_view
            assert jax.tree.structure(view) == initial_structure
            assert view.read_auxiliary().shape[1] == 16
            begin += size
            np.testing.assert_array_equal(np.asarray(view.indexes), [begin])
        # This must actually be sparse beyond top-k, not merely cached dense parity.
        assert np.count_nonzero(np.asarray(full.attention_weight)[0, 0, -1]) < 10
        graph, state = spx.export(layer)
        restored = spx.bind(graph, state)
        cleared = view.reset()
        assert jax.tree.structure(cleared) == initial_structure
        np.testing.assert_array_equal(np.asarray(cleared.read_auxiliary()), 0)
        replay = restored(
            hidden,
            MaskInfo.from_segments(jnp.zeros((1, 10), jnp.int32)),
            positions,
            MODE_PREFILL,
            cleared,
            output_attentions=True,
            frequencies=frequencies,
        )
        np.testing.assert_allclose(np.asarray(replay.attention_output), np.asarray(full.attention_output), atol=2e-6)


@pytest.mark.parametrize("family", ["glm5_next", "glm_moe_dsa"])
def test_cached_indexer_uses_each_rows_postpadded_offsets(family):
    config = _config(family, index_topk=2, dsa_sparse_training=False)
    with config.mesh, jax.default_matmul_precision("highest"):
        model = _model(family, config)
        layer = _layer(family, config)
        layer.eval()
        view = _view(model, batch=2)
        hidden = _hidden(2, 9)
        segments = jnp.asarray([[0, 0, 0, 0, 0, -1], [0, 0, 0, -1, -1, -1]], jnp.int32)
        frequencies = _frequencies(family)
        prefill = layer(
            hidden[:, :6],
            MaskInfo.from_segments(segments),
            jnp.broadcast_to(jnp.arange(6)[None], (2, 6)),
            MODE_PREFILL,
            view,
            TransformerMetadata(postpadded=True),
            frequencies=frequencies,
        )
        np.testing.assert_array_equal(np.asarray(prefill.cache_view.indexes), [5, 3])
        continuation = hidden[:, 6:9]
        positions = jnp.asarray([[5, 6, 7], [3, 4, 5]], jnp.int32)
        result = layer(
            continuation,
            MaskInfo.from_segments(jnp.zeros((2, 3), jnp.int32)),
            positions,
            MODE_PREFILL,
            prefill.cache_view,
            TransformerMetadata(postpadded=True),
            output_attentions=True,
            frequencies=frequencies,
        )
        np.testing.assert_array_equal(np.asarray(result.cache_view.indexes), [8, 6])
        for row, prefix in enumerate((5, 3)):
            separate = jnp.concatenate([hidden[row : row + 1, :prefix], continuation[row : row + 1]], axis=1)
            reference = layer(
                separate,
                MaskInfo.from_segments(jnp.zeros((1, prefix + 3), jnp.int32)),
                jnp.arange(prefix + 3)[None],
                MODE_PREFILL,
                output_attentions=True,
                frequencies=frequencies,
            )
            np.testing.assert_allclose(
                np.asarray(result.attention_output[row]),
                np.asarray(reference.attention_output[0, -3:]),
                rtol=2e-5,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                np.asarray(result.attention_weight[row, :, :, : prefix + 3]),
                np.asarray(reference.attention_weight[0, :, -3:]),
                rtol=2e-5,
                atol=2e-6,
            )
            assert np.all(np.asarray(result.attention_weight[row, :, :, prefix + 3 :]) == 0)


@pytest.mark.parametrize("mask_representation", ["segments", "pairwise", "causal_pairwise"])
def test_glm5_packed_pools_match_independent_documents_and_counterfactual(mask_representation):
    config = _config("glm5_next", index_topk=2, index_kpool=2, dsa_sparse_training=False)
    with config.mesh, jax.default_matmul_precision("highest"):
        layer = _layer("glm5_next", config)
        layer.eval()
        hidden = _hidden(1, 12)
        # Neither boundary is pool-aligned, and both documents exceed the top-k budget.
        segments = jnp.asarray([[0] * 5 + [1] * 7], jnp.int32)
        if mask_representation == "segments":
            packed_mask = MaskInfo.from_segments(segments)
        else:
            visibility = segments[:, :, None] == segments[:, None, :]
            if mask_representation == "causal_pairwise":
                visibility = visibility & jnp.tril(jnp.ones((12, 12), dtype=jnp.bool_))
            packed_mask = MaskInfo.from_attention_mask(visibility[:, None])
        positions = jnp.asarray([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]], jnp.int32)
        step = jax.jit(lambda x, mask: layer(x, mask, positions, MODE_PREFILL, output_attentions=True))
        result = step(hidden, packed_mask)
        # The pool [5, 6] must exist: global (padding-only) labels instead group
        # [4, 5] and [6, 7], neither fully visible from query 6.
        assert np.all(np.asarray(result.attention_weight[:, :, 6, 5:7]) > 0)
        model = _model("glm5_next", config)
        cached = layer(hidden, packed_mask, positions, MODE_PREFILL, _view(model), output_attentions=True)
        np.testing.assert_allclose(
            np.asarray(cached.attention_output), np.asarray(result.attention_output), rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(
            np.asarray(cached.attention_weight[..., :12]),
            np.asarray(result.attention_weight),
            rtol=2e-5,
            atol=2e-6,
        )
        assert np.all(np.asarray(cached.attention_weight[..., 12:]) == 0)
        changed = hidden.at[:, :5].set(hidden[:, :5] * -71.0 + 17.0)
        counterfactual = step(changed, packed_mask)
        np.testing.assert_array_equal(
            np.asarray(result.attention_output[:, 5:]), np.asarray(counterfactual.attention_output[:, 5:])
        )
        for begin, end in ((0, 5), (5, 12)):
            separate = layer(
                hidden[:, begin:end],
                MaskInfo.from_segments(jnp.zeros((1, end - begin), jnp.int32)),
                positions[:, begin:end],
                MODE_PREFILL,
                output_attentions=True,
            )
            np.testing.assert_allclose(
                np.asarray(result.attention_output[:, begin:end]),
                np.asarray(separate.attention_output),
                rtol=2e-5,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                np.asarray(result.attention_weight[:, :, begin:end, begin:end]),
                np.asarray(separate.attention_weight),
                rtol=2e-5,
                atol=2e-6,
            )
        assert np.all(np.asarray(result.attention_weight[:, :, 5:, :5]) == 0)


@pytest.mark.parametrize("mask_representation", ["padding", "segments", "pairwise"])
@pytest.mark.parametrize("override_indexes", [False, True])
def test_glm5_cached_full_history_mask_matches_query_local_mask(mask_representation, override_indexes):
    _check_cached_full_history_mask("glm5_next", mask_representation, override_indexes)


@pytest.mark.parametrize("mask_representation", ["padding", "segments", "pairwise"])
@pytest.mark.parametrize("override_indexes", [False, True])
def test_glm_moe_dsa_cached_full_history_mask_matches_query_local_mask(mask_representation, override_indexes):
    _check_cached_full_history_mask("glm_moe_dsa", mask_representation, override_indexes)


def _check_cached_full_history_mask(family, mask_representation, override_indexes):
    config = _config(family, index_topk=2, dsa_sparse_training=False)
    with config.mesh, jax.default_matmul_precision("highest"):
        model = _model(family, config)
        layer = _layer(family, config)
        layer.eval()
        frequencies = _frequencies(family)
        hidden = _hidden(2, 10)
        # Both rows start with padding, but their current queries are all live.
        # Taking the historical prefix instead of the cache-aligned query rows
        # would suppress valid continuation features and their top-k selection.
        segments = jnp.asarray([[-1, -1] + [0] * 8, [-1] + [0] * 9], jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(10)[None], (2, 10))
        if mask_representation == "segments":
            absolute_mask = MaskInfo.from_segments(segments)
        elif mask_representation == "padding":
            absolute_mask = MaskInfo.from_attention_mask(segments >= 0)
        else:
            valid = segments >= 0
            visible = valid[:, :, None] & valid[:, None, :] & jnp.tril(jnp.ones((10, 10), jnp.bool_))
            absolute_mask = MaskInfo.from_attention_mask(visible[:, None])
        full = layer(hidden, absolute_mask, positions, MODE_PREFILL, output_attentions=True, frequencies=frequencies)
        initial = _view(model, batch=2)
        prefix = layer(hidden[:, :6], absolute_mask, positions[:, :6], MODE_PREFILL, initial, frequencies=frequencies)
        absolute_view = prefix.cache_view
        local_view = prefix.cache_view
        step = jax.jit(
            lambda cache, x, pos, mask, metadata: layer(
                x, mask, pos, MODE_PREFILL, cache, metadata, output_attentions=True, frequencies=frequencies
            )
        )
        begin = 6
        for size in (3, 1):
            metadata = None
            if override_indexes:
                metadata = TransformerMetadata(indexes=jnp.full((2,), begin, jnp.int32))
                # The explicit metadata, not these deliberately stale counters,
                # owns both the auxiliary append and the matching KV append.
                absolute_view = absolute_view.replace(indexes=jnp.zeros((2,), jnp.int32))
                local_view = local_view.replace(indexes=jnp.zeros((2,), jnp.int32))
            result = step(
                absolute_view,
                hidden[:, begin : begin + size],
                positions[:, begin : begin + size],
                absolute_mask,
                metadata,
            )
            local = step(
                local_view,
                hidden[:, begin : begin + size],
                positions[:, begin : begin + size],
                MaskInfo.from_segments(jnp.zeros((2, size), jnp.int32)),
                metadata,
            )
            for actual in (result, local):
                np.testing.assert_allclose(
                    np.asarray(actual.attention_output),
                    np.asarray(full.attention_output[:, begin : begin + size]),
                    rtol=2e-5,
                    atol=2e-6,
                )
                np.testing.assert_allclose(
                    np.asarray(actual.attention_weight[..., :10]),
                    np.asarray(full.attention_weight[:, :, begin : begin + size]),
                    rtol=2e-5,
                    atol=2e-6,
                )
                weights = np.asarray(actual.attention_weight)
                future = np.arange(weights.shape[-1])[None, :] > begin + np.arange(size)[:, None]
                assert np.all(np.where(future[None, None], weights, 0) == 0)
                np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)
                assert np.all(weights[..., 10:] == 0)
                np.testing.assert_array_equal(np.asarray(actual.cache_view.indexes), [begin + size] * 2)
            np.testing.assert_array_equal(
                np.asarray(result.cache_view.read_auxiliary()), np.asarray(local.cache_view.read_auxiliary())
            )
            absolute_view, local_view = result.cache_view, local.cache_view
            begin += size
        assert np.count_nonzero(np.asarray(full.attention_weight)[0, 0, -1]) < 8


@pytest.mark.parametrize("mask_representation", ["padding", "segments", "pairwise", "packed"])
def test_glm_moe_dsa_no_cache_shared_selection_preserves_visibility(mask_representation):
    config = _config(
        "glm_moe_dsa", num_hidden_layers=2, indexer_types=["full", "shared"], index_topk=4, dsa_sparse_training=False
    )
    with config.mesh, jax.default_matmul_precision("highest"):
        full_layer = _layer("glm_moe_dsa", config)
        shared_layer = GlmMoeDsaAttention(
            config,
            layer_idx=1,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
            rngs=spx.Rngs(1),
        )
        full_layer.eval()
        shared_layer.eval()
        hidden = _hidden(1, 8)
        segments = jnp.asarray([[0] * 4 + ([1] * 4 if mask_representation == "packed" else [0] * 4)], jnp.int32)
        positions = jnp.asarray([[0, 1, 2, 3, 0, 1, 2, 3] if mask_representation == "packed" else list(range(8))])
        if mask_representation == "padding":
            mask = MaskInfo.from_attention_mask(jnp.ones((1, 8), jnp.bool_))
        elif mask_representation == "pairwise":
            mask = MaskInfo.from_attention_mask(jnp.ones((1, 1, 8, 8), jnp.bool_))
        else:
            mask = MaskInfo.from_segments(segments)
        frequencies = _frequencies("glm_moe_dsa")
        full, selected = full_layer.forward_mla(
            hidden,
            mask,
            positions,
            MODE_PREFILL,
            output_attentions=True,
            frequencies=frequencies,
            return_topk_indices=True,
        )
        shared, reused = shared_layer.forward_mla(
            hidden,
            mask,
            positions,
            MODE_PREFILL,
            output_attentions=True,
            frequencies=frequencies,
            previous_topk_indices=selected,
            return_topk_indices=True,
        )
        np.testing.assert_array_equal(np.asarray(reused), np.asarray(selected))
        # Independent visibility reference uses sequence order, not rotary IDs,
        # which restart at the packed-document boundary.
        labels = np.asarray(segments)
        visible = (np.arange(8)[None, :] <= np.arange(8)[:, None])[None]
        visible = visible & (labels[:, :, None] == labels[:, None, :])
        indices = np.asarray(selected)
        selected_visible = np.take_along_axis(visible, np.maximum(indices, 0), axis=-1)
        assert np.all((indices < 0) | selected_visible)
        np.testing.assert_array_equal(np.count_nonzero(indices >= 0, axis=-1), np.minimum(visible.sum(-1), 4))
        for result in (full, shared):
            weights = np.asarray(result.attention_weight)
            assert np.all(np.where(visible[:, None], 0, weights) == 0)
            np.testing.assert_allclose(weights.sum(-1), 1.0, atol=1e-6)
            np.testing.assert_array_equal(weights[0, :, 0, :], np.broadcast_to(np.eye(8)[0], (2, 8)))
        # Changing future features must not affect the first query in either
        # layer, even though raw top-k would otherwise fill it with future IDs.
        changed = hidden.at[:, 1:].set(hidden[:, 1:] * -71.0 + 17.0)
        changed_full, changed_selection = full_layer.forward_mla(
            changed, mask, positions, MODE_PREFILL, frequencies=frequencies, return_topk_indices=True
        )
        changed_shared = shared_layer.forward_mla(
            changed, mask, positions, MODE_PREFILL, frequencies=frequencies, previous_topk_indices=changed_selection
        )
        for actual, expected in ((changed_full, full), (changed_shared, shared)):
            np.testing.assert_array_equal(
                np.asarray(actual.attention_output[:, :1]), np.asarray(expected.attention_output[:, :1])
            )


def test_glm_moe_dsa_model_threads_causal_no_cache_selection_to_shared_layer():
    config = _config(
        "glm_moe_dsa", num_hidden_layers=2, indexer_types=["full", "shared"], index_topk=4, dsa_sparse_training=False
    )
    with config.mesh, jax.default_matmul_precision("highest"):
        model = _model("glm_moe_dsa", config)
        model.eval()
        hidden = _hidden(1, 8)
        mask = MaskInfo.from_segments(jnp.zeros((1, 8), jnp.int32))
        result = model(inputs_embeds=hidden, mask_info=mask, mode=MODE_PREFILL, output_attentions=True)
        changed = model(
            inputs_embeds=hidden.at[:, 1:].set(hidden[:, 1:] * -71.0 + 17.0),
            mask_info=mask,
            mode=MODE_PREFILL,
            output_attentions=True,
        )
        assert len(result.attentions) == 2
        future = np.triu(np.ones((8, 8), dtype=bool), k=1)
        for weights in result.attentions:
            weights = np.asarray(weights)
            assert np.all(np.where(future[None, None], weights, 0) == 0)
            np.testing.assert_allclose(weights.sum(-1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(
            np.asarray(result.last_hidden_state[:, :1]), np.asarray(changed.last_hidden_state[:, :1])
        )


def test_glm5_mtp_cache_spec_includes_pooled_indexer_storage():
    config = _config("glm5_next")
    with config.mesh:
        head = Glm5NextMTPHead(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
            max_cache_length=16,
        )
    spec = head.cache_spec()
    assert spec["auxiliary_dim"] == 2 * config.index_head_dim + 1
    assert spec["key_dim"] == config.qk_nope_head_dim + config.qk_rope_head_dim
    assert spec["max_length"] == 16
