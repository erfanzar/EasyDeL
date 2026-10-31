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

"""Self-contained Qwen4-Exp structural and decode-path tests.

Qwen4-Exp (``qwen4_exp``) is not yet in any released transformers, so unlike
the standard spmd suites these tests do not compare against a HuggingFace
reference build. Instead they pin the invariants that reference parity was
established out-of-band (dev-time driver against the transformers main
branch) and that must not silently regress:

* prefill forward produces finite, correctly-shaped logits;
* sparse-attention decode is fp32-exact against the full forward -- the
  indexer's cached top-k must select exactly the keys the prefill pass saw;
* hybrid decode (GDN layers + PLE context carry) stays within the
  chunked-vs-recurrent tolerance inherited from upstream ``qwen3_next``;
* the heterogeneous cache materializes the right per-layer views (indexer
  buffers on QSA layers, PLE conv/token context on linear layers);
* the MTP head runs off the pre-collapse stream state;
* the registry round-trips the config to the registered module.
"""

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import easydel as ed
import jax
import numpy as np
import pytest
from easydel.modules.qwen4_exp import Qwen4ExpTextConfig
from jax import numpy as jnp

SEQ = 12
BATCH = 2
VOCAB = 512


def _config(**overrides):
    """Tiny 4-layer hybrid config: layers 1/3 QSA, 0/2 GDN, PLE on layer 0."""
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[1],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=2000,
        make_ngram_vocab_size_divisible_by=16,
        split_ngram_parts=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        eos_token_id=0,
        bos_token_id=1,
        pad_token_id=2,
        norm_topk_prob=True,
        output_gate_type="sigmoid",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_interleaved": True,
            # sum == rotary_dim // 2 == 4 for this tiny head_dim/partial combo
            "mrope_section": [2, 1, 1],
        },
        # fp32 attention keeps the self-comparison tests at tight numerics
        # (the global attn_dtype default is bfloat16).
        attn_dtype=jnp.float32,
        mtp=None,
        # Single-device mesh: the comparison under test is the decode path,
        # not sharding, and -1 fills would otherwise fan ep across the fake
        # 8-device host against this model's 8 experts.
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    kwargs.update(overrides)
    return Qwen4ExpTextConfig(**kwargs)


def _model(config):
    # These are strict fp32 parity tests. TPU DEFAULT dots use bf16 MXU
    # precision even with fp32 storage and can round differently by chunk size.
    return ed.AutoEasyDeLModelForCausalLM.from_config(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )


def _ids():
    rng = np.random.default_rng(0)
    return rng.integers(3, VOCAB, size=(BATCH, SEQ)).astype("int32")


def _full_forward(model):
    out = model(input_ids=jnp.asarray(_ids()))
    return np.asarray(out.logits, np.float32)


def _decode_last(model):
    """Prefill on the first SEQ-1 tokens, decode the last one; return logits."""
    ids = _ids()
    cache = model.init_cache(batch_size=BATCH, max_length=SEQ + 8)
    pre = model(input_ids=jnp.asarray(ids[:, :-1]), past_key_values=cache)
    last = jnp.asarray(ids[:, -1:])
    pos = jnp.full((BATCH, 1), SEQ - 1, dtype=jnp.int32)
    step = model(input_ids=last, position_ids=pos, past_key_values=pre.past_key_values)
    return np.asarray(step.logits, np.float32)


class TestQwen4ExpPrefill:
    """Prefill smoke over the full hybrid stack."""

    def test_logits_shape_and_finite(self):
        model = _model(_config())
        logits = _full_forward(model)
        assert logits.shape == (BATCH, SEQ, VOCAB)
        assert np.all(np.isfinite(logits)), "prefill produced non-finite logits"


class TestQwen4ExpDecode:
    """Cached decode vs the (authoritative) full forward."""

    def test_qsa_decode_bit_exact(self):
        """All-QSA stack: decode must reproduce the full forward exactly.

        With no GDN layers there is no chunked-vs-recurrent re-association,
        so the only admissible drift is float32 re-association rounding
        between the prefill and decode kernels; anything beyond that means
        the indexer cache or the additive-bias history handling is wrong.
        """
        model = _model(_config(full_attention_interval=1, ple_layer_ids=[]))
        full = _full_forward(model)[:, -1, :]
        dec = _decode_last(model)[:, 0, :]
        delta = float(np.max(np.abs(full - dec)))
        assert delta < 1e-6, f"QSA decode drifted from full forward by {delta:.3e} (expected fp32-exact)"

    def test_hybrid_decode_within_gdn_tolerance(self):
        """Mixed stack: GDN recurrence re-association bounds the drift.

        The GDN decode path evaluates the recurrent form while prefill uses
        the chunked scan; the resulting drift is inherited verbatim from
        upstream ``qwen3_next`` (~2e-2 at these scales) and compounds over
        the two GDN layers plus the PLE context carry in this config.
        """
        model = _model(_config())
        full = _full_forward(model)[:, -1, :]
        dec = _decode_last(model)[:, 0, :]
        delta = float(np.max(np.abs(full - dec)))
        assert delta < 5e-2, f"hybrid decode drifted {delta:.3e} beyond the GDN tolerance"


class TestQwen4ExpCache:
    """Heterogeneous cache materialization."""

    def test_views_match_layer_types(self):
        model = _model(_config())
        cache = model.init_cache(batch_size=BATCH, max_length=SEQ + 8)
        assert len(cache.views) == 4
        for idx, view in enumerate(cache.views):
            # (i + 1) % full_attention_interval == 0 -> qwen_sparse_attention
            is_qsa = idx % 2 == 1
            if is_qsa:
                assert type(view).__name__ == "Qwen4ExpQSAView", f"layer {idx}: expected QSA view"
                assert view.indexer_key is not None and view.indexer_visible is not None
                assert view.mrope_positions is not None
            else:
                assert type(view).__name__ == "Qwen4ExpLinearView", f"layer {idx}: expected linear view"
            has_ple = getattr(view, "ple_conv_state", None) is not None
            # ple_layer_ids is 1-indexed: [1] attaches PLE to layer 0
            assert has_ple == (idx == 0), f"layer {idx}: PLE state presence {has_ple}"

    def test_indexer_buffers_sized_by_budget(self):
        model = _model(_config())
        cache = model.init_cache(batch_size=BATCH, max_length=SEQ + 8)
        view = cache.views[1]  # first QSA layer (layer 0 is GDN)
        compress = 2  # indexer_compress_ratio
        budget = 4  # indexer_budget
        assert view.indexer_key.shape == (
            BATCH,
            SEQ + 8,
            view.indexer_key.shape[-1],
        )
        # visible counts are capped by the compressed history length
        assert int(np.max(np.asarray(view.indexer_visible))) <= (SEQ + 8) // compress + budget


class TestQwen4ExpMTP:
    """Multi-token-prediction head off the pre-collapse streams."""

    def test_mtp_output_shape_and_finite(self):
        config = _config(mtp_num_hidden_layers=1)
        model = _model(config)
        ids = _ids()
        base_out = model.model(input_ids=jnp.asarray(ids))
        assert base_out.last_stream_state is not None
        next_ids = jnp.roll(jnp.asarray(ids), -1, axis=1)  # [B, SEQ]
        mtp = model.compute_mtp_outputs(
            last_stream_state=base_out.last_stream_state,
            next_token_ids=next_ids,
        )
        assert mtp is not None
        logits = np.asarray(model.apply_mtp_lm_head(mtp), np.float32)
        assert logits.shape == (BATCH, SEQ, VOCAB)
        assert np.all(np.isfinite(logits)), "MTP head produced non-finite logits"


class TestQwen4ExpRegistry:
    """Config/module registry roundtrip."""

    def test_factory_resolves_registered_module(self):
        config = _config()
        model = _model(config)
        assert model._model_type == "qwen4_exp_text"
        assert isinstance(model.config, Qwen4ExpTextConfig)
        assert model.config.model_type == "qwen4_exp_text"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])


class TestQwen4ExpTraining:
    def test_training_forward_exposes_mtp_loss_and_logits(self):
        from spectrax import common_types

        model = _model(
            _config(
                num_hidden_layers=2,
                layer_types=["qwen_sparse_attention", "qwen_sparse_attention"],
                ple_layer_ids=[],
                mtp_num_hidden_layers=1,
                mtp_loss_coef=0.2,
            )
        )
        ids = jnp.asarray(_ids()[:1, :8])
        labels = ids.at[:, :4].set(-100)
        out = model(input_ids=ids, labels=labels, mode=common_types.MODE_TRAIN)
        assert out.mtp_logits is not None
        assert out.mtp_logits.shape == (1, 8, VOCAB)
        assert out.mtp_loss is not None and jnp.isfinite(out.mtp_loss)
        expected = model.compute_mtp_loss(out.mtp_logits, labels) * model.config.mtp_loss_coef
        assert jnp.allclose(out.mtp_loss, expected)
        assert out.aux_loss is not None and jnp.isfinite(out.aux_loss)

        # Component-level score-proxy and MTP gradient tests cover CPU;
        # the composed parameter-gradient/optimizer step is exercised on TPU.


def test_qwen4_fp32_recurrent_state_and_qsa_disabled_cache():
    linear_cfg = _config(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        ple_layer_ids=[],
        mtp_num_hidden_layers=0,
        mamba_ssm_dtype="float32",
    )
    linear = ed.AutoEasyDeLModelForCausalLM.from_config(linear_cfg, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    assert linear.model.layers[0].linear_attn.gdr_op.metadata.runtime_dtype == jnp.float32
    cache = linear.init_cache(batch_size=1, max_length=16)
    assert cache.views[0].recurrent_state.dtype == jnp.float32

    no_qsa = _config(
        num_hidden_layers=1,
        layer_types=["qwen_sparse_attention"],
        ple_layer_ids=[],
        mtp_num_hidden_layers=0,
        indexer_n_heads=None,
        indexer_kv_heads=None,
        indexer_head_dim=None,
        indexer_budget=None,
        indexer_compress_ratio=None,
    )
    dense = _model(no_qsa)
    dense_cache = dense.init_cache(batch_size=1, max_length=16)
    assert dense_cache.views[0].indexer_key is not None
    assert dense_cache.views[0].indexer_key.shape[-1] == 0
    ids = jnp.asarray(_ids()[:1, :4])
    prefill = dense(input_ids=ids[:, :3], past_key_values=dense_cache)
    decode = dense(input_ids=ids[:, 3:], past_key_values=prefill.past_key_values)
    assert bool(jnp.isfinite(decode.logits).all())


def test_mtp_uses_dedicated_embedding_and_local_qsa_cache():
    config = _config(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        ple_layer_ids=[],
        mtp={"num_hidden_layers": 1, "layer_types": ["full_attention"]},
        mtp_use_dedicated_embeddings=True,
    )
    model = _model(config)
    assert model.mtp is not None and model.mtp.embed_tokens is not None
    cache = model.mtp.init_cache(batch_size=1, max_length=16)
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpCache

    assert isinstance(cache, Qwen4ExpCache)
    assert len(cache.views) == 1
    assert type(cache.views[0]).__name__ == "Qwen4ExpQSAView"


def test_mtp_cache_is_reusable_across_prefill_and_decode():
    from ejkernel.types import MaskInfo
    from spectrax import common_types

    config = _config(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        ple_layer_ids=[],
        mtp_num_hidden_layers=1,
    )
    model = _model(config)
    cache = model.mtp.init_cache(batch_size=1, max_length=8)
    ids = jnp.asarray(_ids()[:1, :4])
    embeddings = model.model.embed_tokens(ids.astype("i4"))
    stream = jnp.tile(embeddings, (1, 1, config.hc_count))
    prefill = model.mtp(
        prev_stream_state=stream[:, :3],
        next_token_embeds=embeddings[:, :3],
        mask_info=MaskInfo.dynamic_init(inputs_embeds=embeddings[:, :3]),
        position_ids=jnp.arange(3, dtype=jnp.int32)[None],
        frequencies=model.model.frequencies,
        past_key_values=cache,
        mode=common_types.MODE_DECODE,
    )
    decode = model.mtp(
        prev_stream_state=stream[:, 3:],
        next_token_embeds=embeddings[:, 3:],
        mask_info=MaskInfo.dynamic_init(inputs_embeds=embeddings[:, 3:]),
        position_ids=jnp.asarray([[3]], dtype=jnp.int32),
        frequencies=model.model.frequencies,
        past_key_values=prefill.past_key_values,
        mode=common_types.MODE_DECODE,
    )
    assert isinstance(decode.past_key_values, type(cache))
    assert bool(jnp.isfinite(decode.last_hidden_state).all())
    assert int(decode.past_key_values.views[0].indexes[0]) == 4


def test_qwen4_tpu_training_step_updates_parameters():
    """Composed QSA+MoE+MTP model supports a finite optimizer step."""
    import jax
    import optax
    from easydel.infra.base_state import EasyDeLState
    from spectrax import common_types

    config = _config(
        num_hidden_layers=1,
        layer_types=["qwen_sparse_attention"],
        ple_layer_ids=[],
        mtp_num_hidden_layers=1,
        mtp_loss_coef=0.2,
    )
    model = _model(config)
    ids = jnp.asarray(_ids()[:1, :6])
    state = EasyDeLState.create(model=model, tx=optax.adam(1e-3), init_opt_state=True)

    def loss_fn(graphstate):
        mdl = state.merge(graphstate)
        out = mdl(input_ids=ids, mode=common_types.MODE_TRAIN)
        return jnp.mean(out.logits.astype(jnp.float32) ** 2) + out.mtp_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.graphstate)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if hasattr(g, "size") and g.size]
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    assert any(bool(jnp.any(g != 0)) for g in leaves)
    named_grads = [
        (jax.tree_util.keystr(path), grad)
        for path, grad in jax.tree_util.tree_flatten_with_path(grads)[0]
        if hasattr(grad, "size") and grad.size
    ]
    assert any("indexer" in name and bool(jnp.any(grad != 0)) for name, grad in named_grads)
    assert any("mtp" in name and bool(jnp.any(grad != 0)) for name, grad in named_grads)
    updated = state.apply_gradients(grads=grads)
    movement = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(
            jax.tree_util.tree_leaves(state.graphstate),
            jax.tree_util.tree_leaves(updated.graphstate),
            strict=True,
        )
        if hasattr(a, "size") and a.size
    )
    assert movement > 0.0


def test_output_attentions_preserves_linear_layer_slots():
    model = _model(
        _config(
            num_hidden_layers=2,
            layer_types=["linear_attention", "qwen_sparse_attention"],
            ple_layer_ids=[],
            mtp_num_hidden_layers=0,
        )
    )
    out = model(input_ids=jnp.asarray(_ids()[:1, :4]), output_attentions=True)
    assert len(out.attentions) == 2
    assert out.attentions[0] is None
    # Optimized attention implementations may omit matrices, but layer slots
    # must remain aligned with the decoder stack.
    assert out.attentions[1] is None


def test_esurge_v3_cache_pytree_stable_across_packed_prefill():
    import jax
    from easydel.caching import RaggedPagesCacheConfig, RaggedPagesMetadata
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpPagedQSAView
    from spectrax import common_types

    model = _model(
        _config(
            attn_mechanism="ragged_page_attention_v3",
            full_attention_interval=1,
            ple_layer_ids=[],
            num_hidden_layers=1,
            hidden_size=512,
            head_dim=128,
            indexer_head_dim=64,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_interleaved": True,
                "mrope_section": [16, 8, 8],
            },
        )
    )
    ragged = RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=16,
        num_kv_heads=2,
        k_headdim=16,
        v_headdim=16,
        num_pages=8,
        max_num_pages_per_req=4,
        num_slices_per_kv_cache_update_page=4,
        max_num_tokens=4,
        max_num_reqs=2,
        page_size=4,
        _kvdtype_str="float32",
        version="v3",
    )
    cache = model.init_operations_cache(
        batch_size=2, max_length=16, page_size=4, dtype=jnp.float32, ragged_config=ragged
    )
    assert any(isinstance(view, Qwen4ExpPagedQSAView) for view in cache.views)
    metadata = RaggedPagesMetadata(
        pages_tables=jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], jnp.int32),
        context_lens=jnp.array([1, 1], jnp.int32),
        query_start_loc=jnp.array([0, 1, 2], jnp.int32),
        num_seqs=jnp.array([2], jnp.int32),
        request_distribution=jnp.array([0, 1, 1], jnp.int32),
        version="v3",
        page_size=4,
    )
    before = jax.tree_util.tree_structure(cache)
    out = model(
        input_ids=jnp.asarray(_ids()[:1, :2]),
        position_ids=jnp.zeros((1, 2), dtype=jnp.int32),
        past_key_values=cache,
        cache_metadata=metadata,
        mode=common_types.MODE_PREFILL,
    )
    assert jax.tree_util.tree_structure(out.past_key_values) == before
    for old, new in zip(cache.views, out.past_key_values.views, strict=True):
        assert type(new) is type(old)


def test_qsa_disabled_ragged_cache_uses_plain_view():
    from easydel.caching import RaggedPagesCacheConfig
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpPagedQSAView

    config = _config(
        num_hidden_layers=1,
        layer_types=["qwen_sparse_attention"],
        ple_layer_ids=[],
        mtp_num_hidden_layers=0,
        attn_mechanism="ragged_page_attention_v3",
        indexer_n_heads=None,
        indexer_kv_heads=None,
        indexer_head_dim=None,
        indexer_budget=None,
        indexer_compress_ratio=None,
    )
    model = _model(config)
    ragged = RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=16,
        num_kv_heads=2,
        k_headdim=16,
        v_headdim=16,
        num_pages=4,
        max_num_pages_per_req=4,
        num_slices_per_kv_cache_update_page=4,
        max_num_tokens=4,
        max_num_reqs=1,
        page_size=4,
        _kvdtype_str="float32",
        version="v3",
    )
    cache = model.init_operations_cache(
        batch_size=1, max_length=16, page_size=4, dtype=jnp.float32, ragged_config=ragged
    )
    assert not any(isinstance(view, Qwen4ExpPagedQSAView) for view in cache.views)


def test_ple_operations_cache_allocates_continuation_state():
    from easydel.caching import RaggedPagesCacheConfig
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpOperationsLinearView

    config = _config(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        ple_layer_ids=[1],
        mtp_num_hidden_layers=0,
        attn_mechanism="ragged_page_attention_v3",
    )
    model = _model(config)
    ragged = RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=16,
        num_kv_heads=2,
        k_headdim=16,
        v_headdim=16,
        num_pages=4,
        max_num_pages_per_req=4,
        num_slices_per_kv_cache_update_page=4,
        max_num_tokens=4,
        max_num_reqs=2,
        page_size=4,
        _kvdtype_str="float32",
        version="v3",
    )
    cache = model.init_operations_cache(
        batch_size=2, max_length=16, page_size=4, dtype=jnp.float32, ragged_config=ragged
    )
    view = cache.views[0]
    assert isinstance(view, Qwen4ExpOperationsLinearView)
    assert view.ple_token_context.shape == (2, config.ngram_size - 1)
    assert view.ple_segment_context.shape == view.ple_token_context.shape
    assert view.ple_conv_state.shape == (
        2,
        config.hidden_size * config.hc_count,
        (config.ple_conv_kernel_size - 1) * config.ngram_size,
    )


def test_ple_operations_cache_packed_prefill_updates_each_request():
    from easydel.caching import RaggedPagesCacheConfig

    config = _config(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        ple_layer_ids=[1],
        mtp_num_hidden_layers=0,
        attn_mechanism="ragged_page_attention_v3",
    )
    model = _model(config)
    ragged = RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=16,
        num_kv_heads=2,
        k_headdim=16,
        v_headdim=16,
        num_pages=4,
        max_num_pages_per_req=4,
        num_slices_per_kv_cache_update_page=4,
        max_num_tokens=4,
        max_num_reqs=2,
        page_size=4,
        _kvdtype_str="float32",
        version="v3",
    )
    cache = model.init_operations_cache(
        batch_size=2, max_length=16, page_size=4, dtype=jnp.float32, ragged_config=ragged
    )
    view = cache.views[0]
    ple = model.model.layers[0].ple
    _, token_context, conv_state, segment_context = ple(
        jnp.zeros((1, 2, config.hidden_size * config.hc_count), jnp.float32),
        jnp.array([[11, 22]], jnp.int32),
        segment_ids=jnp.array([[0, 1]], jnp.int32),
        ple_token_context=view.ple_token_context,
        ple_segment_context=view.ple_segment_context,
        ple_conv_state=view.ple_conv_state,
        packed_query_start_loc=jnp.array([0, 1, 2], jnp.int32),
    )
    assert jnp.array_equal(token_context[:, -1], jnp.array([11, 22]))
    assert jnp.array_equal(segment_context[:, -1], jnp.array([0, 1]))
    assert conv_state.shape == view.ple_conv_state.shape


@pytest.mark.parametrize("eos_token_id", [0, 7])
def test_ple_short_right_padded_prefix_preserves_missing_history(eos_token_id):
    config = _config(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        ple_layer_ids=[1],
        mtp_num_hidden_layers=0,
        eos_token_id=eos_token_id,
    )
    model = _model(config)
    ple = model.model.layers[0].ple
    width = config.hidden_size * config.hc_count
    ids = jnp.array([[11, eos_token_id, eos_token_id]], jnp.int32)
    hidden = jnp.ones((1, 3, width), jnp.float32)
    reference = ple(hidden[:, :1], ids[:, :1], segment_ids=jnp.array([[4]], jnp.int32))
    padded = ple(
        hidden,
        ids,
        segment_ids=jnp.array([[4, -1, -1]], jnp.int32),
        conv_mask=jnp.array([[True, False, False]]),
    )
    np.testing.assert_array_equal(padded[1], [[eos_token_id, 11]])
    np.testing.assert_array_equal(padded[3], [[-1, 4]])
    np.testing.assert_allclose(padded[2], reference[2], rtol=2e-5, atol=2e-5)
    for state in (reference, padded):
        continued = ple(
            hidden[:, :1],
            jnp.array([[12]], jnp.int32),
            segment_ids=jnp.array([[4]], jnp.int32),
            ple_token_context=state[1],
            ple_conv_state=state[2],
            ple_segment_context=state[3],
        )
        if state is reference:
            expected = continued
        else:
            np.testing.assert_allclose(continued[0], expected[0], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("split", [2, 4, 5])
@pytest.mark.parametrize("right_padding", [0, 2])
def test_ple_cached_segment_boundaries_match_isolated_documents(split, right_padding):
    config = _config(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        ple_layer_ids=[1],
        mtp_num_hidden_layers=0,
        eos_token_id=7,
    )
    model = _model(config)
    ple = model.model.layers[0].ple
    width = config.hidden_size * config.hc_count
    # The production initializer is zero; nonzero taps are essential to
    # expose boundary leaks in the output, not just the saved window.
    ple.conv1d.value = jnp.full_like(ple.conv1d.value, 0.125)
    ids = jnp.arange(11, 21, dtype=jnp.int32)[None]
    hidden = jnp.arange(10 * width, dtype=jnp.float32).reshape(1, 10, width) / 100
    segments = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], jnp.int32)
    isolated = [ple(hidden[:, a:b], ids[:, a:b], segment_ids=segments[:, a:b]) for a, b in ((0, 4), (4, 10))]
    expected = jnp.concatenate([result[0] for result in isolated], axis=1)
    ctx = ple.ple_embedding.context_len
    state = (
        None,
        jnp.full((1, ctx), 7, jnp.int32),
        jnp.zeros((1, width, ple.short_conv_state_len), jnp.float32),
        jnp.full((1, ctx), -1, jnp.int32),
    )
    outputs = []
    for start, end in ((0, split), (split, 10)):
        length = end - start
        state = ple(
            jnp.pad(hidden[:, start:end], ((0, 0), (0, right_padding), (0, 0))),
            jnp.pad(ids[:, start:end], ((0, 0), (0, right_padding)), constant_values=7),
            segment_ids=jnp.pad(segments[:, start:end], ((0, 0), (0, right_padding)), constant_values=-1),
            conv_mask=(jnp.arange(length + right_padding)[None] < length) if right_padding else None,
            ple_token_context=state[1],
            ple_conv_state=state[2],
            ple_segment_context=state[3],
        )
        outputs.append(state[0][:, :length])
    np.testing.assert_allclose(jnp.concatenate(outputs, axis=1), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(state[2], isolated[-1][2], rtol=2e-5, atol=2e-5)
    # An all-padding continuation must leave every piece of cached history intact.
    empty = ple(
        hidden[:, :2],
        jnp.full((1, 2), 7, jnp.int32),
        segment_ids=jnp.full((1, 2), -1, jnp.int32),
        conv_mask=jnp.zeros((1, 2), jnp.bool_),
        ple_token_context=state[1],
        ple_conv_state=state[2],
        ple_segment_context=state[3],
    )
    for actual, previous in zip(empty[1:], state[1:], strict=True):
        np.testing.assert_array_equal(actual, previous)


@pytest.mark.parametrize("eos_token_id", [0, 7])
def test_packed_ple_chunk_continuation_matches_contiguous_prefill(eos_token_id):
    config = _config(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        ple_layer_ids=[1],
        mtp_num_hidden_layers=0,
        eos_token_id=eos_token_id,
    )
    model = _model(config)
    ple = model.model.layers[0].ple
    width = config.hidden_size * config.hc_count
    ids = jnp.array([[11, 12, 13, 14]], jnp.int32)
    hidden = jnp.arange(4 * width, dtype=jnp.float32).reshape(1, 4, width) / 100
    full, full_tokens, full_conv, _full_segments = ple(hidden, ids, segment_ids=jnp.zeros_like(ids))
    ctx = ple.ple_embedding.context_len
    first, tokens, conv, segments = ple(
        hidden[:, :1],
        ids[:, :1],
        segment_ids=jnp.zeros((1, 1), jnp.int32),
        ple_token_context=jnp.zeros((1, ctx), jnp.int32),
        ple_segment_context=jnp.full((1, ctx), -1, jnp.int32),
        packed_query_start_loc=jnp.array([0, 1], jnp.int32),
    )
    second, tokens, conv, segments = ple(
        hidden[:, 1:],
        ids[:, 1:],
        segment_ids=jnp.zeros((1, 3), jnp.int32),
        ple_token_context=tokens,
        ple_segment_context=segments,
        ple_conv_state=conv,
        packed_query_start_loc=jnp.array([0, 3], jnp.int32),
    )
    np.testing.assert_allclose(np.asarray(first), np.asarray(full[:, :1]), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(second), np.asarray(full[:, 1:]), rtol=2e-5, atol=2e-5)
    assert jnp.array_equal(tokens, full_tokens)
    np.testing.assert_allclose(np.asarray(conv), np.asarray(full_conv), rtol=2e-5, atol=2e-5)
