"""Public-forward and cache-lifecycle regressions (accelerator only)."""

from dataclasses import fields

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import (
    HybridCacheConfig,
    RaggedPagesCacheConfig,
    RaggedPagesMetadata,
    RecurrentCacheConfig,
    TransformerCacheConfig,
)
from easydel.modules.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextModel
from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModel, Qwen3VLMoeTextModel
from easydel.modules.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpAttention,
    Qwen4ExpCache,
    Qwen4ExpLinearView,
    Qwen4ExpOperationsLinearView,
    Qwen4ExpPagedQSAView,
    Qwen4ExpQSAView,
    Qwen4ExpSparseMoeBlock,
)
from easydel.modules.qwen4_exp.qwen4_exp_configuration import Qwen4ExpTextConfig
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types

from .test_numerical_regressions import _moe_config
from .test_qwen3_vl_multimodal_numerics import _model


def _qwen4_config(**kwargs):
    return Qwen4ExpTextConfig(
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=0,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
        layer_types=["qwen_sparse_attention"],
        indexer_n_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
        hc_count=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=3,
        moe_force_xla_gmm=True,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )


def _qsa_view(config):
    metadata = TransformerCacheConfig.create(
        batch_size=1,
        sequence_length=4,
        num_hidden_layers=1,
        num_heads=1,
        head_dim=8,
    )
    return Qwen4ExpQSAView.init(
        metadata,
        indexer_head_dim=8,
        indexer_max_blocks=2,
        mesh=config.mesh,
        runtime_sharding_resolver=config.runtime_sharding_resolver,
        dtype=jnp.float32,
    )


@pytest.mark.parametrize("restriction", ["pairwise", "cache_start", "per_head"])
def test_qsa_gathered_decode_preserves_effective_request_mask(restriction):
    config = _qwen4_config(num_experts_per_tok=1)
    layer = Qwen4ExpAttention(
        config,
        layer_idx=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    fresh = _qsa_view(config)
    random = np.random.default_rng(3)
    key = jnp.asarray(random.normal(size=(1, 4, 1, 8)), jnp.float32)
    value = jnp.asarray(random.normal(size=(1, 4, 1, 8)), jnp.float32)
    view = fresh.replace(
        key=key,
        value=value,
        indexes=jnp.array([3], jnp.int32),
        starts=jnp.array([1 if restriction == "cache_start" else 0], jnp.int32),
        indexer_visible=jnp.array([[True, True, True, False]]),
        indexer_key=jnp.asarray(random.normal(size=(1, 4, 8)), jnp.float32),
        mrope_positions=jnp.broadcast_to(jnp.arange(4)[None, None], (1, 3, 4)),
    )
    mask = np.ones((1, 2 if restriction == "per_head" else 1, 1, 4), dtype=bool)
    if restriction != "cache_start":
        mask[0, 0, 0, 0] = False
    if restriction == "per_head":
        mask[0, 1, 0, 1] = False
    hidden = jnp.asarray(random.normal(size=(1, 1, 16)), jnp.float32)
    positions = jnp.full((3, 1, 1), 3, jnp.int32)

    def run(cache, full):
        return layer(
            hidden,
            mask_info=MaskInfo.from_attention_mask(jnp.asarray(mask)),
            position_ids=positions,
            mode=common_types.MODE_DECODE,
            cache_view=cache,
            output_attentions=full,
        ).attention_output

    # output_attentions selects the full-width mask-preserving performer.
    reference = run(view, True)
    gathered = run(view, False)
    np.testing.assert_allclose(np.asarray(gathered), np.asarray(reference), rtol=3e-5, atol=3e-6)
    # Poisoning a forbidden KV slot must not affect the public forward result.
    if restriction != "per_head":
        poisoned = view.replace(value=view.value.at[:, 0].set(10000.0))
        np.testing.assert_allclose(np.asarray(run(poisoned, False)), np.asarray(gathered), rtol=3e-5, atol=3e-6)


def test_paged_qsa_prefill_preserves_pairwise_exclusions():
    config = _qwen4_config(num_experts_per_tok=1)
    dense = Qwen4ExpAttention(
        config,
        layer_idx=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    paged_config = _qwen4_config(num_experts_per_tok=1)
    paged_config.attn_mechanism = "ragged_page_attention_v3"
    paged = Qwen4ExpAttention(
        paged_config,
        layer_idx=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    metadata = RaggedPagesMetadata(
        pages_tables=jnp.array([[0, 1]], jnp.int32),
        context_lens=jnp.array([2], jnp.int32),
        query_start_loc=jnp.array([0, 2], jnp.int32),
        num_seqs=jnp.array([1], jnp.int32),
        request_distribution=jnp.array([0, 1, 1], jnp.int32),
        version="v3",
        page_size=4,
    )
    cache_config = RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=8,
        num_kv_heads=1,
        k_headdim=8,
        v_headdim=8,
        num_pages=2,
        max_num_pages_per_req=2,
        num_slices_per_kv_cache_update_page=2,
        max_num_tokens=2,
        max_num_reqs=1,
        page_size=4,
        _kvdtype_str="float32",
        version="v3",
    )
    object.__setattr__(cache_config, "qwen4_indexer_head_dim", 8)
    object.__setattr__(cache_config, "qwen4_indexer_compress_ratio", 2)
    view = Qwen4ExpPagedQSAView.init(
        cache_config,
        mesh=config.mesh,
        runtime_sharding_resolver=config.runtime_sharding_resolver,
    )
    hidden = jnp.asarray(np.random.default_rng(5).normal(size=(1, 2, 16)), jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(2)[None, None], (3, 1, 2))
    mask = jnp.array([[[[True, False], [False, True]]]])

    def run(layer, hidden, cache=None):
        return layer(
            hidden,
            mask_info=MaskInfo.from_attention_mask(mask),
            position_ids=positions,
            mode=common_types.MODE_PREFILL,
            cache_view=cache,
            cache_metadata=None if cache is None else metadata,
        ).attention_output

    reference = run(dense, hidden)
    actual = run(paged, hidden, view)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=3e-5, atol=3e-6)
    poisoned = run(paged, hidden.at[:, 0].set(100.0), view)
    np.testing.assert_allclose(np.asarray(poisoned[:, 1]), np.asarray(actual[:, 1]), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("view_kind", ["qsa", "linear", "operations_linear"])
def test_reset_clears_all_family_cache_history_and_preserves_type(view_kind):
    config = _qwen4_config(num_experts_per_tok=1)
    if view_kind == "qsa":
        fresh = _qsa_view(config)
    elif view_kind == "linear":
        metadata = HybridCacheConfig.create(
            num_hidden_layers=1,
            partition_axis=config.partition_axis,
            batch_size=1,
            sequence_length=4,
            num_key_value_heads=1,
            head_dim=8,
            d_inner=8,
            d_conv=3,
            d_state=8,
            layer_types=["linear_attention"],
        )
        fresh = Qwen4ExpLinearView.init(
            metadata,
            with_ple=True,
            ple_conv_dim=8,
            ple_conv_state_len=4,
            ple_context_len=1,
            dtype=jnp.float32,
        )
    else:
        metadata = RecurrentCacheConfig.create(
            num_hidden_layers=1,
            partition_axis=config.partition_axis,
            batch_size=1,
            conv_dim=8,
            conv_kernel_size=3,
            recurrent_state_shape=(1, 8, 8),
        )
        for name, value in {
            "qwen4_with_ple": True,
            "qwen4_ple_conv_dim": 8,
            "qwen4_ple_conv_state_len": 4,
            "qwen4_ple_context_len": 1,
        }.items():
            object.__setattr__(metadata, name, value)
        fresh = Qwen4ExpOperationsLinearView.init(metadata, dtype=jnp.float32)
    # Dirty the state buffers only. tree.map also reconstructs metadata (and
    # does not retain attributes outside its declared dataclass fields), so it
    # cannot establish an identity-preservation contract for reset itself.
    dirty = fresh.replace(
        **{
            field.name: jnp.full_like(getattr(fresh, field.name), 7)
            for field in fields(fresh)
            if isinstance(getattr(fresh, field.name), jax.Array)
        }
    )
    reset = Qwen4ExpCache(views=[dirty]).reset().views[0]
    assert type(reset) is type(fresh)
    assert reset.metadata is dirty.metadata is fresh.metadata
    for actual, expected in zip(jax.tree.leaves(reset), jax.tree.leaves(fresh), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    # Reset remains functional: the old cache must still carry its history.
    assert np.any(np.asarray(jax.tree.leaves(dirty)[0]) != np.asarray(jax.tree.leaves(reset)[0]))
    if view_kind == "qsa":
        layer = Qwen4ExpAttention(
            config,
            layer_idx=0,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        hidden = jnp.ones((1, 1, 16), jnp.float32)

        def decode(cache):
            return layer(
                hidden,
                mask_info=MaskInfo.from_attention_mask(jnp.ones((1, 1), jnp.bool_)),
                position_ids=jnp.zeros((3, 1, 1), jnp.int32),
                mode=common_types.MODE_DECODE,
                cache_view=cache,
            ).attention_output

        np.testing.assert_allclose(np.asarray(decode(reset)), np.asarray(decode(fresh)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("normalize", [False, True])
def test_qwen4_routed_and_shared_experts_match_numpy_reference(top_k, normalize):
    config = _qwen4_config(num_experts_per_tok=top_k, norm_topk_prob=normalize)
    layer = Qwen4ExpSparseMoeBlock(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    hidden = np.random.default_rng(5).normal(size=(1, 3, 16)).astype(np.float32)
    with jax.default_matmul_precision("highest"):
        actual, router = layer(jnp.asarray(hidden))
    flat = hidden.reshape(-1, 16)
    logits = flat @ np.asarray(layer.gate.weight.value)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    indices = np.argsort(probs, axis=-1)[:, -top_k:]
    selected = np.take_along_axis(probs, indices, axis=-1)
    if normalize:
        selected /= selected.sum(axis=-1, keepdims=True)
    routed = np.zeros_like(flat)
    kernels = np.asarray(layer.experts.gate_up_proj.weight.value)
    downs = np.asarray(layer.experts.down_proj.weight.value)
    for token in range(flat.shape[0]):
        for slot, expert in enumerate(indices[token]):
            gate, up = np.split(flat[token] @ kernels[expert], 2)
            routed[token] += selected[token, slot] * ((gate / (1 + np.exp(-gate))) * up) @ downs[expert]
    shared_gate = flat @ np.asarray(layer.shared_expert.gate_proj.weight.value)
    shared_up = flat @ np.asarray(layer.shared_expert.up_proj.weight.value)
    shared = ((shared_gate / (1 + np.exp(-shared_gate))) * shared_up) @ np.asarray(
        layer.shared_expert.down_proj.weight.value
    )
    blend = flat @ np.asarray(layer.shared_expert_gate.weight.value)
    reference = routed + shared / (1 + np.exp(-blend))
    np.testing.assert_allclose(np.asarray(router).reshape(logits.shape), logits, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual), reference.reshape(hidden.shape), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("cls,text_cls", [(Qwen3VLModel, Qwen3VLTextModel), (Qwen3VLMoeModel, Qwen3VLMoeTextModel)])
def test_embeddings_only_forward_matches_real_text_decoder_automatic_positions(cls, text_cls):
    # A real decoder, not a metadata-consuming stub, verifies default mRoPE.
    model = _model(cls)
    config = _moe_config(True)
    config.attn_mechanism = "vanilla"
    config.attn_dtype = jnp.float32
    config.attn_softmax_dtype = jnp.float32
    config.head_dim = 8
    config.max_position_embeddings = 16
    config.rope_scaling = {"rope_type": "default", "mrope_section": [2, 1, 1]}
    object.__setattr__(model, "get_rope_index", cls.get_rope_index.__get__(model, cls))
    decoder = text_cls(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    object.__setattr__(model, "language_model", decoder)
    embeddings = jnp.asarray(np.random.default_rng(0).normal(size=(1, 3, 16)), jnp.float32)
    mask = jnp.array([[False, True, True]])
    reference = decoder(inputs_embeds=embeddings, attention_mask=mask).last_hidden_state
    actual = model(inputs_embeds=embeddings, attention_mask=mask).last_hidden_state
    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=2e-5, atol=2e-6)
