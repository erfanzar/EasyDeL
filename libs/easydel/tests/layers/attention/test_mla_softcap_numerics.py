"""Independent attention-output regressions; accelerator-only."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import SoftCappedAttentionModule, UnifiedAttention, _unified
from easydel.operations import AttentionOutput
from jax import numpy as jnp


@pytest.fixture(autouse=True)
def highest_matmul_precision():
    """Compare fp32 references to fp32 products, not TPU's default bf16 products."""
    with jax.default_matmul_precision("highest"):
        yield


class _Projection:
    def __init__(self, weight):
        self.weight = SimpleNamespace(value=weight)

    def __call__(self, inputs):
        return inputs @ self.weight.value


class _DenseReferenceBackend:
    """A small dense backend that accepts either MLA execution layout."""

    softmax_scale = 1.0

    def forward(self, *, query_states, key_states, value_states, cache_view, **kwargs):
        if "queries_nope" in kwargs:
            q = kwargs["queries_nope"][:, :, 0]
            k = kwargs["keys_values"]
            q_pe = kwargs["queries_pe"][:, :, 0]
            k_pe = kwargs["keys_pe"]
            scores = jnp.einsum("bqd,bkd->bqk", q, k) + jnp.einsum("bqd,bkd->bqk", q_pe, k_pe)
            weights = jax.nn.softmax(scores, axis=-1)
            output = jnp.einsum("bqk,bkd->bqd", weights, k).reshape(-1, 1, k.shape[-1])
        else:
            scores = jnp.einsum("bqd,bkd->bqk", query_states[:, :, 0], key_states[:, :, 0])
            weights = jax.nn.softmax(scores, axis=-1)
            output = jnp.einsum("bqk,bkd->bqd", weights, value_states[:, :, 0])[:, :, None]
        return AttentionOutput(attention_outputs=output, cache_view=cache_view)


class _QueryTransformAttention(UnifiedAttention):
    def _postprocess_mla_queries(self, q_nope, q_pe, position_ids):
        if not self.transform_queries:
            return super()._postprocess_mla_queries(q_nope, q_pe, position_ids)
        scale = (position_ids + 1)[:, None, :, None]
        # Adding to the rotated component makes hook ordering observable (unlike scaling alone).
        return q_nope * scale, q_pe * scale + 1

    def concatenate(self, *, key, value, mask_info, cache_view, cache_metadata, **kwargs):
        return key, value, mask_info, None, cache_view, cache_metadata

    def shard_attention_prod(self, inputs, **kwargs):
        return inputs


def _attention(kv, kv_weight, transform_queries):
    attention = object.__new__(_QueryTransformAttention)
    for name, value in {
        "config": SimpleNamespace(),  # No rope_interleave field: preserve the shared fallback.
        "num_heads": 1,
        "q_head_dim": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "kv_lora_rank": 2,
        "v_head_dim": 2,
        "head_dim": 2,
        "use_mla_lora": False,
        "causal": False,
        "sliding_window": None,
        "q_proj": lambda inputs: inputs,
        "kv_a_proj_with_mqa": lambda inputs: kv,
        "kv_a_layernorm": lambda inputs: inputs,
        "kv_b_proj": _Projection(kv_weight),
        "o_proj": lambda inputs: inputs,
        "attention_performer": _DenseReferenceBackend(),
        "transform_queries": transform_queries,
    }.items():
        object.__setattr__(attention, name, value)
    return attention


@pytest.mark.parametrize("absorbed", [False, True])
@pytest.mark.parametrize("transform_queries", [False, True])
def test_mla_post_rope_query_transform_preserves_both_attention_paths(absorbed, transform_queries):
    hidden = np.array([[[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, -2.0, 1.0]]], np.float32)
    kv = np.array([[[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 3.0, 4.0]]], np.float32)
    kv_weight = np.array([[1.0, 2.0, 0.5, 1.0], [-1.0, 0.25, 1.0, -0.5]], np.float32)
    attention = _attention(jnp.asarray(kv), jnp.asarray(kv_weight), transform_queries)
    cache = object.__new__(_unified.MLARaggedPagesCacheView) if absorbed else None
    result = attention.forward_mla(
        jnp.asarray(hidden),
        mask_info=None,
        position_ids=jnp.array([[0, 1]], jnp.int32),
        mode=None,
        frequencies=jnp.array([[0.0, 1.0], [0.0, 1.0]], jnp.float32),
        cache_view=cache,
    )
    # Independent dense NumPy reference with a quarter-turn split-half RoPE.
    q_nope = hidden[..., :2]
    q_pe = np.stack((-hidden[..., 3], hidden[..., 2]), axis=-1)
    if transform_queries:
        scale = np.array([1.0, 2.0], np.float32)[None, :, None]
        q_nope = q_nope * scale
        q_pe = q_pe * scale + 1
    projected = kv[..., :2] @ kv_weight
    k_pe = np.stack((-kv[..., 3], kv[..., 2]), axis=-1)
    queries = np.concatenate((q_nope, q_pe), axis=-1)
    keys = np.concatenate((projected[..., :2], k_pe), axis=-1)
    scores = queries @ keys.swapaxes(-1, -2)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    expected = weights @ projected[..., 2:]
    np.testing.assert_allclose(result.attention_output, expected, rtol=2e-5, atol=2e-5)
    assert result.attention_output.dtype == jnp.float32
    assert result.cache_view is cache


@pytest.mark.parametrize("override", [None, 1.5])
def test_shared_soft_cap_attention_matches_independent_tanh_reference(override):
    config = EasyDeLBaseConfig(
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    module = SoftCappedAttentionModule(config, softmax_scale=0.5, logits_soft_cap=2.0, requires_cache=False)
    q = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    k = np.array([[4.0, 1.0], [-2.0, 3.0]], np.float32)
    v = np.array([[1.0, 2.0], [4.0, -1.0]], np.float32)
    shape = (jax.device_count(), 2, 1, 2)
    result = module(
        *(jnp.broadcast_to(jnp.asarray(x)[None, :, None], shape) for x in (q, k, v)),
        mode=None,
        causal=False,
        logits_soft_cap=override,
        precision=jax.lax.Precision.HIGHEST,
    )
    cap = 2.0 if override is None else override
    scores = cap * np.tanh((q @ k.T * 0.5) / cap)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    expected = np.broadcast_to((weights @ v)[None, :, None], shape)
    np.testing.assert_allclose(result.attention_outputs, expected, rtol=2e-5, atol=2e-5)
