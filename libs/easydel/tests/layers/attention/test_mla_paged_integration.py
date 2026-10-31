"""Abstract absorbed-MLA integration contracts; no CPU array execution."""

from types import SimpleNamespace

import jax
import pytest
import spectrax as spx
from easydel.caching import RaggedPagesMetadata
from easydel.caching.mla_ragged_page import MLARaggedPagesCacheConfig, MLARaggedPagesCacheView
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from jax import numpy as jnp
from spectrax import common_types


class _UnshardedMLA(UnifiedAttention):
    """Keep the production forward/concatenate/backend, with fixed projections."""

    def shard_attention_prod(self, inputs, **kwargs):
        return inputs


def _attention(config, weight):
    attention = _UnshardedMLA.__new__(_UnshardedMLA)
    performer = FlexibleAttentionModule(
        config, softmax_scale=0.5, dropout_prob=0.25, rngs=spx.Rngs(dropout=jax.random.key(123))
    )
    # Force active dropout in decode too, exercising the registered stochastic
    # adapter rather than the accelerator-only deterministic MLA kernel.
    performer.deterministic = False
    values = {
        "config": config,
        "dtype": jnp.float32,
        "num_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "q_head_dim": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "kv_lora_rank": 4,
        "v_head_dim": 2,
        "use_mla_lora": False,
        "causal": True,
        "sliding_window": None,
        "attention_type": "mla",
        "precision": jax.lax.Precision.HIGHEST,
        "q_proj": lambda inputs: inputs,
        "kv_a_proj_with_mqa": lambda inputs: inputs[..., :6],
        "kv_a_layernorm": lambda inputs: inputs,
        "kv_b_proj": SimpleNamespace(weight=SimpleNamespace(value=weight)),
        "o_proj": lambda inputs: inputs,
        "attention_performer": performer,
    }
    for name, value in values.items():
        object.__setattr__(attention, name, value)
    return attention


@pytest.mark.cpu_ok
@pytest.mark.parametrize("batch,seq_len", [(1, 4), (2, 2)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("mode", [common_types.MODE_TRAIN, common_types.MODE_DECODE])
def test_unified_absorbed_mla_dropout_traces_with_history_and_returns_updated_pages(batch, seq_len, dtype, mode):
    """Trace actual UnifiedAttention -> FlexibleAttentionModule -> registered op.

    ShapeDtypeStruct inputs and make_jaxpr perform abstract evaluation only.
    Latent=4 and RoPE=2 deliberately require independent v1 page alignment;
    two requests both retain cached history before appending current tokens.
    """
    config = EasyDeLBaseConfig(
        attn_mechanism="multi_latent_ragged_page_attention_v1",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    cache_config = MLARaggedPagesCacheConfig(
        1, 4, 1, 4, 2, page_size=2, num_pages=4, version="v1", _kvdtype_str="fp32" if dtype == jnp.float32 else "bf16"
    )
    page_shape = cache_config.get_shape_and_axes()[0]

    def forward(hidden, positions, weight, pages):
        attention = _attention(config, weight)
        view = MLARaggedPagesCacheView(cache_config, 0, pages)
        metadata = RaggedPagesMetadata(
            jnp.array([[2, 0], [3, 1]], dtype=jnp.int32),
            jnp.array([3, 4], dtype=jnp.int32),
            jnp.array([0, 2, 4], dtype=jnp.int32),
            jnp.array([3, 4], dtype=jnp.int32),
            page_size=2,
        )
        result = attention(
            hidden,
            mask_info=None,
            position_ids=positions,
            mode=mode,
            cache_view=view,
            cache_metadata=metadata,
        )
        assert isinstance(result.cache_view, MLARaggedPagesCacheView)
        return result.attention_output, result.cache_view.kv_pages

    program = jax.make_jaxpr(forward)(
        jax.ShapeDtypeStruct((batch, seq_len, 8), jnp.float32),
        jax.ShapeDtypeStruct((batch, seq_len), jnp.int32),
        jax.ShapeDtypeStruct((4, 8), jnp.float32),
        jax.ShapeDtypeStruct(page_shape, dtype),
    )
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [
        ((batch, seq_len, 4), jnp.float32),
        (page_shape, dtype),
    ]
    # Public cache output must include the scheduled writes, not the input pages.
    assert program.jaxpr.outvars[1] != program.jaxpr.invars[-1]
