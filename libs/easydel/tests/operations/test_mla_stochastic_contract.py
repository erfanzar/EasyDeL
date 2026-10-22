"""Abstract public MLA stochastic-adapter storage and output contracts."""

import jax
import pytest
import spectrax as spx
from easydel.caching import RaggedPagesMetadata
from easydel.caching.mla_ragged_page import MLARaggedPagesCacheConfig, MLARaggedPagesCacheView
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import FlexibleAttentionModule
from easydel.modules.deepseek_v2.deepseek_configuration import DeepseekV2Config
from easydel.modules.deepseek_v2.modeling_deepseek import DeepseekV2Attention
from easydel.operations import OperationMetadata, OperationRegistry
from jax import numpy as jnp
from spectrax import common_types

pytestmark = pytest.mark.cpu_ok


def _config():
    return EasyDeLBaseConfig(
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="multi_latent_ragged_page_attention_v1",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
    )


def _cache_config(dtype=jnp.float32, rope_dim=2):
    return MLARaggedPagesCacheConfig(
        1,
        4,
        1,
        3,
        rope_dim,
        page_size=2,
        num_pages=4,
        version="v1",
        _kvdtype_str="fp32" if dtype == jnp.float32 else "bf16",
    )


@pytest.mark.parametrize("through_layer", [False, True])
@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("rope_dim", [0, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_mla_dropout_keeps_latent_output_and_aligned_cache(through_layer, update, rope_dim, dtype):
    config = _config()
    operation = (
        FlexibleAttentionModule(config, softmax_scale=0.5)
        if through_layer
        else OperationRegistry.create(
            "paged_attention_reference", OperationMetadata(base_config=config, runtime_dtype=jnp.float32)
        )
    )
    cache_config = _cache_config(dtype, rope_dim)
    page_shape = cache_config.get_shape_and_axes()[0]
    shapes = [(1, 4, 2, 3), (1, 4, 2, rope_dim), (1, 4, 3), (1, 4, rope_dim), page_shape]
    descriptors = [jax.ShapeDtypeStruct(shape, dtype if i == 4 else jnp.float32) for i, shape in enumerate(shapes)]

    def forward(qn, qp, kv, kp, pages):
        view = MLARaggedPagesCacheView(cache_config, 0, pages)
        runtime = RaggedPagesMetadata(
            jnp.array([[2, 0], [3, 1]], dtype=jnp.int32),
            jnp.array([3, 2], dtype=jnp.int32),
            jnp.array([0, 2, 4], dtype=jnp.int32),
            jnp.array([3, 2], dtype=jnp.int32),
            page_size=2,
        )
        kwargs = dict(queries_nope=qn, queries_pe=qp)
        if update:
            kwargs.update(keys_values=kv, keys_pe=kp)
        combined = jnp.concatenate((qn, qp), axis=-1)
        result = (
            operation(
                query_states=combined,
                key_states=None,
                value_states=None,
                mode=None,
                cache_view=view,
                **dict(
                    cache_metadata=runtime,
                    deterministic=False,
                    dropout_prob=0.25,
                    dropout_rng=jax.random.key(123),
                    compute_dtype=jnp.float32,
                    precision=jax.lax.Precision.HIGHEST,
                    **kwargs,
                ),
            )
            if through_layer
            else operation(
                combined,
                cache_view=view,
                cache_metadata=runtime,
                deterministic=False,
                dropout_prob=0.25,
                dropout_rng=jax.random.key(123),
                compute_dtype=jnp.float32,
                precision=jax.lax.Precision.HIGHEST,
                **kwargs,
            )
        )
        assert isinstance(result.cache_view, MLARaggedPagesCacheView)
        return result.attention_outputs, result.cache_view.kv_pages

    program = jax.make_jaxpr(forward)(*descriptors)
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [((1, 4, 2, 3), jnp.float32), (page_shape, dtype)]
    if not update:
        assert program.jaxpr.outvars[1] == program.jaxpr.invars[-1]


@pytest.mark.parametrize("q_lora_rank", [None, 4])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_full_forward_mla_stochastic_absorbed_projection(q_lora_rank, dtype):
    """Trace real MLA projections, stochastic routing and the caller's absorbed V contraction."""
    config = DeepseekV2Config(
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=3,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        max_position_embeddings=8,
        attention_dropout=0.25,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="multi_latent_ragged_page_attention_v1",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
    )
    cache_config = _cache_config(dtype)
    page_shape = cache_config.get_shape_and_axes()[0]

    def forward(hidden_states, pages):
        # Construct inside the trace so parameter/RNG initialization does not
        # execute any numerical work in this abstract-only CPU-safe contract.
        layer = DeepseekV2Attention(
            config,
            dtype=dtype,
            param_dtype=dtype,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(74),
            layer_idx=0,
        )
        runtime = RaggedPagesMetadata(
            jnp.array([[0, 1]], dtype=jnp.int32),
            jnp.array([4], dtype=jnp.int32),
            jnp.array([0, 4], dtype=jnp.int32),
            jnp.array([4], dtype=jnp.int32),
            page_size=2,
        )
        result = layer.forward_mla(
            hidden_states,
            mask_info=None,
            position_ids=jnp.arange(4, dtype=jnp.int32)[None, :],
            mode=common_types.MODE_TRAIN,
            cache_view=MLARaggedPagesCacheView(cache_config, 0, pages),
            cache_metadata=runtime,
        )
        return result.attention_output, result.cache_view.kv_pages

    program = jax.make_jaxpr(forward)(jax.ShapeDtypeStruct((1, 4, 8), dtype), jax.ShapeDtypeStruct(page_shape, dtype))
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [((1, 4, 8), dtype), (page_shape, dtype)]
    assert program.jaxpr.outvars[1] != program.jaxpr.invars[1]
