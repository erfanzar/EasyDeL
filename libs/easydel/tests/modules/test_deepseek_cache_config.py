"""DeepSeek cache configuration contracts, without model weights or cache arrays."""

from types import SimpleNamespace

import pytest
from easydel.caching import MLARaggedPagesCacheConfig, RaggedPagesCacheConfig
from easydel.layers.attention import AttentionMechanisms
from easydel.modules.deepseek_v2.deepseek_configuration import DeepseekV2Config
from easydel.modules.deepseek_v2.modeling_deepseek import DeepseekV2ForCausalLM
from easydel.modules.deepseek_v3.deepseek_configuration import DeepseekV3Config
from easydel.modules.deepseek_v3.modeling_deepseek import DeepseekV3ForCausalLM
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


@pytest.fixture(params=[(DeepseekV2Config, DeepseekV2ForCausalLM), (DeepseekV3Config, DeepseekV3ForCausalLM)])
def cache_model(request, monkeypatch):
    """Use real family APIs/configs while replacing hardware and operation discovery."""
    config_cls, model_cls = request.param
    monkeypatch.setattr(config_cls, "mesh", property(lambda self: SimpleNamespace(shape={})))
    # No attention modules are constructed: empty discovery uses the config's layer count.
    monkeypatch.setattr(model_cls, "get_operations_cache_view", lambda self: {})
    monkeypatch.setattr(model_cls, "get_operations_cache_view_by_slot", lambda self: {})
    for cache_cls in (RaggedPagesCacheConfig, MLARaggedPagesCacheConfig):
        monkeypatch.setattr(cache_cls, "_compute_free_hbm", staticmethod(lambda **_: 1 << 28))

    def create(**kwargs):
        options = dict(
            hidden_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            kv_lora_rank=32,
            kvdtype=jnp.bfloat16,
            pad_token_id=7,
        )
        options.update(kwargs)
        model = object.__new__(model_cls)
        object.__setattr__(model, "config", config_cls(**options))
        return model

    return create


@pytest.mark.parametrize("key_dim,value_dim", [(192, 128), (64, 64)])
@pytest.mark.parametrize("kwargs", [{}, {"pad_token_id": 19, "layer_idx": 9, "legacy_option": True}])
def test_transformer_geometry_and_legacy_kwargs(cache_model, key_dim, value_dim, kwargs):
    model = cache_model(qk_nope_head_dim=key_dim - 32, qk_rope_head_dim=32, v_head_dim=value_dim)
    config = model.create_transformer_cache_config(3, 256, **kwargs)
    assert (config.key_heads, config.value_heads, config.num_heads) == (8, 8, 8)
    assert (config.key_dim, config.value_dim) == (key_dim, value_dim)
    assert (config.batch_size, config.sequence_length, config.num_hidden_layers) == (3, 256, 2)
    # The overrides must honor kwargs, including padding.
    assert config.pad_token_id == -100


@pytest.mark.parametrize(
    "mechanism,version",
    [(AttentionMechanisms.RAGGED_PAGE_ATTENTION_V2, "v2"), ("ragged_page_attention_v3", "v3")],
)
@pytest.mark.parametrize("key_dim,value_dim", [(192, 128), (64, 64)])
@pytest.mark.parametrize("dtype", [None, jnp.float32])
def test_standard_ragged_geometry_dtype_and_token_cap(cache_model, mechanism, version, key_dim, value_dim, dtype):
    model = cache_model(
        attn_mechanism=mechanism,
        qk_nope_head_dim=key_dim - 32,
        qk_rope_head_dim=32,
        v_head_dim=value_dim,
    )
    config = model.create_ragged_page_cache_config(
        256, page_size=16, hbm_utilization=0.75, dtype=dtype, max_cache_tokens=1024
    )
    assert type(config) is RaggedPagesCacheConfig
    assert (config.num_kv_heads, config.k_headdim, config.v_headdim) == (8, key_dim, value_dim)
    assert (config.max_model_length, config.num_hidden_layers, config.page_size) == (256, 2, 16)
    assert config.version == version
    assert config.hbm_utilization == 0.75
    assert config.kvdtype == (jnp.bfloat16 if dtype is None else dtype)
    assert config.num_pages * config.page_size == 1024


@pytest.mark.parametrize("mechanism", ["multi_latent_ragged_page_attention_v1", "multi_latent_ragged_page_attention_v2"])
@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("dtype", [None, jnp.float32])
def test_compressed_mla_layout_dtype_token_cap_and_extensions(cache_model, mechanism, extended, dtype):
    options = dict(mla_cache_rope_width=128, index_kpool=4, index_head_dim=32) if extended else {}
    model = cache_model(attn_mechanism=mechanism, **options)
    config = model.create_ragged_page_cache_config(
        256, page_size=16, hbm_utilization=0.75, dtype=dtype, max_cache_tokens=1024
    )
    assert isinstance(config, MLARaggedPagesCacheConfig)
    assert (config.num_kv_heads, config.kv_lora_rank, config.qk_rope_head_dim) == (8, 32, 128 if extended else 64)
    assert (config.max_model_length, config.num_hidden_layers, config.page_size) == (256, 2, 16)
    assert config.hbm_utilization == 0.75
    assert config.kvdtype == (jnp.bfloat16 if dtype is None else dtype)
    assert config.num_pages * config.page_size == 1024
    if extended:
        assert (config.indexer_key_dim, config.indexer_gate_dim, config.indexer_kpool) == (32, 32, 4)
        assert config.indexer_num_layers == 2
    else:
        assert (config.indexer_key_dim, config.indexer_gate_dim, config.indexer_kpool) == (0, 0, 1)


def test_compressed_mla_builder_accepts_config_only_caller(cache_model, monkeypatch):
    """Unbound public builders need no instance inheritance or model weights."""
    from types import MethodType

    from easydel.infra.mixins.generation import EasyGenerationMixin

    model = cache_model(attn_mechanism="multi_latent_ragged_page_attention_v2")
    dummy = SimpleNamespace(config=model.config, mesh=model.config.mesh)
    dummy._create_mla_ragged_page_cache_config = MethodType(
        EasyGenerationMixin._create_mla_ragged_page_cache_config, dummy
    )
    monkeypatch.setattr(MLARaggedPagesCacheConfig, "create", staticmethod(lambda **kwargs: SimpleNamespace(**kwargs)))
    result = type(model).create_ragged_page_cache_config(dummy, 256, page_size=16)
    assert (result.kv_lora_rank, result.qk_rope_head_dim, result.num_kv_heads) == (32, 64, 8)
    assert (result.max_model_length, result.num_hidden_layers, result.page_size) == (256, 2, 16)


@pytest.mark.parametrize(
    "mechanism,version",
    [
        (AttentionMechanisms.RAGGED_PAGE_ATTENTION_V2, "v2"),
        ("ragged_page_attention_v2", "v2"),
        ("ragged_page_attention_v3", "v3"),
        ("flash_attn2", "v3"),
    ],
)
@pytest.mark.parametrize("dtype", [None, jnp.float32])
def test_standard_ragged_builder_accepts_config_only_caller(cache_model, monkeypatch, mechanism, version, dtype):
    """Config/mesh-only callers retain expanded K/V geometry on every standard route."""
    from types import MethodType

    from easydel.infra.mixins.generation import EasyGenerationMixin

    model = cache_model(attn_mechanism=mechanism, qk_nope_head_dim=48, qk_rope_head_dim=32, v_head_dim=64)
    dummy = SimpleNamespace(config=model.config, mesh=model.config.mesh)
    dummy._create_mla_ragged_page_cache_config = MethodType(
        EasyGenerationMixin._create_mla_ragged_page_cache_config, dummy
    )
    resolver = model.config.runtime_sharding_resolver
    monkeypatch.setattr(type(model.config), "runtime_sharding_resolver", property(lambda self: resolver))
    monkeypatch.setattr(RaggedPagesCacheConfig, "create", staticmethod(lambda **kwargs: SimpleNamespace(**kwargs)))
    result = type(model).create_ragged_page_cache_config(
        dummy, 256, page_size=16, hbm_utilization=0.75, dtype=dtype, max_cache_tokens=1024
    )
    assert (result.num_kv_heads, result.kv_head_dim_size, result.k_headdim, result.v_headdim) == (8, 80, 80, 64)
    assert (result.max_model_length, result.num_hidden_layers, result.page_size) == (256, 2, 16)
    assert result.mesh is dummy.mesh
    assert result.runtime_sharding_resolver is resolver
    assert result.kvdtype == (jnp.bfloat16 if dtype is None else dtype)
    assert (result.version, result.hbm_utilization, result.max_cache_tokens) == (version, 0.75, 1024)
