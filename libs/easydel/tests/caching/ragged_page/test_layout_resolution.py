import jax.numpy as jnp
from easydel.caching.ragged_page.cache import RaggedPagesCacheConfig, RaggedPagesCacheView, _resolve_v3_kv_head_layout


def test_fp8_replication_restores_logical_kv_head_count_when_tp_is_wider():
    dtype, shards, heads = _resolve_v3_kv_head_layout(
        jnp.float8_e4m3fn,
        logical_num_kv_heads=2,
        k_headdim=256,
        physical_kv_head_shards=4,
    )
    assert dtype == jnp.dtype(jnp.float8_e4m3fn)
    assert shards == 1
    assert heads == 2


def test_v3_head64_accessors_split_key_value_on_last_axis():
    config = RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=16,
        num_kv_heads=1,
        k_headdim=64,
        v_headdim=64,
        page_size=4,
        num_pages=2,
        version="v3",
    )
    key = jnp.arange(2 * 4 * 64, dtype=jnp.float32).reshape(2, 4, 1, 64)
    value = key + 10000
    raw = jnp.concatenate((key, value), axis=-1).reshape(2, 4, 1, 1, 128)
    view = RaggedPagesCacheView(metadata=config, layer_index=0, kv_pages=raw)
    assert jnp.array_equal(view.key_pages, key)
    assert jnp.array_equal(view.value_pages, value)
