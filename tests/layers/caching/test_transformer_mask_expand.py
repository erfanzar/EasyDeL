import jax.numpy as jnp
from ejkernel.types import MaskInfo

from easydel.layers.caching.transformer.cache import _expand_mask_kv_dim


def test_expand_mask_kv_dim_preserves_padding_mask():
    # Left-padded example: first 3 tokens are padding (masked out as KV).
    attention_mask = jnp.array([[0, 0, 0, 1, 1, 1]], dtype=jnp.int32)
    mask_info = MaskInfo.from_attention_mask(attention_mask=attention_mask)

    expanded = _expand_mask_kv_dim(
        mask_info=mask_info,
        target_kv_len=10,
        cache_position=jnp.array([0], dtype=jnp.int32),
        query_len=attention_mask.shape[1],
    )

    attn = expanded.get_or_compute_attention_mask(dtype=jnp.bool_)
    assert attn.shape[-1] == 10

    # Original padding KV positions must stay masked for all queries.
    assert not bool(jnp.any(attn[..., :3]))
    # Newly added KV positions are future cache slots; they must not be treated as padding.
    # They will be kept inactive by kv-lengths and causal masking during attention.
    assert bool(jnp.any(attn[..., 3:, 6:]))
    # Sanity: keep some unmasked area.
    assert bool(jnp.any(attn[..., 3:6]))
