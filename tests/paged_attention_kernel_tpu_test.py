import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest  # type:ignore
from jax._src import test_util as jtu

from easydel.kernels.tpu_ops import pallas_paged_attention

MAX_SEQ_LEN = 2048
BLOCK_SIZE = 512
TEST_SEQ_LENS = np.asarray(
    [
        0,
        3,
        15,
        16,
        513,
        1023,
        2047,
        MAX_SEQ_LEN,
    ]
)
# Tolerances based on whether grouped query attention is used
GQA_ATOL, GQA_RTOL = 1e-2, 0
MHA_ATOL, MHA_RTOL = 1e-2, 0


# --- Helper Functions (adapted from original test) ---


def _generate_qkv(
    seq_lens: np.ndarray,
    page_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    num_heads: int,
    head_dim: int,
    prng_key: jax.random.PRNGKey,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Generates random Q, paged K/V, and page indices for testing."""
    if max_seq_len % page_size != 0:
        raise ValueError("max_seq_len must be divisible by page_size")

    pages_per_sequence = max_seq_len // page_size
    batch_size = len(seq_lens)
    total_pages = batch_size * pages_per_sequence

    k1, k2, k3, k4 = jax.random.split(prng_key, 4)

    # Generate K/V pages (layout: [num_kv_heads, total_pages, page_size, head_dim])
    k_pages = jax.random.normal(k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype)
    v_pages = jax.random.normal(k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype)

    # Generate page indices: Create a permutation for each sequence
    # Ensure indices within a sequence map uniquely to the global page pool slice for that sequence
    all_page_indices = []
    global_page_offset = 0
    for _ in range(batch_size):
        seq_global_indices = jnp.arange(global_page_offset, global_page_offset + pages_per_sequence, dtype=jnp.int32)
        # Permute indices *within* the range allocated for this sequence
        permuted_indices = jax.random.permutation(k3, seq_global_indices, independent=True)
        k3, _ = jax.random.split(k3)  # Consume key
        all_page_indices.append(permuted_indices)
        global_page_offset += pages_per_sequence

    page_indices = jnp.stack(all_page_indices)  # Shape: [batch_size, pages_per_sequence]

    # Generate Q (layout: [batch_size, num_heads, head_dim])
    q = jax.random.normal(k4, (batch_size, num_heads, head_dim), dtype=dtype)

    return q, k_pages, v_pages, page_indices


def _reconstruct_kv(page_indices: jax.Array, pages: jax.Array) -> jax.Array:
    """Reconstructs the full K/V tensor from paged format for reference."""
    batch_size = page_indices.shape[0]
    num_heads, _, page_size, head_dim = pages.shape

    def gather_for_head(head_pages, indices_per_batch):
        return jax.vmap(lambda idx: head_pages[idx, :, :])(indices_per_batch)

    gathered_per_head = jax.vmap(gather_for_head, in_axes=(0, None))(pages, page_indices)
    gathered_swapped = gathered_per_head.transpose(1, 0, 2, 3, 4)
    max_seq_len = page_indices.shape[1] * page_size
    return gathered_swapped.reshape(batch_size, num_heads, max_seq_len, head_dim)


def _grouped_query_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: np.ndarray,
    attn_logits_soft_cap: float | None,
) -> jax.Array:
    """Reference implementation for grouped query attention."""
    batch_size, num_heads, head_dim = q.shape
    _, num_kv_heads, max_seq_len, _ = k.shape

    if k.shape != v.shape:
        raise ValueError("K and V shapes must match")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    # Reshape Q for grouped query attention calculation
    q_grouped = q.reshape(batch_size, num_kv_heads, num_heads // num_kv_heads, head_dim)

    # Compute attention logits (using float32 for stability)
    # q: [b, h_kv, g, d], k: [b, h_kv, t, d] -> logits: [b, h_kv, g, t]
    logits = jnp.einsum("bhgd,bhtd->bhgt", q_grouped.astype(jnp.float32), k.astype(jnp.float32))

    # Apply optional logit capping
    if attn_logits_soft_cap is not None:
        logits = jnp.tanh(logits / attn_logits_soft_cap) * attn_logits_soft_cap

    # Apply causal/padding mask based on sequence lengths
    mask = jnp.arange(max_seq_len)[None, :] < lengths[:, None]  # Shape: [b, t]
    mask_value = np.finfo(np.float32).min / 2.0  # Use a large negative number
    # Broadcast mask: [b, t] -> [b, 1, 1, t]
    logits = logits + jnp.where(mask[:, None, None, :], 0.0, mask_value)

    # Compute attention weights
    weights = jax.nn.softmax(logits, axis=-1)  # Shape: [b, h_kv, g, t]

    # Compute output
    # weights: [b, h_kv, g, t], v: [b, h_kv, t, d] -> o_grouped: [b, h_kv, g, d]
    o_grouped = jnp.einsum("bhgt,bhtd->bhgd", weights.astype(v.dtype), v)

    # Reshape output back to [batch_size, num_heads, head_dim]
    return o_grouped.reshape(batch_size, num_heads, head_dim)


# --- Pytest Parameterization ---

# Define the parameter combinations using itertools.product
param_combinations = list(
    itertools.product(
        (jnp.bfloat16,),  # dtype
        (16, 64),  # page_size
        (2, 4, 8),  # num_kv_heads
        (8, 1),  # q_kv_head_ratio (1 for MHA, >1 for GQA)
        (128, 256),  # head_dim
        (None, "batch", "kv_head"),  # megacore_mode
        (None, 50.0),  # attn_logits_soft_cap
    )
)

# Create parameter names string matching the order in param_combinations
param_names = "dtype, page_size, num_kv_heads, q_kv_head_ratio, head_dim, megacore_mode, attn_logits_soft_cap"


@pytest.mark.parametrize(param_names, param_combinations)
def test_paged_attention(
    dtype: jnp.dtype,
    page_size: int,
    num_kv_heads: int,
    q_kv_head_ratio: int,
    head_dim: int,
    megacore_mode: str | None,
    attn_logits_soft_cap: float | None,
):
    """Tests the Pallas paged attention kernel against a reference implementation."""
    # Skip invalid configurations
    if num_kv_heads % 2 != 0 and megacore_mode == "kv_head":
        pytest.skip(f"Skip kv_head megacore mode when num_kv_heads ({num_kv_heads}) is odd")

    # Derived parameters
    num_heads = num_kv_heads * q_kv_head_ratio
    seq_lens = TEST_SEQ_LENS[TEST_SEQ_LENS <= MAX_SEQ_LEN]

    # Generate test data
    key = jax.random.key(0)  # Use a fixed key for reproducibility
    q, k_pages, v_pages, page_indices = _generate_qkv(
        seq_lens,
        page_size,
        MAX_SEQ_LEN,
        num_kv_heads,
        num_heads,
        head_dim,
        key,
        dtype,
    )

    # Run the Pallas kernel
    o_pallas = pallas_paged_attention(
        q,
        k_pages,
        v_pages,
        seq_lens,
        page_indices,
        pages_per_compute_block=BLOCK_SIZE // page_size,
        megacore_mode=megacore_mode,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    # Reconstruct K/V for the reference calculation
    k_ref = _reconstruct_kv(page_indices, k_pages)
    v_ref = _reconstruct_kv(page_indices, v_pages)

    # Run the reference implementation
    o_ref = _grouped_query_attention_reference(q, k_ref, v_ref, seq_lens, attn_logits_soft_cap)

    # Determine tolerances based on MHA vs GQA
    if q_kv_head_ratio > 1:
        atol, rtol = GQA_ATOL, GQA_RTOL
    else:
        atol, rtol = MHA_ATOL, MHA_RTOL

    # Compare results, excluding sequences with length 0
    valid_indices = np.where(seq_lens > 0)[0]
    if valid_indices.size > 0:
        # Use jtu.check_close for JAX array comparison
        jtu.check_close(o_pallas[valid_indices], o_ref[valid_indices], atol=atol, rtol=rtol)
    else:
        # If all sequences have length 0, the output should be all zeros (or empty)
        assert o_pallas.shape == o_ref.shape  # Check shapes still match
        # Optionally check if the output is zero, depending on kernel behavior for len 0
        # assert np.all(o_pallas == 0)


if __name__ == "__main__":
    pytest.main([__file__])
