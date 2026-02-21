"""Attention output container for EasyDeL attention operations.

This module provides the AttentionOutput dataclass that encapsulates the results
of attention computations across all attention implementations in EasyDeL.

The AttentionOutput class serves as a standardized container for:
- Primary attention outputs (the attended representations)
- Optional attention weights (when requested and supported)
- Optional cache view (for KV-cache management during inference)

This design allows different attention implementations (vanilla, flash, ring,
blocksparse, etc.) to return consistent output structures while supporting
their unique capabilities.

Example:
    >>> from easydel.layers.operations import FlashAttn, OperationMetadata
    >>>
    >>> metadata = OperationMetadata(runtime_dtype=jnp.float16)
    >>> attn = FlashAttn(metadata)
    >>> output = attn(query, key, value, causal=True)
    >>>
    >>> # Access the attended representations
    >>> attended = output.attention_outputs
    >>>
    >>> # Attention weights may be None for memory-efficient implementations
    >>> if output.attention_weights is not None:
    ...     # Inspect attention patterns
    ...     print(output.attention_weights.shape)
"""

from __future__ import annotations

from eformer.pytree import auto_pytree
from jax import Array
from jaxtyping import Float

from ..caching import RaggedPagesCacheView, TransformerCacheView, UnifiedAttentionCacheView
from ._operation_impl import OperationOutput


@auto_pytree
class AttentionOutput(OperationOutput):
    """Container for attention computation results.

    This dataclass encapsulates all outputs from attention operations, providing
    a standardized interface across different attention implementations in EasyDeL.

    The class is decorated with `@auto_pytree` to enable seamless integration
    with JAX's pytree system, allowing AttentionOutput instances to be used
    in JAX transformations like `jax.jit`, `jax.vmap`, and `jax.pmap`.

    Attributes:
        attention_weights: Optional attention weight matrix with shape
            [batch, num_heads, seq_len, seq_len] (or [batch, num_heads, seq_len, kv_len]
            for cross-attention). Contains the softmax-normalized attention scores
            between query and key positions. This is typically None for memory-efficient
            implementations like FlashAttention that avoid materializing the full
            attention matrix. Available when explicitly requested and supported by
            the attention implementation.
        attention_outputs: The primary attention output tensor with shape
            [batch, seq_len, num_heads, head_dim]. Contains the attended
            representations computed as the weighted combination of value vectors
            based on attention scores. This is the main output used for downstream
            processing and is always populated.
        cache_view: Optional cache view containing updated key-value cache state
            for autoregressive generation. Can be one of:
            - TransformerCacheView: Standard transformer KV-cache
            - RaggedPagesCacheView: Paged attention cache for continuous batching
            - UnifiedAttentionCacheView: vLLM-style unified paged attention cache
            - None: When caching is not used (e.g., during training)

    Example:
        >>> # Standard attention output
        >>> output = AttentionOutput(
        ...     attention_weights=weights,  # Optional, may be None
        ...     attention_outputs=attended,  # Always present
        ...     cache_view=updated_cache,   # For inference with caching
        ... )
        >>>
        >>> # Access outputs
        >>> hidden_states = output.attention_outputs
        >>> if output.cache_view is not None:
        ...     # Update cache for next generation step
        ...     cache = output.cache_view

    Note:
        Different attention implementations populate these fields differently:
        - VanillaAttn: May provide attention_weights
        - FlashAttn, RingAttn: attention_weights is None (memory efficient)
        - Inference operations: Typically include cache_view
        - Training operations: cache_view is typically None
    """

    attention_weights: Float[Array, "batch num_heads seq_len seq_len"] | None = None
    attention_outputs: Float[Array, "batch seq_len num_heads head_dim"] | None = None
    cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None
