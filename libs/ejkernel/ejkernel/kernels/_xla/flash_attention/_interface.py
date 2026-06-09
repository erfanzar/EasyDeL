# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Flash Attention implementation using XLA/JAX operations.

This module provides a memory-efficient implementation of Flash Attention
using pure JAX operations that compiles to XLA. It implements the online
softmax algorithm to compute attention in chunks, reducing memory usage
from O(N²) to O(N) for sequence length N.

Flash Attention is a breakthrough algorithm that maintains exact attention
semantics while dramatically reducing memory footprint. The key insight is
to split the attention computation into blocks and use online softmax to
avoid materializing the full attention matrix.

Key advantages over standard attention:
1. Subquadratic memory: O(N) instead of O(N²) for sequence length N
2. Hardware-agnostic: Works on any XLA-supported backend (CPU, GPU, TPU)
3. Exact attention: No approximation, produces identical results to standard attention
4. Custom VJP: Efficient gradient computation through custom backward pass

Algorithm overview:
- Query sequence is processed in chunks of size chunk_size_q
- For each query chunk, iterate through key/value chunks of size chunk_size_k
- Online softmax accumulates partial results without full materialization
- Numerically stable using log-sum-exp trick with running maximum

Supported features:
- Causal and non-causal (bidirectional) attention
- Attention bias for relative position encodings
- Boolean attention masks for padding
- Segment IDs for packed sequence processing
- Sliding window attention for local patterns
- Grouped-query attention (GQA) and multi-query attention (MQA)
- Attention sinks via softmax_aux parameter
- Logits soft capping for numerical stability
- Dropout with reproducible randomness
- Multiple precision modes (DEFAULT, HIGH, HIGHEST)

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.flash_attention import flash_attention
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 2048, 8, 64
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>> # Basic causal attention
    >>> output = flash_attention(q, k, v, causal=True)
    >>>
    >>> # With sliding window
    >>> output = flash_attention(q, k, v, causal=True, sliding_window=256)
    >>>
    >>> # With attention sinks (4 sink tokens)
    >>> sinks = jnp.zeros((4,))
    >>> output = flash_attention(q, k, v, causal=True, softmax_aux=sinks)

Reference:
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
    https://arxiv.org/abs/2205.14135

Internal Functions:
    _make_core_func: Creates specialized attention cores with custom VJP for given static params
    _precision_to_code: Convert JAX precision enum to integer code for caching
    _dtype_to_code: Convert dtype to integer code for JIT compilation caching
"""

from __future__ import annotations

import math

import chex
import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

from ejkernel.callib._ejit import ejit
from ejkernel.errors import EjkernelRuntimeError
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _flash_attention_bwd
from ._xla_impl_fwd import _flash_attention_fwd

PagedKV = Float[Array, "num_blocks block_size num_kv_heads head_dim"]
DenseKV = Float[Array, "batch seq_len_k num_kv_heads head_dim"]
BlockTables = Int[Array, "batch max_blocks"]

_PREC_TO_CODE = {
    jax.lax.Precision.DEFAULT: 0,
    jax.lax.Precision.HIGHEST: 1,
    jax.lax.Precision.HIGH: 2,
}
_CODE_TO_PREC = {
    0: jax.lax.Precision.DEFAULT,
    1: jax.lax.Precision.HIGHEST,
    2: jax.lax.Precision.HIGH,
}
_DTYPE_TO_CODE = {
    jnp.dtype("float16"): 0,
    jnp.dtype("bfloat16"): 1,
    jnp.dtype("float32"): 2,
    jnp.dtype("float64"): 3,
}
_CODE_TO_DTYPE = {
    0: jnp.float16,
    1: jnp.bfloat16,
    2: jnp.float32,
    3: jnp.float64,
}


def _precision_to_code(precision) -> int:
    """Convert JAX precision enum to integer code for function caching.

    This enables caching of compiled attention cores based on precision settings,
    since JAX precision enums are not hashable in the same way across sessions.

    Args:
        precision: JAX precision setting (jax.lax.Precision.DEFAULT/HIGH/HIGHEST)
            or an integer code (0, 1, 2).

    Returns:
        Integer code representing the precision level.

    Raises:
        ValueError: If precision is not a valid JAX precision enum or int code.
    """
    if isinstance(precision, int):
        return int(precision)
    try:
        return _PREC_TO_CODE[precision]
    except KeyError as e:
        raise ValueError("precision must be jax.lax.Precision.{DEFAULT|HIGHEST|HIGH} or an int code {0,1,2}.") from e


def _dtype_to_code(dtype) -> int:
    """Convert dtype to integer code for function caching.

    This enables caching of compiled attention cores based on logits dtype,
    since dtypes need to be converted to hashable values for the cache key.

    Args:
        dtype: JAX/NumPy dtype (float16, bfloat16, float32, or float64).

    Returns:
        Integer code representing the dtype.

    Raises:
        ValueError: If dtype is not one of the supported types.
    """
    d = jnp.dtype(dtype)
    try:
        return _DTYPE_TO_CODE[d]
    except KeyError as e:
        raise ValueError("logits_dtype must be one of float16, bfloat16, float32, float64.") from e


def _make_core_func(
    precision_code_val: int,
    logits_dtype_code_val: int,
    chunk_size_q_val: int,
    chunk_size_k_val: int,
    normalize_output_val: bool,
    causal_val: bool,
    dropout_prob_val: float,
):
    """Create a specialized flash attention core function with custom VJP.

    This factory function creates a specialized attention function with the given
    static parameters baked in. The returned function has a custom VJP (Vector-Jacobian
    Product) defined for efficient gradient computation.

    The specialization enables XLA to optimize the computation for the specific
    configuration, and the caching mechanism (_CORE_FUNC_CACHE) ensures each
    unique configuration is compiled only once.

    Args:
        precision_code_val: Integer code for JAX precision setting.
        logits_dtype_code_val: Integer code for logits computation dtype.
        chunk_size_q_val: Number of query tokens to process per chunk.
        chunk_size_k_val: Number of key/value tokens to process per chunk.
        normalize_output_val: Whether to normalize output by attention weights sum.
        causal_val: Whether to apply causal masking.
        dropout_prob_val: Dropout probability for attention weights.

    Returns:
        A specialized flash attention function with custom VJP defined.
    """
    precision = _CODE_TO_PREC[precision_code_val]
    logits_dtype = _CODE_TO_DTYPE[logits_dtype_code_val]

    @jax.custom_vjp
    def _flash_attention_core_specialized(
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        bias: chex.Array | None,
        attention_mask: chex.Array | None,
        q_segment_ids: chex.Array | None,
        kv_segment_ids: chex.Array | None,
        softmax_aux: chex.Array | None,
        sliding_window: tuple[int, int] | None,
        softmax_scale: float,
        logits_soft_cap: float | None,
        dropout_key: PRNGKeyArray | None,
    ) -> chex.Array:
        """Core flash attention computation with custom VJP.

        This inner function performs the actual chunked attention computation
        using online softmax. Static parameters (precision, chunk sizes, etc.)
        are captured from the enclosing scope.

        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim].
            key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim].
            value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim].
            bias: Optional attention bias [batch, num_heads, seq_len_q, seq_len_k].
            attention_mask: Optional boolean mask [batch, 1, seq_len_q, seq_len_k].
            q_segment_ids: Optional query segment IDs [batch, seq_len_q].
            kv_segment_ids: Optional key/value segment IDs [batch, seq_len_k].
            softmax_aux: Optional attention sink logits [num_sinks].
            sliding_window: Optional (left, right) window bounds.
            softmax_scale: Scaling factor for attention scores.
            logits_soft_cap: Optional soft cap for attention logits.
            dropout_key: Optional PRNG key for dropout.

        Returns:
            Attention output [batch, seq_len_q, num_heads, head_dim].
        """
        return _flash_attention_fwd(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            bias=bias,
            mask=attention_mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            window=sliding_window,
            chunk_size_q=chunk_size_q_val,
            chunk_size_k=chunk_size_k_val,
            normalize_output=normalize_output_val,
            precision=precision,
            logits_dtype=logits_dtype,
            softmax_aux=softmax_aux,
            causal=causal_val,
            dropout_prob=dropout_prob_val,
            dropout_key=dropout_key,
        )

    def _fwd(
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        bias: chex.Array | None,
        attention_mask: chex.Array | None,
        q_segment_ids: chex.Array | None,
        kv_segment_ids: chex.Array | None,
        softmax_aux: chex.Array | None,
        sliding_window: tuple[int, int] | None,
        softmax_scale: float,
        logits_soft_cap: float | None,
        dropout_key: PRNGKeyArray | None,
    ):
        """Forward pass for custom VJP: compute output and save residuals.

        This function is called during the forward pass of differentiation.
        It computes the attention output and returns residuals (context) needed
        for the backward pass gradient computation.

        Returns:
            Tuple of (attention_output, residuals) where residuals contain all
            values needed for backward pass gradient computation.
        """
        y = _flash_attention_fwd(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            bias=bias,
            mask=attention_mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            window=sliding_window,
            chunk_size_q=chunk_size_q_val,
            chunk_size_k=chunk_size_k_val,
            normalize_output=normalize_output_val,
            precision=precision,
            logits_dtype=logits_dtype,
            softmax_aux=softmax_aux,
            causal=causal_val,
            dropout_prob=dropout_prob_val,
            dropout_key=dropout_key,
        )

        ctx = (
            bias,
            attention_mask,
            q_segment_ids,
            kv_segment_ids,
            softmax_aux,
            sliding_window,
            softmax_scale,
            logits_soft_cap,
            chunk_size_q_val,
            chunk_size_k_val,
            normalize_output_val,
            query,
            key,
            value,
            causal_val,
            dropout_prob_val,
            dropout_key,
        )
        return y, ctx

    def _bwd(ctx, g):
        """Backward pass for custom VJP: compute gradients from residuals.

        This function computes gradients with respect to query, key, and value
        tensors using the saved residuals from the forward pass.

        Args:
            ctx: Residuals tuple from forward pass containing (bias, attention_mask,
                q_segment_ids, kv_segment_ids, softmax_aux, sliding_window, softmax_scale,
                logits_soft_cap, chunk_size_q, chunk_size_k, normalize_output,
                query, key, value, causal, dropout_prob, dropout_key).
            g: Gradient of the loss with respect to the attention output.

        Returns:
            Tuple of gradients for each input argument. Non-differentiable inputs
            (masks, segment_ids, etc.) return None.
        """
        (
            bias,
            attention_mask,
            q_segment_ids,
            kv_segment_ids,
            softmax_aux,
            sliding_window,
            softmax_scale,
            logits_soft_cap,
            chunk_size_q_val_,
            chunk_size_k_val_,
            normalize_output_val_,
            query,
            key,
            value,
            causal_val_,
            dropout_prob_val_,
            dropout_key,
        ) = ctx

        dq, dk, dv = _flash_attention_bwd(
            bias=bias,
            mask=attention_mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            softmax_aux=softmax_aux,
            window=sliding_window,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            chunk_size_q=chunk_size_q_val_,
            chunk_size_k=chunk_size_k_val_,
            normalize_output=normalize_output_val_,
            precision_code=precision_code_val,
            logits_dtype_code=logits_dtype_code_val,
            causal=causal_val_,
            dropout_prob=dropout_prob_val_,
            dropout_key=dropout_key,
            res=(query, key, value),
            g=g,
        )

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    _flash_attention_core_specialized.defvjp(_fwd, _bwd)
    return _flash_attention_core_specialized


_CORE_FUNC_CACHE = {}


def _flash_attention_core(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    bias: chex.Array | None,
    attention_mask: chex.Array | None,
    q_segment_ids: chex.Array | None,
    kv_segment_ids: chex.Array | None,
    softmax_aux: chex.Array | None,
    sliding_window: tuple[int, int] | None,
    softmax_scale: float,
    logits_soft_cap: float | None,
    chunk_size_q: int,
    chunk_size_k: int,
    normalize_output: bool,
    precision_code: int,
    logits_dtype_code: int,
    causal: bool,
    dropout_prob: float,
    dropout_key: PRNGKeyArray | None,
) -> chex.Array:
    """Dispatch flash attention to cached specialized core function.

    This function looks up or creates a specialized attention function for the
    given static configuration and calls it with the dynamic tensor arguments.
    The caching mechanism ensures each unique configuration is compiled only once.

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim].
        key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim].
        value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim].
        bias: Optional attention bias.
        attention_mask: Optional boolean attention mask.
        q_segment_ids: Optional query segment IDs.
        kv_segment_ids: Optional key/value segment IDs.
        softmax_aux: Optional attention sink logits.
        sliding_window: Optional (left, right) window bounds.
        softmax_scale: Scaling factor for attention scores.
        logits_soft_cap: Optional soft cap for attention logits.
        chunk_size_q: Query chunk size for blocked computation.
        chunk_size_k: Key/value chunk size for blocked computation.
        normalize_output: Whether to normalize by attention weight sum.
        precision_code: Integer code for JAX precision.
        logits_dtype_code: Integer code for logits dtype.
        causal: Whether to apply causal masking.
        dropout_prob: Dropout probability.
        dropout_key: Optional PRNG key for dropout.

    Returns:
        Attention output [batch, seq_len_q, num_heads, head_dim].
    """
    cache_key = (precision_code, logits_dtype_code, chunk_size_q, chunk_size_k, normalize_output, causal, dropout_prob)
    if cache_key not in _CORE_FUNC_CACHE:
        _CORE_FUNC_CACHE[cache_key] = _make_core_func(
            precision_code, logits_dtype_code, chunk_size_q, chunk_size_k, normalize_output, causal, dropout_prob
        )

    return _CORE_FUNC_CACHE[cache_key](
        query,
        key,
        value,
        bias,
        attention_mask,
        q_segment_ids,
        kv_segment_ids,
        softmax_aux,
        sliding_window,
        softmax_scale,
        logits_soft_cap,
        dropout_key,
    )


@kernel_registry.register("flash_attention", Platform.XLA, Backend.ANY)
@ejit(
    static_argnames=[
        "softmax_scale",
        "dropout_prob",
        "causal",
        "dropout_seed",
        "sliding_window",
        "fwd_params",
        "bwd_params",
        "logits_soft_cap",
        "normalize_output",
        "logits_dtype",
        "precision",
    ]
)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: DenseKV | PagedKV,
    value: DenseKV | PagedKV,
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    *,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    block_tables: BlockTables | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Compute flash attention with memory-efficient chunked computation.

    This implementation uses online softmax to compute attention in chunks,
    reducing memory usage from O(N²) to O(N). It supports all modern attention
    features including sliding window, logit soft capping, GQA/MQA, and
    attention sinks.

    The algorithm processes queries in blocks and maintains running statistics
    (max and sum) for numerically stable softmax computation without ever
    materializing the full attention matrix.

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim].
        key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim].
        value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim].
        attention_mask: Optional boolean/int mask [batch, 1, seq_len_q, seq_len_k].
            True/1 values indicate positions to attend to.
        bias: Optional attention bias [batch, num_heads, seq_len_q, seq_len_k].
        softmax_scale: Scaling factor for QK^T. Defaults to 1/sqrt(head_dim).
        dropout_prob: Dropout probability for attention weights. Default 0.0.
        causal: If True, applies causal masking. Default False.
        dropout_seed: Integer seed for dropout PRNG. Required if dropout_prob > 0.
        cum_seqlens_q: Not implemented in XLA backend.
        cum_seqlens_k: Not implemented in XLA backend.
        sliding_window: Local attention window. Can be:
            - int: Symmetric window (same left and right)
            - tuple[int, int]: Asymmetric (left_window, right_window)
            - None: Full attention (default)
        fwd_params: Forward pass parameters (q_blocksize, kv_blocksize, etc.).
        bwd_params: Backward pass parameters (unused in XLA backend).
        logits_soft_cap: Soft cap for attention logits via tanh.
            Applies: cap * tanh(logits / cap). Common in Gemma2.
        softmax_aux: Attention sink logits [num_sinks] or [num_heads, num_sinks].
            Participate in softmax but don't contribute to output.
        normalize_output: Whether to normalize output by sum of weights. Default True.
        precision: JAX precision for matrix multiplications.
            One of: Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST.
        logits_dtype: Dtype for logits computation. Default float32.
        block_tables: Optional paged-KV block table of shape
            ``(batch, max_blocks)``. Unsupported by the XLA backend.
        q_segment_ids: Query segment IDs [batch, seq_len_q] for packed sequences.
        kv_segment_ids: Key/value segment IDs [batch, seq_len_k] for packed sequences.

    Returns:
        Attention output [batch, seq_len_q, num_heads, head_dim].

    Raises:
        NotImplementedError: If cum_seqlens_q or cum_seqlens_k are provided.
        ValueError: If sliding_window contains negative values.

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._xla.flash_attention import flash_attention
        >>>
        >>> batch, seq_len, num_heads, head_dim = 2, 2048, 8, 64
        >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>>
        >>> output = flash_attention(q, k, v, causal=True)
        >>> output.shape
        (2, 2048, 8, 64)

    Note:
        This is the XLA/JAX reference implementation. For GPU workloads with
        CUDA support, consider using the Triton-based flash_attention for
        potentially better performance.
    """
    reasons: list[str] = []
    if cum_seqlens_k is not None and attention_mask is None:
        reasons.append("cum_seqlens_k requires attention_mask on XLA")
    if cum_seqlens_q is not None and attention_mask is None:
        reasons.append("cum_seqlens_q requires attention_mask on XLA")
    if block_tables is not None:
        reasons.append("block_tables (paged_kv) is not supported")
    if reasons:
        raise EjkernelRuntimeError("flash_attention (platform=xla): " + "; ".join(reasons))

    dropout_key = None
    if dropout_prob > 0.0:
        if dropout_seed is None:
            dropout_seed = 0
        dropout_key = jax.random.PRNGKey(dropout_seed)

    if isinstance(sliding_window, int):
        window_tuple = (int(sliding_window), int(sliding_window))
    elif sliding_window is None:
        window_tuple = None
    else:
        window_left, window_right = sliding_window
        if window_left < 0 or window_right < 0:
            raise ValueError("Window bounds must be non-negative.")
        window_tuple = (int(window_left), int(window_right))

    if softmax_scale is None:
        D = query.shape[-1]
        scale_val = float(1.0 / math.sqrt(D))
    else:
        scale_val = float(softmax_scale)

    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=min(128, query.shape[1]), kv_blocksize=min(128, key.shape[1]))

    q_block = min(128, query.shape[1]) if fwd_params.q_blocksize is None else int(fwd_params.q_blocksize)
    kv_block = min(128, key.shape[1]) if fwd_params.kv_blocksize is None else int(fwd_params.kv_blocksize)
    q_block = max(1, q_block)
    kv_block = max(1, kv_block)
    if logits_soft_cap is not None:
        min_block = 32 if softmax_aux is not None else 16
        q_block = max(min_block, q_block)
        kv_block = max(min_block, kv_block)

    precision_code = _precision_to_code(precision)
    logits_dtype_code = _dtype_to_code(logits_dtype)

    return _flash_attention_core(
        query,
        key,
        value,
        bias,
        attention_mask,
        q_segment_ids,
        kv_segment_ids,
        softmax_aux,
        window_tuple,
        scale_val,
        logits_soft_cap,
        q_block,
        kv_block,
        bool(normalize_output),
        precision_code,
        logits_dtype_code,
        bool(causal),
        float(dropout_prob),
        dropout_key,
    )
