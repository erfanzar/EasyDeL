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


"""Forward implementation of GPU scaled dot-product attention via cuDNN.

Despite residing in the ``_pallas`` tree, this module does NOT use a
custom Pallas/Triton kernel.  It delegates entirely to
``jax.nn.dot_product_attention(..., implementation="cudnn")``, which invokes
NVIDIA cuDNN's highly optimised FlashAttention kernel on CUDA-capable devices.

Tensor layout convention (same as JAX SDPA):
    query, key, value: ``[batch, seq_len, num_heads, head_dim]``
    output:            ``[batch, seq_len, num_q_heads, head_dim]``

Note:
    On non-CUDA devices (e.g., CPU, TPU) the cuDNN path will raise at
    runtime.  Use the corresponding ``_xla`` or ``_pallas/tpu`` backend for
    those platforms.
"""

import jax
from beartype.typing import Callable
from jaxtyping import Array, Bool, Float, Int


def scaled_dot_product_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    cum_seqlens_q: Int[Array, "batch"] | None = None,
    cum_seqlens_k: Int[Array, "batch"] | None = None,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Compute scaled dot-product attention via NVIDIA cuDNN.

    Delegates to ``jax.nn.dot_product_attention(..., implementation="cudnn")``.
    All masking, bias, and variable-length features are handled by cuDNN.

    Tensor layout (JAX SDPA convention):
        ``[batch, seq_len, num_heads, head_dim]``

    Args:
        query: Query tensor of shape ``[batch, seq_len, num_q_heads, head_dim]``.
        key: Key tensor of shape ``[batch, kv_len, num_kv_heads, head_dim]``.
        value: Value tensor of shape ``[batch, kv_len, num_kv_heads, head_dim]``.
        attention_mask: Optional boolean mask of shape
            ``[batch, num_heads_or_1, seq_len, kv_len]``.
            Positions with ``True`` are attended; ``False`` positions are
            masked out.  Passed as ``mask`` to ``jax.nn.dot_product_attention``.
            Prefer ``bias`` for additive masking; this parameter is for
            compatibility with legacy boolean masks.
        bias: Optional additive attention bias of shape
            ``[batch, num_heads, seq_len, kv_len]``.  Added to logits before
            softmax.  When ``init_bias`` is provided and ``bias`` is ``None``,
            the bias is computed by calling ``init_bias()``.
        init_bias: Optional zero-argument callable returning a bias tensor.
            Called only when ``bias`` is ``None``.  Useful for deferring
            bias construction until the kernel is actually invoked.
        softmax_scale: Multiplicative scale applied to logits before softmax.
            Defaults to ``1/sqrt(head_dim)`` (cuDNN default) when ``None``.
        causal: If ``True``, applies causal (lower-triangular) masking.
            Passed as ``is_causal`` to ``jax.nn.dot_product_attention``.
        sliding_window: Optional local-attention window.  Can be:
            - ``int``: symmetric window of that size.
            - ``tuple[int, int]``: ``(left_window, right_window)`` for
              asymmetric windows.
            Passed as ``local_window_size`` to cuDNN.
        cum_seqlens_q: Optional cumulative query sequence lengths for packed
            (variable-length) inputs, shape ``[batch+1]``.  Passed as
            ``query_seq_lengths``.
        cum_seqlens_k: Optional cumulative key/value sequence lengths for
            packed inputs, shape ``[batch+1]``.  Passed as
            ``key_value_seq_lengths``.

    Returns:
        Attention output of shape ``[batch, seq_len, num_q_heads, head_dim]``.

    Note:
        This function requires a CUDA-capable device with cuDNN installed.
        It does not fall back gracefully on CPU or TPU.

    Example:
        >>> import jax.numpy as jnp
        >>> query = jnp.ones((2, 512, 8, 64))
        >>> key = jnp.ones((2, 512, 8, 64))
        >>> value = jnp.ones((2, 512, 8, 64))
        >>> output = scaled_dot_product_attention(query, key, value, causal=True)
        >>> output = scaled_dot_product_attention(
        ...     query, key, value, causal=True, sliding_window=256
        ... )
        >>> bias = jnp.ones((2, 8, 512, 512)) * -1e9
        >>> output = scaled_dot_product_attention(query, key, value, bias=bias)
    """
    if bias is None and init_bias is not None:
        bias = init_bias()
    return jax.nn.dot_product_attention(
        query=query,
        key=key,
        value=value,
        mask=attention_mask,
        bias=bias,
        is_causal=causal,
        scale=softmax_scale,
        local_window_size=sliding_window,
        key_value_seq_lengths=cum_seqlens_k,
        query_seq_lengths=cum_seqlens_q,
        implementation="cudnn",
    )
