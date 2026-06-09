# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-head attention layers.

:class:`MultiheadAttention` implements the standard "Attention Is All
You Need" multi-head attention with independent Q, K, V projections
plus an output projection, optional KV-cache decoding for incremental
inference, and dropout on the attention weights. Heads are split
internally into ``(..., num_heads, seq, head_dim)`` and re-merged
after attention so external API stays at ``(..., seq, embed_dim)``.

:class:`CausalSelfAttention` is a thin convenience wrapper around
:class:`MultiheadAttention` that hardcodes ``is_causal=True`` and
exposes a single-input ``forward(x)`` signature.

Decode caches live in :class:`~spectrax.Buffer` cells under the
``"cache"`` collection so writes during incremental decode require
``mutable="cache"`` on the surrounding transform.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import Buffer
from ..functional.attention import scaled_dot_product_attention
from ..rng.rngs import Rngs
from .linear import Linear


class MultiheadAttention(Module):
    """Standard multi-head attention with separate Q/K/V/out projections.

    Layout: external inputs and outputs are
    ``(..., seq, embed_dim)``; the heads are split internally to
    ``(..., num_heads, seq, head_dim)`` (with ``head_dim =
    embed_dim // num_heads``) and merged again after the attention
    primitive.

    Decoding mode (``decode=True``): when :meth:`init_cache` has been
    called, every :meth:`forward` step appends the freshly-projected
    K/V slices into the cache buffers via
    :func:`jax.lax.dynamic_update_slice_in_dim`, attends against the
    full cached K/V, and advances :attr:`cache_index` by the input's
    sequence length. The decode mask zeroes out future positions
    relative to the current write index (additive triangular mask).

    Mutability: cache writes require ``mutable="cache"`` on the
    surrounding transform — the cache cells are :class:`~spectrax.Buffer`
    s with kind ``"cache"`` to make that explicit.
    """

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear
    cache_k: Buffer
    cache_v: Buffer
    cache_index: Buffer

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        use_bias: bool = True,
        dropout: float = 0.0,
        decode: bool = False,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        qkv_sharding: Sharding | AxisNames | None = None,
        out_sharding: Sharding | AxisNames | None = None,
        qkv_bias_sharding: Sharding | AxisNames | None = None,
        out_bias_sharding: Sharding | AxisNames | None = None,
        cache_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            embed_dim: Model (token-embedding) dimension. Must be
                exactly divisible by ``num_heads``.
            num_heads: Number of attention heads. ``head_dim`` is set
                to ``embed_dim // num_heads``.
            use_bias: Whether the Q / K / V / out projections carry
                biases. Forwarded to the four embedded :class:`Linear`
                instances.
            dropout: Attention-weight dropout probability, applied
                inside :func:`scaled_dot_product_attention` only when
                the module is in training mode and a per-call
                :class:`Rngs` is supplied.
            decode: When ``True``, switches :meth:`forward` into
                incremental-decoding mode. Requires :meth:`init_cache`
                to be called once before the first decoding step.
            rngs: PRNG source used by the four embedded projection
                linears for parameter initialization.
            dtype: Parameter dtype forwarded to the projections.
            param_dtype: Alias for ``dtype``; takes precedence when
                both are supplied.
            qkv_sharding: Optional sharding for the Q / K / V weights
                (shared across the three).
            out_sharding: Optional sharding for the output-projection
                weight.
            qkv_bias_sharding: Optional sharding for the Q / K / V
                biases.
            out_bias_sharding: Optional sharding for the
                output-projection bias.
            cache_sharding: Optional sharding for the K / V cache
                buffers (used only when :meth:`init_cache` is called).

        Raises:
            ValueError: If ``embed_dim`` is not divisible by
                ``num_heads``.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.decode = decode
        self.cache_sharding = cache_sharding
        self.q_proj = Linear(
            embed_dim,
            embed_dim,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=qkv_sharding,
            bias_sharding=qkv_bias_sharding,
        )
        self.k_proj = Linear(
            embed_dim,
            embed_dim,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=qkv_sharding,
            bias_sharding=qkv_bias_sharding,
        )
        self.v_proj = Linear(
            embed_dim,
            embed_dim,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=qkv_sharding,
            bias_sharding=qkv_bias_sharding,
        )
        self.out_proj = Linear(
            embed_dim,
            embed_dim,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=out_sharding,
            bias_sharding=out_bias_sharding,
        )

    def init_cache(
        self,
        batch_shape: tuple[int, ...],
        max_length: int,
        *,
        dtype: DType | None = None,
    ) -> None:
        """Allocate the K/V cache and the write-index buffers.

        Creates three :class:`~spectrax.Buffer` cells in the
        ``"cache"`` collection:

        * ``cache_k`` and ``cache_v`` of shape
          ``(*batch_shape, num_heads, max_length, head_dim)``
          (zero-initialized);
        * ``cache_index``, a 0-d ``int32`` scalar for the next write
          position.

        Must be called once before the first ``decode=True`` forward
        step. Subsequent steps mutate the cache in place; the
        surrounding transform must therefore declare
        ``mutable="cache"``.

        Args:
            batch_shape: Leading batch dimensions; whatever is
                consistent with the future query inputs.
            max_length: Maximum sequence length the cache will hold.
            dtype: Storage dtype for ``cache_k`` / ``cache_v``;
                defaults to ``float32``. Note ``cache_index`` is
                always ``int32``.
        """
        dt = dtype or jnp.float32
        shape = (*batch_shape, self.num_heads, max_length, self.head_dim)
        self.cache_k = Buffer(jnp.zeros(shape, dtype=dt), kind="cache", sharding=self.cache_sharding)
        self.cache_v = Buffer(jnp.zeros(shape, dtype=dt), kind="cache", sharding=self.cache_sharding)
        self.cache_index = Buffer(jnp.zeros((), dtype=jnp.int32), kind="cache")
        self.max_length = max_length

    def _split_heads(self, x: Array) -> Array:
        """Reshape ``(..., seq, embed_dim)`` to ``(..., num_heads, seq, head_dim)``.

        Splits the trailing ``embed_dim`` axis into
        ``(num_heads, head_dim)`` and then swaps the new
        ``num_heads`` axis with the ``seq`` axis so head dimensions
        sit in front of the sequence — the layout expected by
        :func:`scaled_dot_product_attention`.

        Args:
            x: Tensor with shape ``(..., seq, embed_dim)``.

        Returns:
            ``(..., num_heads, seq, head_dim)``.
        """
        *batch, seq, _ = x.shape
        return x.reshape(*batch, seq, self.num_heads, self.head_dim).swapaxes(-2, -3)

    def _merge_heads(self, x: Array) -> Array:
        """Inverse of :meth:`_split_heads`.

        Args:
            x: Tensor with shape ``(..., num_heads, seq, head_dim)``.

        Returns:
            ``(..., seq, embed_dim)`` — the per-head outputs are
            concatenated along the trailing axis.
        """
        *batch, _, seq, _ = x.shape
        x = x.swapaxes(-2, -3)
        return x.reshape(*batch, seq, self.embed_dim)

    def forward(
        self,
        q: ArrayLike,
        k: ArrayLike | None = None,
        v: ArrayLike | None = None,
        *,
        mask: ArrayLike | None = None,
        is_causal: bool = False,
        rngs: Rngs | None = None,
        **_: object,
    ) -> Array:
        """Project Q/K/V, run scaled dot-product attention, merge, and project out.

        Sequence of operations:

        1. ``q``, ``k``, ``v`` are projected with the corresponding
           linears and split into per-head views via
           :meth:`_split_heads`.
        2. In ``decode=True`` mode the projected K/V are appended to
           the cache buffers; otherwise the projected tensors are
           used directly.
        3. :func:`scaled_dot_product_attention` produces the
           per-head context.
        4. :meth:`_merge_heads` collapses the head axis and
           :attr:`out_proj` projects back to ``embed_dim``.

        Args:
            q: Query tensor with shape ``(..., seq, embed_dim)``.
            k: Key tensor; defaults to ``q`` (self-attention).
            v: Value tensor; defaults to ``k``.
            mask: Optional attention mask broadcastable to
                ``(..., seq_q, seq_k)``. In decode mode it is combined
                (logical AND) with the auto-generated causal decode
                mask.
            is_causal: When ``True``, asks the attention primitive to
                apply a lower-triangular causal mask. Ignored in
                decode mode (the decode mask already enforces
                causality).
            rngs: :class:`Rngs` used to draw the dropout key. Required
                whenever :attr:`dropout` is positive *and*
                :attr:`training` is ``True``; otherwise may be ``None``.
            **_: Ignored; accepted for container interoperability.

        Returns:
            ``(..., seq, embed_dim)`` output tensor.

        Raises:
            RuntimeError: If ``decode=True`` was set on the module
                but :meth:`init_cache` has not been called.
        """
        if k is None:
            k = q
        if v is None:
            v = k
        qp = self._split_heads(self.q_proj(q))
        kp = self._split_heads(self.k_proj(k))
        vp = self._split_heads(self.v_proj(v))
        drop = self.dropout if self.training else 0.0
        key = rngs.key("dropout") if (drop > 0.0 and rngs is not None) else None

        if self.decode:
            if not hasattr(self, "cache_k"):
                raise RuntimeError("decode=True requires init_cache(batch_shape, max_length) first.")
            idx = self.cache_index.value
            step = kp.shape[-2]
            cache_dtype = self.cache_k.value.dtype
            ck = jax.lax.dynamic_update_slice_in_dim(
                self.cache_k.value,
                kp.astype(cache_dtype),
                idx,
                axis=-2,
            )
            cv = jax.lax.dynamic_update_slice_in_dim(
                self.cache_v.value,
                vp.astype(cache_dtype),
                idx,
                axis=-2,
            )
            self.cache_k.value = ck
            self.cache_v.value = cv
            self.cache_index.value = idx + step
            max_len = self.cache_k.value.shape[-2]
            positions = jnp.arange(max_len)
            q_pos = idx + jnp.arange(step)
            decode_mask = positions[None, :] <= q_pos[:, None]
            if mask is None:
                mask_in = decode_mask
            else:
                mask_in = jnp.logical_and(mask, decode_mask)
            out = scaled_dot_product_attention(
                qp,
                ck,
                cv,
                mask=mask_in,
                dropout_rate=drop,
                key=key,
                is_causal=False,
            )
        else:
            out = scaled_dot_product_attention(
                qp,
                kp,
                vp,
                mask=mask,
                dropout_rate=drop,
                key=key,
                is_causal=is_causal,
            )
        out = self._merge_heads(out)
        return cast(Array, self.out_proj(out))


class CausalSelfAttention(Module):
    """Self-attention layer that always applies the causal mask.

    Convenience wrapper around :class:`MultiheadAttention`: the
    embedded ``attn`` instance is configured identically and
    :meth:`forward` calls it with ``is_causal=True`` and ``q`` /
    ``k`` / ``v`` all set to the single positional input. Use this
    when the same module slot needs both shape uniformity (a
    one-argument forward) and the causal-attention guarantee.
    """

    attn: MultiheadAttention

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        qkv_sharding: Sharding | AxisNames | None = None,
        out_sharding: Sharding | AxisNames | None = None,
        qkv_bias_sharding: Sharding | AxisNames | None = None,
        out_bias_sharding: Sharding | AxisNames | None = None,
        cache_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the wrapped :class:`MultiheadAttention` instance.

        All keyword arguments are forwarded unchanged. See
        :meth:`MultiheadAttention.__init__` for their meaning.

        Args:
            embed_dim: Embed dim value consumed by this operation.
            num_heads: Num heads value consumed by this operation.
            dropout: Dropout value consumed by this operation.
            use_bias: Use bias value consumed by this operation.
            rngs: Random-number generator collection used to initialize or run the module.
            dtype: Array dtype requested for the produced value.
            param_dtype: Param dtype value consumed by this operation.
            qkv_sharding: Qkv sharding value consumed by this operation.
            out_sharding: Out sharding value consumed by this operation.
            qkv_bias_sharding: Qkv bias sharding value consumed by this operation.
            out_bias_sharding: Out bias sharding value consumed by this operation.
            cache_sharding: Cache sharding value consumed by this operation.
        """
        super().__init__()
        self.attn = MultiheadAttention(
            embed_dim,
            num_heads,
            use_bias=use_bias,
            dropout=dropout,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            qkv_sharding=qkv_sharding,
            out_sharding=out_sharding,
            qkv_bias_sharding=qkv_bias_sharding,
            out_bias_sharding=out_bias_sharding,
            cache_sharding=cache_sharding,
        )

    def forward(self, x: ArrayLike, *, rngs: Rngs | None = None, **_: object) -> Array:
        """Apply causal self-attention to ``x``.

        Args:
            x: Input tensor of shape ``(..., seq, embed_dim)``; used
                as both query and key/value.
            rngs: Optional :class:`Rngs`; forwarded to the wrapped
                attention. Required only when dropout is positive
                and the module is in training mode.
            **_: Ignored; accepted for container interoperability.

        Returns:
            ``(..., seq, embed_dim)`` output tensor with the lower
            triangular causal mask applied.
        """
        return cast(Array, self.attn(x, is_causal=True, rngs=rngs))
