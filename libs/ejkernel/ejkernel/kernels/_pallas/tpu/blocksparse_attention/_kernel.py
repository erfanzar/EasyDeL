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


"""Implementation of Sparse Flash Attention (Splash Attention) for TPU.

This module provides an optimized block-sparse attention implementation for
Google TPUs using the Pallas kernel framework. Splash Attention extends
Flash Attention to support arbitrary sparsity patterns defined by block masks.

Splash Attention achieves efficiency by:
1. Pre-computing which blocks have all-zeros, all-ones, or partial masks
2. Skipping computation for fully-masked blocks entirely
3. Using efficient Pallas kernels for partial blocks
4. Leveraging TPU MXU for dense block operations

Key features:
- Arbitrary block-sparse attention patterns via Mask objects
- Multi-head and multi-query attention support
- Segment IDs for packed sequence masking
- Attention sinks for StreamingLLM-style decoding
- Logits soft capping for numerical stability
- Custom VJP for efficient gradient computation

Architecture:
- Forward pass computes attention with block-level sparsity
- dQ/dKV backward passes handle gradient computation
- MaskInfo preprocessing identifies block types at trace time

Example:
    >>> from ejkernel.kernels._pallas.tpu.blocksparse_attention import (
    ...     splash_attention, BlockSizes, CausalMask
    ... )
    >>> mask = CausalMask((seq_len, seq_len))
    >>> output = splash_attention(
    ...     q, k, v, mask=mask,
    ...     block_sizes=BlockSizes.get_default(),
    ...     is_mqa=False
    ... )
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import typing
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple, Union, overload

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable
from jax import ad_checkpoint, lax, tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Bool, Float, Int

from ejkernel.callib import ejit
from ejkernel.ops import BwdParams, FwdParams

from . import _info as mask_info_lib
from . import _masks as mask_lib
from ._masks import (
    CausalMask,
    ChunkedCausalMask,
    FullMask,
    LocalMask,
    Mask,
    MultiHeadMask,
)

if typing.TYPE_CHECKING:
    from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask

partial = functools.partial
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8

NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))

# mypy: ignore-errors


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are a mechanism to ensure that there is no cross-attention between
    segments (fraction of a sequence) that have been concatenated together into a
    sequence. Each array is a list of ids (integers). Only tokens with the same
    id are allowed to attend to each other.

    The static mask (e.g. causal) is "and-ed" with the segment id mask to form
    the actual attention mask. It is important that the latter does not have any
    all-zero rows (along dimension kv). Otherwise it would result in a invalid
    softmax (the denominator would be 0).
    This condition holds for causal self-attention because in this case segment
    ids form a block diagonal matrix so at least one element in each row is set.
    It is easy to break this condition with non-self-attention configurations.
    Attributes:
      q: segment ids along the Q sequence
      kv: segment ids along the KV sequence
    """

    q: jax.Array
    kv: jax.Array


SplashCustomReturnType = Union[jax.Array, tuple[jax.Array, tuple[jax.Array,]]]  # noqa

SplashResidualsType = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    SegmentIds | None,
    jax.Array | None,
    jax.Array,
    jax.Array,
    mask_info_lib.MaskInfo | None,
    mask_info_lib.MaskInfo | None,
]

MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(
    block_metadata: Mapping[str, Any],
    is_mqa: bool,
    save_residuals: bool,
    is_segmented: bool,
    phase: str,
) -> str:
    """Returns a unique name for all SplashAttention kernel variants."""
    assert phase == "dq" or phase == "dkv" or phase == "fwd"

    assert not save_residuals or phase == "fwd"
    residuals = ""
    if save_residuals:
        residuals = "_residuals"
    elif phase == "fwd":
        residuals = "_no_residuals"
    attention_type = "mqa" if is_mqa else "mha"
    segments = "_segmented" if is_segmented else ""
    return f"splash_{attention_type}_{phase}{segments}{residuals}_" + "_".join(
        f"{k}={v}" for k, v in sorted(block_metadata.items())
    )


@overload
def _attention_reference(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: Literal[False],
    mask_value: float,
    custom_type: str,
    logits_soft_cap: float | None,
) -> jax.Array: ...


@overload
def _attention_reference(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: Literal[True],
    mask_value: float,
    custom_type: str,
    logits_soft_cap: float | None,
) -> tuple[jax.Array, tuple[jax.Array]]: ...


def _attention_reference(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    logits_soft_cap: float | None,
):
    return _attention_reference_default(
        mask,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value,
        save_residuals,
        custom_type,
        logits_soft_cap,
    )


def _attention_reference_default(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    logits_soft_cap: float | None,
):
    """Compute reference attention output using standard softmax formulation.

    Implements the default attention reference computation with support for
    segment masking, logits soft capping, and attention sinks. Used as a
    correctness baseline for the optimized Splash Attention kernel.

    Args:
        mask: Boolean attention mask of shape [q_seq_len, kv_seq_len].
        q: Query tensor of shape [q_seq_len, head_dim].
        k: Key tensor of shape [kv_seq_len, head_dim].
        v: Value tensor of shape [kv_seq_len, head_dim].
        segment_ids: Optional segment IDs for cross-segment masking.
        sinks: Optional attention sink values for StreamingLLM-style decoding.
        mask_value: Value to fill masked positions (typically large negative).
        save_residuals: If True, return log-sum-exp residuals for backward pass.
        custom_type: Type identifier for custom VJP dispatch (unused here).
        logits_soft_cap: Optional soft cap value for attention logits.

    Returns:
        If save_residuals is False, returns output tensor of shape [q_seq_len, head_dim].
        If save_residuals is True, returns (output, (log_sum_exp,)) tuple.
    """
    del custom_type
    logits = jnp.einsum("sd,td->st", q.astype(jnp.float32), k.astype(jnp.float32))

    if segment_ids is not None:
        mask = jnp.logical_and(mask, segment_ids.q[:, None] == segment_ids.kv[None, :])

    if logits_soft_cap is not None:
        logits = jnp.tanh(logits / logits_soft_cap)
        logits = logits * logits_soft_cap

    logits = jnp.where(mask, logits, mask_value)
    m = logits.max(axis=-1)
    sinks = None if sinks is None else sinks.astype(logits.dtype)
    m = m if sinks is None else jnp.maximum(m, sinks)
    s = jnp.exp(logits - m[..., None])
    l = s.sum(axis=-1) + (0 if sinks is None else jnp.exp(sinks - m))
    s = s / l[..., None]

    o = jnp.einsum("st,td->sd", s, v.astype(jnp.float32))

    lse = m + jnp.log(l)
    if save_residuals:
        return o, (lse,)
    return o


def attention_reference(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None = None,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    save_residuals: bool = False,
    custom_type: str = "flash",
    logits_soft_cap: float | None = None,
) -> SplashCustomReturnType:
    """Compute reference attention with default (non-custom-VJP) backward pass.

    Public entry point for computing attention using the standard JAX autodiff
    backward pass. For custom VJP backward, use ``attention_reference_custom``.

    Args:
        mask: Boolean attention mask of shape [q_seq_len, kv_seq_len].
        q: Query tensor of shape [q_seq_len, head_dim].
        k: Key tensor of shape [kv_seq_len, head_dim].
        v: Value tensor of shape [kv_seq_len, head_dim].
        segment_ids: Optional segment IDs for packed sequence masking.
        sinks: Optional attention sink values.
        mask_value: Fill value for masked positions. Defaults to DEFAULT_MASK_VALUE.
        save_residuals: Whether to return log-sum-exp for backward pass.
        custom_type: VJP computation type ("flash" or "vanilla").
        logits_soft_cap: Optional soft cap for attention logits.

    Returns:
        Attention output, or (output, (log_sum_exp,)) if save_residuals is True.
    """
    return _attention_reference(
        mask,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value=mask_value,
        save_residuals=save_residuals,
        custom_type=custom_type,
        logits_soft_cap=logits_soft_cap,
    )


def _attention_reference_custom_fwd(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    logits_soft_cap: float | None,
):
    """Forward pass for custom VJP attention reference.

    Computes attention output and saves residuals needed for the custom
    backward pass. Higher-order autodiff is not supported.

    Args:
        mask: Boolean attention mask of shape [q_seq_len, kv_seq_len].
        q: Query tensor of shape [q_seq_len, head_dim].
        k: Key tensor of shape [kv_seq_len, head_dim].
        v: Value tensor of shape [kv_seq_len, head_dim].
        segment_ids: Optional segment IDs for packed sequence masking.
        sinks: Optional attention sink values.
        mask_value: Fill value for masked positions.
        save_residuals: Must be False (higher-order AD not supported).
        custom_type: VJP computation type ("flash" or "vanilla").
        logits_soft_cap: Optional soft cap for attention logits.

    Returns:
        Tuple of (output, residuals) where residuals contains all tensors
        needed for the backward pass.

    Raises:
        NotImplementedError: If save_residuals is True.
    """
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported.")

    o, (lse,) = _attention_reference(
        mask,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value=mask_value,
        save_residuals=True,
        custom_type=custom_type,
        logits_soft_cap=logits_soft_cap,
    )
    return o, (mask, q, k, v, segment_ids, sinks, o, lse)


def _attention_reference_custom_bwd(
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    logits_soft_cap: float | None,
    res,
    do: jax.Array,
) -> tuple[None, jax.Array, jax.Array, jax.Array, None, jax.Array | None]:
    """Backward pass for custom VJP attention reference.

    Computes gradients for query, key, value, and optional sinks using
    saved residuals from the forward pass. Supports both "flash" and
    "vanilla" gradient computation styles, as well as logits soft capping.

    Args:
        mask_value: Fill value for masked positions (non-differentiable).
        save_residuals: Whether residuals were saved (non-differentiable).
        custom_type: VJP type - "flash" uses output-based correction,
            "vanilla" uses direct softmax Jacobian (non-differentiable).
        logits_soft_cap: Optional soft cap value (non-differentiable).
        res: Residuals tuple from forward pass containing
            (mask, q, k, v, segment_ids, sinks, output, log_sum_exp).
        do: Gradient of the output tensor.

    Returns:
        Tuple of gradients (None, dq, dk, dv, None, dsinks) corresponding
        to (mask, q, k, v, segment_ids, sinks) inputs.
    """
    del save_residuals
    mask, q, k, v, segment_ids, sinks, o, lse = res

    uncapped_logits = jnp.einsum("qc,kc->qk", q, k, preferred_element_type=jnp.float32)

    if logits_soft_cap is not None:
        logits = jnp.tanh(uncapped_logits / logits_soft_cap)
        logits = logits * logits_soft_cap
    else:
        logits = uncapped_logits

    if segment_ids is not None:
        mask = jnp.logical_and(mask, segment_ids.q[:, None] == segment_ids.kv[None, :])
    logits = jnp.where(mask, logits, mask_value)

    p = jnp.exp(logits - lse[..., None])
    do = do.astype(jnp.float32)
    dv = jnp.einsum("pt,pd->td", p, do).astype(v.dtype)
    dp = jnp.einsum("pd,td->pt", do, v.astype(jnp.float32))

    if custom_type == "flash":
        di = jnp.sum(o.astype(jnp.float32) * do, axis=-1)[..., None]
    else:
        di = jnp.einsum("st,st->s", dp, p)[:, None]
    ds = (dp - di) * p
    if logits_soft_cap is not None:
        normalized = uncapped_logits / logits_soft_cap
        d = jnp.tanh(normalized)
        g = ds * (1 - d)
        ds = g + g * d
    dk = jnp.einsum("sd,st->td", q.astype(jnp.float32), ds).astype(k.dtype)
    dq = jnp.einsum("st,td->sd", ds, k.astype(jnp.float32)).astype(q.dtype)
    dsinks = None
    if sinks is not None:
        sinks_exp = -jnp.exp(sinks[..., None, None].astype(jnp.float32) - lse[..., None].astype(jnp.float32))
        dsinks = jnp.sum(sinks_exp.astype(o.dtype) * do * o)
    return None, dq, dk, dv, None, dsinks


_attention_reference_custom = jax.custom_vjp(
    _attention_reference, nondiff_argnames=("mask_value", "save_residuals", "custom_type", "logits_soft_cap")
)
_attention_reference_custom.defvjp(_attention_reference_custom_fwd, _attention_reference_custom_bwd)


def attention_reference_custom(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None = None,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    save_residuals: bool = False,
    custom_type: str = "flash",
    logits_soft_cap: float | None = None,
):
    """Compute reference attention with custom VJP backward pass.

    Uses a custom VJP rule for more efficient gradient computation compared
    to standard JAX autodiff. The "flash" type uses output-based gradient
    correction (FlashAttention-style), while "vanilla" uses direct computation.

    Args:
        mask: Boolean attention mask of shape [q_seq_len, kv_seq_len].
        q: Query tensor of shape [q_seq_len, head_dim].
        k: Key tensor of shape [kv_seq_len, head_dim].
        v: Value tensor of shape [kv_seq_len, head_dim].
        segment_ids: Optional segment IDs for packed sequence masking.
        sinks: Optional attention sink values.
        mask_value: Fill value for masked positions. Defaults to DEFAULT_MASK_VALUE.
        save_residuals: Whether to return log-sum-exp residuals.
        custom_type: Backward computation type ("flash" or "vanilla").
        logits_soft_cap: Optional soft cap for attention logits.

    Returns:
        Attention output, or (output, (log_sum_exp,)) if save_residuals is True.
    """
    return _attention_reference_custom(
        mask,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value,
        save_residuals,
        custom_type=custom_type,
        logits_soft_cap=logits_soft_cap,
    )


def make_attention_reference(
    mask: mask_lib.Mask | np.ndarray,
    is_mqa: bool,
    backward_impl: str = "vanilla",
    **params: Any,
) -> Callable:
    """Create a JIT-compiled reference attention function with the given mask.

    Factory function that constructs a batched attention computation function
    with the mask baked in. Supports MHA, MQA, and GQA configurations with
    optional custom backward passes.

    Args:
        mask: Attention mask as a Mask object or numpy array of shape
            [num_heads, q_seq_len, kv_seq_len].
        is_mqa: If True, use multi-query attention (single KV head shared
            across all query heads). If False, use multi-head attention.
        backward_impl: Backward pass implementation to use. Options:
            "vanilla" - standard JAX autodiff,
            "custom" - custom VJP with flash-style correction,
            "custom_vanilla" - custom VJP with vanilla correction.
        **params: Additional keyword arguments passed to the attention function.

    Returns:
        A JIT-compiled callable that computes attention given (q, k, v, ...).
    """

    @partial(
        jax.jit,
        static_argnames=[
            "mask_value",
            "save_residuals",
            "logits_soft_cap",
        ],
    )
    def _wrapped(
        mask: jax.Array,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        segment_ids: SegmentIds | None = None,
        sinks: jax.Array | None = None,
        *,
        mask_value: float = DEFAULT_MASK_VALUE,
        save_residuals: bool = False,
        logits_soft_cap: float | None = None,
    ):
        if backward_impl == "custom":
            attn_impl = partial(
                attention_reference_custom,
                custom_type="flash",
            )
        elif backward_impl == "custom_vanilla":
            attn_impl = partial(
                attention_reference_custom,
                custom_type="vanilla",
            )
        else:
            attn_impl = attention_reference
        func = partial(
            attn_impl,
            mask_value=mask_value,
            save_residuals=save_residuals,
            logits_soft_cap=logits_soft_cap,
            **params,
        )

        if is_mqa:
            func = jax.vmap(func, in_axes=(0, 0, None, None, None, 0))
            is_grouped = False
        else:
            kv_heads = k.shape[0]
            assert kv_heads == v.shape[0]
            q_heads, q_seq_len, head_dim = q.shape
            is_grouped = kv_heads < q_heads
            if is_grouped:
                assert q_heads % kv_heads == 0
                assert mask.shape[0] == q_heads
                q_heads_per_kv_head = q_heads // kv_heads
                q = q.reshape((kv_heads, q_heads_per_kv_head, q_seq_len, head_dim))
                mask = mask.reshape((kv_heads, q_heads_per_kv_head, *mask.shape[1:]))
                if sinks is not None:
                    sinks = sinks.reshape((kv_heads, q_heads_per_kv_head))

                func = jax.vmap(func, in_axes=(0, 0, None, None, None, 0))

            func = jax.vmap(func, in_axes=(0, 0, 0, 0, None, 0))

        out = func(mask, q, k, v, segment_ids, sinks)

        if is_grouped:

            def reshape_activations(activations):
                if activations.ndim == 4:
                    kv_heads, q_heads_per_kv_head, q_seq_len, head_dim = activations.shape
                    return activations.reshape(kv_heads * q_heads_per_kv_head, q_seq_len, head_dim)
                return activations

            def reshape_residuals(residuals):
                if residuals.ndim == 3:
                    kv_heads, q_heads_per_kv_head, q_seq_len = residuals.shape
                    return residuals.reshape(kv_heads * q_heads_per_kv_head, q_seq_len)
                return residuals

            if save_residuals:
                assert isinstance(out, tuple)
                assert isinstance(out[1], tuple)

                return (reshape_activations(out[0]), (reshape_residuals(out[1][0]),))
            else:
                return reshape_activations(out)
        else:
            return out

    return functools.partial(_wrapped, jnp.array(mask[:, :, :]))


make_masked_mha_reference = partial(make_attention_reference, is_mqa=False)
make_masked_mqa_reference = partial(make_attention_reference, is_mqa=True)


class QKVLayout(enum.IntEnum):
    """Physical memory layout for Q, K, V tensors in Splash Attention.

    Controls how the head and sequence dimensions are arranged in memory.
    HEAD_DIM_MINOR places head_dim as the innermost (fastest-varying) dimension,
    while SEQ_MINOR places the sequence dimension as innermost.

    Attributes:
        HEAD_DIM_MINOR: Layout with shape (..., seq_len, head_dim).
        SEQ_MINOR: Layout with shape (..., head_dim, seq_len).
    """

    HEAD_DIM_MINOR = enum.auto()
    SEQ_MINOR = enum.auto()


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
    """Convert index tuple from HEAD_DIM_MINOR layout to the target layout.

    For HEAD_DIM_MINOR, indices are returned as-is. For SEQ_MINOR, the
    last two indices are swapped to match the transposed memory layout.

    Args:
        vals: Tuple of indices in HEAD_DIM_MINOR order.
        layout: Target QKVLayout to convert to.

    Returns:
        Index tuple reordered for the target layout.
    """
    if layout == QKVLayout.HEAD_DIM_MINOR:
        return vals
    return (*vals[:-2], vals[-1], vals[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
    """Tile sizes parameterizing SplashAttention kernels.

    Those parameters have negligible effect on numerics, but affect performance
    greatly.

    Note that changing the layouts only influences the physical layout that the
    kernel will enforce. The logical interface to blocksparse_attention attention always takes
    the head dimension as the minormost one.
    """

    block_q: int
    block_kv: int
    block_kv_compute: int | None = None

    block_q_dkv: int | None = None
    block_kv_dkv: int | None = None
    block_kv_dkv_compute: int | None = None

    block_q_dq: int | None = None
    block_kv_dq: int | None = None

    use_fused_bwd_kernel: bool = False

    q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
    k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
    v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

    def __post_init__(self):
        if self.block_kv_compute is None:
            object.__setattr__(self, "block_kv_compute", self.block_kv)
        if self.block_kv_dkv_compute is None:
            object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)
        if self.use_fused_bwd_kernel:
            if self.block_q_dq is not None or self.block_kv_dq is not None:
                raise ValueError("Block sizes for dq kernel are not needed with a fused kernel.")

    @property
    def has_backward_blocks(self) -> bool:
        backward_blocks = (
            self.block_q_dkv,
            self.block_kv_dkv,
            self.block_kv_dkv_compute,
        )
        if not self.use_fused_bwd_kernel:
            backward_blocks += (self.block_q_dq, self.block_kv_dq)
        return all(b is not None for b in backward_blocks)

    @classmethod
    def get_default(cls):
        return BlockSizes(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            block_q_dq=128,
            block_kv_dq=128,
        )


def _next_nonzero(
    h,
    i,
    j,
    data_next_ref,
    block_mask_ref,
    m_next_ref,
    next_i=False,
):
    """Look up the next nonzero block index from the sparse mask metadata.

    Reads block_mask, data_next, and mask_next arrays stored in TPU scalar
    memory to determine the next KV block to process and whether masking
    is needed for the current block.

    Args:
        h: Head index in the computation grid.
        i: Query block index in the computation grid.
        j: KV block index in the computation grid.
        data_next_ref: Reference to data_next array for block prefetch indices.
        block_mask_ref: Reference to block_mask array (0=empty, 1=partial, 2=full).
        m_next_ref: Reference to mask_next array for partial mask block indices.
        next_i: If True, return query index instead of KV index for dKV kernels.

    Returns:
        Tuple of (next_data_index, next_mask_index, is_nonzero, should_not_mask):
            - next_data_index: Index of the next KV (or Q) block to process.
            - next_mask_index: Index into partial_mask_blocks, or None.
            - is_nonzero: Whether the current block has any nonzero entries.
            - should_not_mask: Whether the block is fully unmasked (all ones).
    """
    assert (data_next_ref is None) == (block_mask_ref is None)

    if data_next_ref is None and block_mask_ref is None:
        assert m_next_ref is None
        next_data = i if next_i else j
        return (
            next_data,
            None,
            True,
            False,
        )

    assert data_next_ref.shape == block_mask_ref.shape
    assert m_next_ref is None or data_next_ref.shape[0] == m_next_ref.shape[0]

    if data_next_ref.shape[0] == 1:
        h = 0

    def to_i32(x):
        return x.astype(jnp.int32)

    is_nonzero = to_i32(block_mask_ref[h, i, j]) > 0
    if m_next_ref is None:
        should_not_mask = True
        next_m = None
    else:
        should_not_mask = to_i32(block_mask_ref[h, i, j]) != 1
        next_m = to_i32(m_next_ref[h, i, j])
    next_j = to_i32(data_next_ref[h, i, j])
    return next_j, next_m, is_nonzero, should_not_mask


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    should_not_mask,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    logits_soft_cap: float,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes=True,
    mask_function=None,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply attention masking and optional logits soft capping to QK scores.

    Combines multiple masking sources (block mask, computed mask function,
    segment IDs) and applies them to the raw attention logits. Optionally
    applies tanh-based soft capping to prevent extreme logit values.

    Args:
        qk: Raw attention logits of shape [bq, bkv] or [bkv, bq].
        mask_value: Value to fill masked positions (large negative float).
        should_not_mask: Boolean indicating the block is fully unmasked.
        mask_ref: Reference to partial mask block, or None if not needed.
        q_sequence_ref: Reference to query sequence indices for computed masks.
        q_segment_ids_ref: Reference to query segment IDs, or None.
        kv_segment_ids_ref: Reference to KV segment IDs, or None.
        logits_soft_cap: Soft cap value for logits, or None to disable.
        k_slice: Pallas slice selecting the current KV block.
        k_offset: Global offset of the current KV block.
        bq: Query block size.
        k_in_lanes: If True, KV dimension is in lanes (minor axis).
        mask_function: Optional callable for computing mask on-the-fly.

    Returns:
        Masked (and optionally soft-capped) attention logits.
    """
    assert mask_ref is None or q_sequence_ref is None
    assert (q_sequence_ref is None) == (mask_function is None)

    masks = []
    if mask_ref is not None:
        if k_in_lanes:
            mask = mask_ref[:, k_slice]
        else:
            mask = mask_ref[k_slice, :]

        masks.append(jnp.bitwise_or(mask, jnp.broadcast_to(should_not_mask, mask.shape)))
    if mask_function is not None:
        if k_in_lanes:
            assert q_sequence_ref.shape == (bq, NUM_LANES)

            k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (bq, k_slice.size), 1)

            repeats, rem = divmod(k_slice.size, NUM_LANES)
            assert rem == 0
            q_sequence = pltpu.repeat(q_sequence_ref[...], repeats, axis=1)
        else:
            assert q_sequence_ref.shape == (NUM_SUBLANES, bq)

            k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (k_slice.size, bq), 0)
            q_sequence = q_sequence_ref[:1, :]
            q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

        assert q_sequence.shape == k_sequence.shape
        computed_mask = mask_function(q_sequence, k_sequence)
        if computed_mask.dtype != jnp.dtype(jnp.bool_):
            raise ValueError(f"Mask function must return a boolean-valued array, but got: {computed_mask.dtype}")
        masks.append(computed_mask)

    if q_segment_ids_ref is not None:
        if k_in_lanes:
            kv_ids = kv_segment_ids_ref[:1, k_slice]
            repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
            if rem:
                raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
            q_ids = pltpu.repeat(q_segment_ids_ref[:], repeats, axis=1)
        else:
            assert bq == q_segment_ids_ref.shape[-1]
            repeats, rem = divmod(bq, NUM_LANES)
            if rem:
                raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
            kv_ids = pltpu.repeat(kv_segment_ids_ref[k_slice, :], repeats, axis=1)
            q_ids = q_segment_ids_ref[:1, :]
        masks.append(q_ids == kv_ids)

    def cap_logits(logits):
        if logits_soft_cap is not None:
            logits = jnp.tanh(qk / logits_soft_cap)
            return logits * logits_soft_cap
        else:
            return logits

    if masks:
        mask = functools.reduce(jnp.logical_and, masks)
        qk = cap_logits(qk)
        qk = jnp.where(mask, qk, mask_value)
    else:
        qk = cap_logits(qk)
    return qk


def flash_attention_kernel(
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    mask_ref,
    q_sequence_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    o_ref,
    logsumexp_ref=None,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    logits_soft_cap: float | None,
    mask_function: MaskFunctionType | None,
):
    float32 = jnp.float32
    HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
    head_dim_v_repeats = pl.cdiv(head_dim_v, NUM_LANES)

    h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    @pl.when(j == 0)
    def init():
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
        if sinks_ref is not None:
            sinks = sinks_ref[h].astype(m_scratch_ref.dtype)

            m_scratch_ref[...] = sinks * jnp.ones_like(m_scratch_ref)
            l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
        else:
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
            l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

    global_kv_index, _, should_run, should_not_mask = _next_nonzero(
        h,
        i,
        j,
        data_next_ref,
        block_mask_ref,
        mask_next_ref,
    )

    def body(kv_compute_index, _):
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
        assert m_prev.shape == (bq, NUM_LANES)
        assert l_prev.shape == (bq, NUM_LANES)

        q = q_ref[...] if q_layout == HEAD_DIM_MINOR else q_ref[...].T
        qk_dims = NT_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
        if k_layout == HEAD_DIM_MINOR:
            k = k_ref[slice_k, :]
        else:
            k = k_ref[:, slice_k]
        qk = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

        assert qk.shape == (bq, bkv_compute)
        apply_mask_and_soft_cap = functools.partial(
            _apply_mask_and_soft_cap,
            qk,
            mask_value,
            should_not_mask,
            mask_ref,
            q_sequence_ref,
            q_segment_ids_ref,
            kv_segment_ids_ref,
            logits_soft_cap=logits_soft_cap,
            k_slice=slice_k,
            k_offset=global_kv_index * bkv + kv_compute_index * bkv_compute,
            bq=bq,
            mask_function=mask_function,
        )

        qk = apply_mask_and_soft_cap()

        m_curr = qk.max(axis=-1)[:, None]
        assert m_curr.shape == (bq, 1)
        m_next = jnp.maximum(m_prev, m_curr)
        assert m_next.shape == (bq, NUM_LANES)

        bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
        if rem != 0:
            raise NotImplementedError(f"{bkv_compute=} should be a multiple of {NUM_LANES}")

        s_curr = jnp.exp(qk - pltpu.repeat(m_next, bkv_repeats, axis=1))
        assert s_curr.shape == (bq, bkv_compute)

        l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
        assert l_curr.shape == (bq, NUM_LANES)

        alpha = jnp.exp(m_prev - m_next)
        l_next = l_curr + alpha * l_prev
        m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next

        sv_dims = NN_DIM_NUMBERS if v_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
        if v_layout == HEAD_DIM_MINOR:
            v = v_ref[slice_k, :]
        else:
            v = v_ref[:, slice_k]
        v = v.astype(float32)
        o_curr = lax.dot_general(s_curr, v, sv_dims)

        alpha_o = pltpu.repeat(alpha, head_dim_v_repeats, axis=1)[..., : o_scratch_ref.shape[-1]]
        o_scratch_ref[:] = alpha_o * o_scratch_ref[:] + o_curr

    @pl.when(should_run)
    def run():
        assert bkv % bkv_compute == 0
        num_iters = k_ref.shape[0 if k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
        lax.fori_loop(0, num_iters, body, None, unroll=True)

    @pl.when(j == grid_width - 1)
    def end():
        l = l_scratch_ref[...]
        l_inv = pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=1)[..., : o_scratch_ref.shape[-1]]
        o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
        if logsumexp_ref is not None:
            assert logsumexp_ref.shape == (bq, NUM_LANES)
            logsumexp_ref[...] = (jnp.log(l) + m_scratch_ref[...]).astype(logsumexp_ref.dtype)

        m_scratch_ref[...] = jnp.zeros_like(m_scratch_ref)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)


@overload
def _splash_attention_forward(
    fwd_mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    save_residuals: Literal[False] = False,
    logits_soft_cap: float | None = None,
) -> jax.Array: ...


@overload
def _splash_attention_forward(
    fwd_mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    save_residuals: Literal[True],
    logits_soft_cap: float | None = None,
) -> SplashCustomReturnType: ...


def _div(dividend: int, divisor: int):
    if divisor == 1:
        return dividend

    return lax.div(dividend, divisor)


def _splash_attention_forward(
    fwd_mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None = None,
    interpret: bool = False,
) -> SplashCustomReturnType:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = v.shape[-1]
    bq, bkv = block_sizes.block_q, block_sizes.block_kv
    bkv_compute = block_sizes.block_kv_compute

    if is_mqa:
        expected_kv_rank = 2
        kv_head_dimension = 1
        kv_seq_len_dimension = 0
        num_kv_heads = 1
    else:
        expected_kv_rank = 3
        kv_head_dimension = 2
        kv_seq_len_dimension = 1
        num_kv_heads = k.shape[0]

    partial_mask_blocks = fwd_mask_info.partial_mask_blocks
    if partial_mask_blocks is not None and jnp.dtype(partial_mask_blocks.dtype) != np.bool_:
        raise ValueError(f"partial_mask_blocks must be of type np.bool_ but got {partial_mask_blocks.dtype}")

    if len(k.shape) != expected_kv_rank:
        raise ValueError(f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a {len(k.shape)}-dim one.")

    if k.shape[kv_head_dimension] != head_dim_qk:
        raise ValueError(
            f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got: {k.shape[kv_head_dimension]}."
        )

    if not is_mqa and num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
            f" multiple of the number of 'query' heads ({num_q_heads})"
        )

    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError(f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same leading dimensions.")

    if bkv % bkv_compute:
        raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
    if bkv_compute % NUM_LANES:
        raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

    kv_seq_len = k.shape[kv_seq_len_dimension]

    q_heads_per_kv_head = num_q_heads // num_kv_heads

    if segment_ids is not None:
        if segment_ids.q.shape != (q_seq_len,):
            raise ValueError(f"Invalid shape for q segment_ids: {segment_ids.q.shape}. Expected: {(q_seq_len,)}")
        if segment_ids.kv.shape != (kv_seq_len,):
            raise ValueError(f"Invalid shape for kv segment_ids: {segment_ids.kv.shape}. Expected: {(kv_seq_len,)}")

    q_layout = block_sizes.q_layout

    def q_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
        del j, data_next_ref, mask_next_ref, block_mask_ref
        return from_head_minor((h, i, 0), q_layout)

    def out_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
        del j, data_next_ref, mask_next_ref, block_mask_ref
        return h, i, 0

    k_layout = block_sizes.k_layout

    def k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
        next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
        return from_head_minor((*prefix, next_j, 0), k_layout)

    v_layout = block_sizes.v_layout

    def v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
        next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
        return from_head_minor((*prefix, next_j, 0), v_layout)

    def mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
        _, next_m, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        return next_m, 0, 0

    def q_segment_ids_index_map(h, i, j, *_):
        del h, j
        return i, 0

    def kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
        next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        return 0, next_j

    in_specs = [
        pl.BlockSpec(from_head_minor((None, bq, head_dim_qk), q_layout), q_index_map),
        pl.BlockSpec(
            from_head_minor((bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk), k_layout),
            k_index_map,
        ),
        pl.BlockSpec(
            from_head_minor((bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v), v_layout),
            v_index_map,
        ),
    ]
    if segment_ids is not None:
        in_specs += [
            pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map),
            pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map),
        ]
        q_segment_ids = jax.lax.broadcast_in_dim(segment_ids.q, (q_seq_len, NUM_LANES), (0,))
        kv_segment_ids = jax.lax.broadcast_in_dim(segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,))
    else:
        in_specs += [None, None]
        q_segment_ids = kv_segment_ids = None

    if sinks is not None:
        assert sinks.shape == (num_q_heads,)
        in_specs += [pl.BlockSpec((num_q_heads,), lambda h, i, j, *_: (0,), memory_space=pltpu.SMEM)]
        sinks = sinks.astype(jnp.float32)
    else:
        in_specs += [None]

    if fwd_mask_info.partial_mask_blocks is not None:
        in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
    else:
        in_specs.append(None)

    assert fwd_mask_info.partial_mask_blocks is None or fwd_mask_info.q_sequence is None

    if fwd_mask_info.q_sequence is not None:
        q_sequence = jax.lax.broadcast_in_dim(fwd_mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,))
        in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
    else:
        q_sequence = None
        in_specs.append(None)

    num_scalar_prefetch = 3

    out_shapes = [
        jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),
        jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),
        jax.ShapeDtypeStruct((bq, head_dim_v), jnp.float32),
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), q.dtype),
    ]
    out_specs = [
        pl.BlockSpec((bq, NUM_LANES), lambda h, i, j, *_: (0, 0)),
        pl.BlockSpec((bq, NUM_LANES), lambda h, i, j, *_: (0, 0)),
        pl.BlockSpec((bq, head_dim_v), lambda h, i, j, *_: (0, 0)),
        pl.BlockSpec((None, bq, head_dim_v), out_index_map),
    ]
    if save_residuals:
        out_shapes += [
            jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32),
        ]

        def logsumexp_index_map(h, i, *_):
            return h, i, 0

        out_specs += [
            pl.BlockSpec((None, bq, NUM_LANES), logsumexp_index_map),
        ]
    else:
        out_shapes += [None]
        out_specs += [None]

    kernel_name = get_kernel_name(
        dataclasses.asdict(block_sizes),
        is_mqa=is_mqa,
        save_residuals=save_residuals,
        is_segmented=segment_ids is not None,
        phase="fwd",
    )

    if fwd_mask_info.data_next is not None:
        grid_width = fwd_mask_info.data_next.shape[-1]
    else:
        grid_width = kv_seq_len // bkv

    grid = (num_q_heads, q_seq_len // bq, grid_width)
    with jax.named_scope(kernel_name):
        all_out = pl.pallas_call(
            partial(
                flash_attention_kernel,
                mask_value=mask_value,
                grid_width=grid_width,
                bq=bq,
                bkv=bkv,
                bkv_compute=bkv_compute,
                head_dim_v=head_dim_v,
                q_layout=q_layout,
                k_layout=k_layout,
                v_layout=v_layout,
                logits_soft_cap=logits_soft_cap,
                mask_function=mask_function,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=num_scalar_prefetch,
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            ),
            out_shape=out_shapes,
            name=kernel_name,
            interpret=interpret,
        )(
            fwd_mask_info.data_next,
            fwd_mask_info.block_mask,
            fwd_mask_info.mask_next,
            q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
            k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
            v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.swapaxes(-1, -2),
            q_segment_ids,
            kv_segment_ids,
            sinks,
            fwd_mask_info.partial_mask_blocks,
            q_sequence,
        )

    (
        _,
        _,
        _,
        out,
        lse,
    ) = all_out

    if save_residuals:
        assert lse is not None
        lse = lse[..., 0]

    if residual_checkpoint_name is not None:
        out = ad_checkpoint.checkpoint_name(out, name=residual_checkpoint_name)
        if lse is not None:
            lse = ad_checkpoint.checkpoint_name(lse, name=residual_checkpoint_name)
    if save_residuals:
        return out, (lse,)
    return out


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "save_residuals",
        "mask_value",
        "is_mqa",
        "block_sizes",
        "residual_checkpoint_name",
        "mask_function",
        "logits_soft_cap",
        "interpret",
    ),
)
def _splash_attention_custom(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None = None,
    interpret: bool = False,
) -> SplashCustomReturnType:
    del dq_mask_info, dkv_mask_info

    return _splash_attention_forward(
        fwd_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks=sinks,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        residual_checkpoint_name=residual_checkpoint_name,
        save_residuals=save_residuals,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        interpret=interpret,
    )


def _splash_attention_fwd(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None = None,
    interpret: bool = False,
) -> tuple[
    tuple[jax.Array],
    SplashResidualsType,
]:
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")

    out, (lse,) = _splash_attention_forward(
        fwd_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        residual_checkpoint_name=residual_checkpoint_name,
        save_residuals=True,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        interpret=interpret,
    )
    return out, (
        q,
        k,
        v,
        segment_ids,
        sinks,
        out,
        lse,
        dq_mask_info,
        dkv_mask_info,
    )


def _flash_attention_dq_kernel(
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    dq_scratch_ref,
    dq_ref,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    logits_soft_cap: float | None = None,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    mask_function: MaskFunctionType | None,
):
    del sinks_ref
    float32 = jnp.float32
    HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR

    h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    @pl.when(j == 0)
    def init():
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

    global_kv_index, _, should_run, should_not_mask = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )

    @pl.when(should_run)
    def run():
        q = q_ref[...] if q_layout == HEAD_DIM_MINOR else q_ref[...].T

        k = k_ref[...]
        v = v_ref[...]
        lse = jnp.expand_dims(logsumexp_ref[0], -1)
        do = do_ref[...]
        di = jnp.expand_dims(di_ref[0], -1)

        qk_dims = NT_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
        qk_uncapped = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

        qk = _apply_mask_and_soft_cap(
            qk_uncapped,
            mask_value,
            should_not_mask,
            mask_ref,
            q_sequence_ref,
            q_segment_ids_ref,
            kv_segment_ids_ref,
            logits_soft_cap=logits_soft_cap,
            k_slice=pl.ds(0, bkv),
            k_offset=global_kv_index * bkv,
            bq=bq,
            mask_function=mask_function,
        )
        p = jnp.exp(qk - lse)
        dp_dims = NT_DIM_NUMBERS if v_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
        dp = lax.dot_general(
            do.astype(v.dtype),
            v,
            dp_dims,
            preferred_element_type=jnp.float32,
        )
        ds = (dp - di) * p
        if logits_soft_cap is not None:
            normalized = qk_uncapped / logits_soft_cap
            d = jnp.tanh(normalized)
            g = ds * (1 - d)
            ds = g + g * d

        dq_dims = NN_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
        dq_scratch_ref[...] += lax.dot_general(
            ds.astype(k.dtype),
            k,
            dq_dims,
            preferred_element_type=jnp.float32,
        )

    @pl.when(j == grid_width - 1)
    def end():
        dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _splash_attention_bwd_dq(
    q,
    k,
    v,
    segment_ids,
    sinks,
    lse,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    is_mqa: bool,
    mask_info: mask_info_lib.MaskInfo,
    mask_value: float,
    logits_soft_cap: float | None,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    mask_function: MaskFunctionType | None,
    interpret: bool,
):
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = v.shape[-1]
    if is_mqa:
        kv_seq_len = k.shape[0]
        num_kv_heads = 1
    else:
        kv_seq_len = k.shape[1]
        num_kv_heads = k.shape[0]

    if bq > q_seq_len:
        raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
    if bkv > kv_seq_len:
        raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")

    if not is_mqa and num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
            f" multiple of the number of 'query' heads ({num_q_heads})"
        )

    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError(f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same leading dimensions.")

    if bkv % NUM_LANES:
        raise ValueError(f"{bkv=} must be a multiple of {NUM_LANES}.")

    q_heads_per_kv_head = num_q_heads // num_kv_heads

    if mask_info.data_next is not None:
        grid_width = mask_info.data_next.shape[-1]
    else:
        grid_width = kv_seq_len // bkv

    grid = (num_q_heads, q_seq_len // bq, grid_width)

    def o_index_map(h, i, *_):
        return h, i, 0

    o_spec = pl.BlockSpec((None, bq, head_dim_v), o_index_map)

    def q_index_map(h, i, *_):
        return from_head_minor((h, i, 0), q_layout)

    q_spec = pl.BlockSpec(from_head_minor((None, bq, head_dim_qk), q_layout), q_index_map)

    def k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_):
        next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
        return from_head_minor((*prefix, next_j, 0), k_layout)

    k_spec = pl.BlockSpec(
        from_head_minor((bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk), k_layout),
        k_index_map,
    )

    def v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_):
        next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
        return from_head_minor((*prefix, next_j, 0), v_layout)

    v_spec = pl.BlockSpec(
        from_head_minor((bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v), v_layout),
        v_index_map,
    )

    def mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_):
        _, next_m, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        return next_m, 0, 0

    mask_spec = pl.BlockSpec((None, bq, bkv), mask_index_map)

    def q_segment_ids_index_map(h, i, j, *_):
        del h, j
        return i, 0

    if segment_ids is not None:

        def kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_):
            next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
            return 0, next_j

        q_segment_spec = pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map)
        kv_segment_spec = pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map)
        q_segment_ids = jax.lax.broadcast_in_dim(segment_ids.q, (q_seq_len, NUM_LANES), (0,))
        kv_segment_ids = jax.lax.broadcast_in_dim(segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,))
    else:
        q_segment_spec = kv_segment_spec = None
        q_segment_ids = kv_segment_ids = None

    if sinks is not None:
        assert sinks.shape == (num_q_heads,)
        sinks_spec = pl.BlockSpec((num_q_heads,), lambda h, *_: (0,), memory_space=pltpu.SMEM)
        sinks = sinks.astype(jnp.float32)
    else:
        sinks_spec = None

    do_spec = o_spec

    def logsumexp_index_map(h, i, *_):
        return h, 0, i

    lse = jnp.expand_dims(lse, axis=-2)
    logsumexp_spec = pl.BlockSpec((None, 1, bq), logsumexp_index_map)
    assert lse.ndim == len(logsumexp_spec.block_shape)

    di = jnp.expand_dims(di, axis=-2)
    di_spec = pl.BlockSpec((None, 1, bq), logsumexp_index_map)
    assert di.ndim == len(di_spec.block_shape)

    in_specs = [
        q_spec,
        k_spec,
        v_spec,
        q_segment_spec,
        kv_segment_spec,
        sinks_spec,
        logsumexp_spec,
        do_spec,
        di_spec,
    ]
    if mask_info.partial_mask_blocks is not None:
        in_specs.append(mask_spec)
    else:
        in_specs.append(None)

    assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

    if mask_info.q_sequence is not None:
        q_sequence = jax.lax.broadcast_in_dim(mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,))
        in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
    else:
        q_sequence = None
        in_specs.append(None)

    out_shapes = [
        jax.ShapeDtypeStruct((bq, head_dim_qk), jnp.float32),
        jax.ShapeDtypeStruct(q.shape, q.dtype),
    ]
    out_specs = [
        pl.BlockSpec((bq, head_dim_qk), lambda *_: (0, 0)),
        pl.BlockSpec((None, bq, head_dim_qk), lambda h, i, *_: (h, i, 0)),
    ]

    kernel = functools.partial(
        _flash_attention_dq_kernel,
        grid_width=grid_width,
        mask_value=mask_value,
        bq=bq,
        bkv=bkv,
        logits_soft_cap=logits_soft_cap,
        q_layout=q_layout,
        k_layout=k_layout,
        v_layout=v_layout,
        mask_function=mask_function,
    )
    num_scalar_prefetch = 3

    kernel_name = get_kernel_name(
        dict(
            block_q_dq=bq,
            block_kv_dq=bkv,
            q_layout=q_layout,
            k_layout=k_layout,
            v_layout=v_layout,
        ),
        is_mqa=is_mqa,
        save_residuals=False,
        is_segmented=segment_ids is not None,
        phase="dq",
    )
    with jax.named_scope(kernel_name):
        _, dq = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=num_scalar_prefetch,
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
            ),
            out_shape=out_shapes,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
            ),
            name=kernel_name,
            interpret=interpret,
        )(
            mask_info.data_next,
            mask_info.block_mask,
            mask_info.mask_next,
            q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
            k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
            v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.swapaxes(-1, -2),
            q_segment_ids,
            kv_segment_ids,
            sinks,
            lse,
            do,
            di,
            mask_info.partial_mask_blocks,
            q_sequence,
        )
    return dq


def _flash_attention_dkv_kernel(
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    dq_scratch_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv_compute: int,
    is_mqa: bool,
    logits_soft_cap: float | None,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    bkv: int,
    mask_function: MaskFunctionType | None,
):
    del sinks_ref
    HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
    kv_index, q_head_index, q_index = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
    )
    should_initialize = q_index == 0

    q_heads_per_kv_heads = None
    q_head_index_per_kv_head = None

    if is_mqa:
        should_initialize = jnp.logical_and(should_initialize, q_head_index == 0)
    elif num_kv_heads < num_q_heads:
        q_heads_per_kv_heads = num_q_heads // num_kv_heads
        q_head_index_per_kv_head = lax.rem(q_head_index, q_heads_per_kv_heads)
        should_initialize = jnp.logical_and(should_initialize, q_head_index_per_kv_head == 0)

    @pl.when(should_initialize)
    def init():
        dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
        dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

    _, _, should_run, should_not_mask = _next_nonzero(
        q_head_index,
        q_index,
        kv_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref,
        next_i=True,
    )

    def body(i, _):
        slice_k = pl.ds(i * bkv_compute, bkv_compute)
        q = q_ref[...]

        def _load_kv(ref, layout):
            if layout == HEAD_DIM_MINOR:
                return ref[slice_k, :]
            return ref[:, slice_k].T

        k = _load_kv(k_ref, k_layout)
        v = _load_kv(v_ref, v_layout)
        lse = logsumexp_ref[:1, :]
        do = do_ref[...]
        di = di_ref[:1, :]

        qk_dims = NT_DIM_NUMBERS if q_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
        qk_uncapped = lax.dot_general(k, q, qk_dims, preferred_element_type=jnp.float32)

        qk = _apply_mask_and_soft_cap(
            qk_uncapped,
            mask_value,
            should_not_mask,
            mask_ref,
            q_sequence_ref,
            q_segment_ids_ref,
            kv_segment_ids_ref,
            logits_soft_cap=logits_soft_cap,
            k_slice=slice_k,
            k_offset=kv_index * bkv + i * bkv_compute,
            bq=bq,
            k_in_lanes=False,
            mask_function=mask_function,
        )
        p = jnp.exp(qk - lse)
        dv = lax.dot(p.astype(do.dtype), do, preferred_element_type=jnp.float32)
        dv = dv.astype(dv_scratch_ref.dtype) + dv_scratch_ref[slice_k, :]
        dv_scratch_ref[slice_k, :] = dv

        dp = lax.dot_general(
            v,
            do,
            NT_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        ds = (dp - di) * p
        if logits_soft_cap is not None:
            normalized = qk_uncapped / logits_soft_cap
            d = jnp.tanh(normalized)
            g = ds * (1 - d)
            ds = g + g * d
        dk_dims = NN_DIM_NUMBERS if q_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
        dk = lax.dot_general(ds.astype(do.dtype), q, dk_dims, preferred_element_type=jnp.float32)
        dk = dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[slice_k, :]
        dk_scratch_ref[slice_k, :] = dk
        if dq_scratch_ref is not None or dq_ref is not None:
            dq = lax.dot_general(
                ds.T.astype(k.dtype),
                k,
                NN_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            if dq_scratch_ref is not None:
                dq_scratch_ref[...] += dq
            else:
                assert dq_ref is not None
                dq_ref[...] = dq.astype(dq_ref.dtype)

    if dq_scratch_ref is not None:
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)
    elif dq_scratch_ref is None and dq_ref is not None:
        dq_ref[...] = jnp.zeros_like(dq_ref)

    @pl.when(should_run)
    def run():
        num_iters = k_ref.shape[0 if k_layout is HEAD_DIM_MINOR else 1] // bkv_compute
        lax.fori_loop(0, num_iters, body, None, unroll=True)

    if dq_scratch_ref is not None:
        assert dq_ref is not None
        dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)

    should_write = q_index == grid_width - 1
    if is_mqa:
        should_write = jnp.logical_and(should_write, q_head_index == num_q_heads - 1)
    elif num_kv_heads < num_q_heads:
        should_write = jnp.logical_and(should_write, q_head_index_per_kv_head == q_heads_per_kv_heads - 1)

    @pl.when(should_write)
    def end():
        dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
        dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)
        if dq_scratch_ref is not None:
            dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

        dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
        dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)


def _splash_attention_bwd_dkv(
    q,
    k,
    v,
    segment_ids,
    sinks,
    lse,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    is_mqa: bool,
    mask_info: mask_info_lib.MaskInfo,
    mask_value: float,
    logits_soft_cap: float | None,
    use_fused_bwd_kernel: bool,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    mask_function: MaskFunctionType | None,
    interpret: bool,
):
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = v.shape[-1]
    if is_mqa:
        num_kv_heads, kv_seq_len = 1, k.shape[0]
    else:
        num_kv_heads, kv_seq_len, _ = k.shape

    if bq > q_seq_len:
        raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
    if bkv > kv_seq_len:
        raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")
    if bkv_compute > bkv:
        raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}")
    if bkv % bkv_compute:
        raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}")

    if not is_mqa and num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
            f" multiple of the number of 'query' heads ({num_q_heads})"
        )

    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError(f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same leading dimensions.")

    q_heads_per_kv_head = num_q_heads // num_kv_heads

    if mask_info.data_next is not None:
        grid_width = mask_info.data_next.shape[-2]
    else:
        grid_width = q_seq_len // bq

    grid = (
        kv_seq_len // bkv,
        num_q_heads,
        grid_width,
    )

    def o_index_map(
        kv_index,
        head_index,
        q_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref=None,
    ):
        next_i, *_ = _next_nonzero(
            head_index,
            q_index,
            kv_index,
            data_next_ref,
            block_mask_ref,
            mask_next_ref,
            next_i=True,
        )
        return head_index, next_i, 0

    o_spec = pl.BlockSpec((None, bq, head_dim_v), o_index_map)

    def q_index_map(
        kv_index,
        head_index,
        q_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref=None,
    ):
        next_i, *_ = _next_nonzero(
            head_index,
            q_index,
            kv_index,
            data_next_ref,
            block_mask_ref,
            mask_next_ref,
            next_i=True,
        )
        return from_head_minor((head_index, next_i, 0), q_layout)

    q_spec = pl.BlockSpec(from_head_minor((None, bq, head_dim_qk), q_layout), q_index_map)

    def k_index_map(kv_index, head_index, *_):
        prefix = () if is_mqa else (_div(head_index, q_heads_per_kv_head),)
        return from_head_minor((*prefix, kv_index, 0), k_layout)

    k_spec = pl.BlockSpec(
        from_head_minor(
            (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
            k_layout,
        ),
        k_index_map,
    )

    def v_index_map(kv_index, head_index, *_):
        prefix = () if is_mqa else (_div(head_index, q_heads_per_kv_head),)
        return from_head_minor((*prefix, kv_index, 0), v_layout)

    v_spec = pl.BlockSpec(
        from_head_minor(
            (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
            v_layout,
        ),
        v_index_map,
    )

    if use_fused_bwd_kernel:

        def dq_index_map(kv_index, head_index, q_index, *_):
            return (kv_index, head_index, q_index, 0)

        dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
        dq_shape = jax.ShapeDtypeStruct((kv_seq_len // bkv, *q.shape), q.dtype)
        if bkv == bkv_compute:
            dq_scratch_spec = dq_scratch_shape = None
        else:
            dq_scratch_spec = pl.BlockSpec((bq, head_dim_qk), lambda *_: (0, 0))
            dq_scratch_shape = jax.ShapeDtypeStruct((bq, head_dim_qk), jnp.float32)
    else:
        dq_spec = dq_shape = dq_scratch_spec = dq_scratch_shape = None

    def dkv_index_map(kv_index, head_index, *_):
        prefix = () if is_mqa else (_div(head_index, q_heads_per_kv_head),)
        return (*prefix, kv_index, 0)

    dk_spec = pl.BlockSpec(
        (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
        dkv_index_map,
    )

    dv_spec = pl.BlockSpec(
        (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
        dkv_index_map,
    )

    def mask_index_map(
        kv_index,
        head_index,
        q_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref,
    ):
        _, next_m, *_ = _next_nonzero(
            head_index,
            q_index,
            kv_index,
            data_next_ref,
            block_mask_ref,
            mask_next_ref,
            next_i=True,
        )
        return next_m, 0, 0

    mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

    def q_segment_ids_index_map(
        kv_index,
        head_index,
        q_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref=None,
    ):
        next_i, *_ = _next_nonzero(
            head_index,
            q_index,
            kv_index,
            data_next_ref,
            block_mask_ref,
            mask_next_ref,
            next_i=True,
        )
        return 0, next_i

    if segment_ids is not None:

        def kv_segment_ids_index_map(kv_index, *_):
            return kv_index, 0

        q_segment_spec = pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map)
        kv_segment_spec = pl.BlockSpec((bkv, NUM_LANES), kv_segment_ids_index_map)
        q_segment_ids = jax.lax.broadcast_in_dim(segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,))
        kv_segment_ids = jax.lax.broadcast_in_dim(segment_ids.kv, (kv_seq_len, NUM_LANES), (0,))
    else:
        q_segment_spec = kv_segment_spec = None
        q_segment_ids = kv_segment_ids = None

    if sinks is not None:
        assert sinks.shape == (num_q_heads,)
        sinks_spec = pl.BlockSpec((num_q_heads,), lambda kv_index, h, *_: (0,), memory_space=pltpu.SMEM)
        sinks = sinks.astype(jnp.float32)
    else:
        sinks_spec = None

    do_spec = o_spec

    def logsumexp_index_map(
        kv_index,
        head_index,
        q_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref=None,
    ):
        next_i, *_ = _next_nonzero(
            head_index,
            q_index,
            kv_index,
            data_next_ref,
            block_mask_ref,
            mask_next_ref,
            next_i=True,
        )
        return head_index, 0, next_i

    assert lse.shape == di.shape == (num_q_heads, q_seq_len)

    logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
    lse = jnp.broadcast_to(jnp.expand_dims(lse, -2), logsumexp_shape)
    logsumexp_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
    assert lse.ndim == len(logsumexp_spec.block_shape)

    di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
    di_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
    assert di.ndim == len(di_spec.block_shape)

    in_specs = [
        q_spec,
        k_spec,
        v_spec,
        q_segment_spec,
        kv_segment_spec,
        sinks_spec,
        logsumexp_spec,
        do_spec,
        di_spec,
    ]
    if mask_info.partial_mask_blocks is not None:
        in_specs.append(mask_spec)
    else:
        in_specs.append(None)

    if mask_info.q_sequence is not None:
        in_specs.append(pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map))
        q_sequence = jax.lax.broadcast_in_dim(mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,))
    else:
        q_sequence = None
        in_specs.append(None)

    out_shapes = [
        dq_scratch_shape,
        jax.ShapeDtypeStruct((bkv, head_dim_qk), jnp.float32),
        jax.ShapeDtypeStruct((bkv, head_dim_v), jnp.float32),
        dq_shape,
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
    ]
    out_specs = [
        dq_scratch_spec,
        pl.BlockSpec((bkv, head_dim_qk), lambda *_: (0, 0)),
        pl.BlockSpec((bkv, head_dim_v), lambda *_: (0, 0)),
        dq_spec,
        dk_spec,
        dv_spec,
    ]

    kernel = functools.partial(
        _flash_attention_dkv_kernel,
        mask_value=mask_value,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        is_mqa=is_mqa,
        grid_width=grid_width,
        bq=bq,
        bkv_compute=bkv_compute,
        logits_soft_cap=logits_soft_cap,
        q_layout=q_layout,
        k_layout=k_layout,
        v_layout=v_layout,
        bkv=bkv,
        mask_function=mask_function,
    )
    num_scalar_prefetch = 3

    kernel_name = get_kernel_name(
        dict(
            block_q_dkv=bq,
            block_kv_dkv=bkv,
            block_kv_dkv_compute=bkv_compute,
            q_layout=q_layout,
            k_layout=k_layout,
            v_layout=v_layout,
        ),
        is_mqa=is_mqa,
        save_residuals=False,
        is_segmented=segment_ids is not None,
        phase="dkv",
    )
    with jax.named_scope(kernel_name):
        _, _, _, dq_unreduced, dk, dv = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=num_scalar_prefetch,
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
            ),
            out_shape=out_shapes,
            compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary", "arbitrary")),
            name=kernel_name,
            interpret=interpret,
        )(
            mask_info.data_next,
            mask_info.block_mask,
            mask_info.mask_next,
            q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
            k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
            v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.swapaxes(-1, -2),
            q_segment_ids,
            kv_segment_ids,
            sinks,
            lse,
            do,
            di,
            mask_info.partial_mask_blocks,
            q_sequence,
        )
    if use_fused_bwd_kernel:
        assert dq_unreduced is not None
        dq = dq_unreduced.sum(axis=0)
    else:
        assert dq_unreduced is None
        dq = None
    return dq, dk, dv


def _splash_attention_bwd(
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    interpret: bool,
    res: SplashResidualsType,
    do: jax.Array,
) -> tuple[
    mask_info_lib.MaskInfo | None,
    mask_info_lib.MaskInfo | None,
    mask_info_lib.MaskInfo | None,
    jax.Array,
    jax.Array,
    jax.Array,
    SegmentIds | None,
    jax.Array | None,
]:
    del save_residuals, residual_checkpoint_name
    if not block_sizes.has_backward_blocks:
        raise ValueError("Need to specify backward blocks.")
    bq_dq, bkv_dq = block_sizes.block_q_dq, block_sizes.block_kv_dq
    bq_dkv, bkv_dkv_memory, bkv_dkv_compute = (
        block_sizes.block_q_dkv,
        block_sizes.block_kv_dkv,
        block_sizes.block_kv_dkv_compute,
    )
    use_fused_bwd_kernel = block_sizes.use_fused_bwd_kernel
    (
        q,
        k,
        v,
        segment_ids,
        sinks,
        o,
        lse,
        dq_mask_info,
        dkv_mask_info,
    ) = res

    di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))
    dq, dk, dv = _splash_attention_bwd_dkv(
        q,
        k,
        v,
        segment_ids,
        sinks,
        lse,
        do,
        di,
        bq=bq_dkv,
        bkv=bkv_dkv_memory,
        bkv_compute=bkv_dkv_compute,
        is_mqa=is_mqa,
        mask_info=dkv_mask_info,
        mask_value=mask_value,
        logits_soft_cap=logits_soft_cap,
        use_fused_bwd_kernel=use_fused_bwd_kernel,
        q_layout=block_sizes.q_layout,
        k_layout=block_sizes.k_layout,
        v_layout=block_sizes.v_layout,
        mask_function=mask_function,
        interpret=interpret,
    )
    if not use_fused_bwd_kernel:
        assert dq is None
        dq = _splash_attention_bwd_dq(
            q,
            k,
            v,
            segment_ids,
            sinks,
            lse,
            do,
            di,
            bq=bq_dq,
            bkv=bkv_dq,
            is_mqa=is_mqa,
            mask_info=dq_mask_info,
            mask_value=mask_value,
            logits_soft_cap=logits_soft_cap,
            q_layout=block_sizes.q_layout,
            k_layout=block_sizes.k_layout,
            v_layout=block_sizes.v_layout,
            mask_function=mask_function,
            interpret=interpret,
        )

    assert dq is not None
    dsinks = None
    if sinks is not None:
        sinks_exp = -jnp.exp(sinks[..., None, None].astype(jnp.float32) - lse[..., None].astype(jnp.float32))
        dsinks = jnp.sum(sinks_exp.astype(o.dtype) * o * do, axis=(-1, -2))
    return (
        None,
        None,
        None,
        dq,
        dk,
        dv,
        None,
        dsinks,
    )


_splash_attention_custom.defvjp(_splash_attention_fwd, _splash_attention_bwd)


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "block_sizes",
        "save_residuals",
        "mask_value",
        "logits_soft_cap",
        "residual_checkpoint_name",
        "mask_function",
        "interpret",
    ],
)
def _splash_attention(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    block_sizes: BlockSizes | None,
    save_residuals: bool,
    mask_value: float,
    logits_soft_cap: float | None,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    interpret: bool,
) -> SplashCustomReturnType:
    """
    For dynamic masks, `partial_mask_blocks` has shape (head_count, q_blocks, kv_blocks, block_q, block_kv).
    This shape allows sharding across both head count and query sequence dimensions.

    Note: The leading dimensions (head_count, q_blocks, kv_blocks) must be
    collapsed into a single dimension before being passed to the kernel.
    """

    def _collapse_partial_mask_blocks(mask_info: mask_info_lib.MaskInfo | None):
        if mask_info is None or mask_info.partial_mask_blocks is None:
            return mask_info

        return mask_info._replace(
            partial_mask_blocks=mask_info.partial_mask_blocks.reshape(-1, *mask_info.partial_mask_blocks.shape[-2:])
        )

    fwd_mask_info = _collapse_partial_mask_blocks(fwd_mask_info)
    dq_mask_info = _collapse_partial_mask_blocks(dq_mask_info)
    dkv_mask_info = _collapse_partial_mask_blocks(dkv_mask_info)
    return _splash_attention_custom(
        fwd_mask_info,
        dq_mask_info,
        dkv_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        save_residuals=save_residuals,
        logits_soft_cap=logits_soft_cap,
        residual_checkpoint_name=residual_checkpoint_name,
        mask_function=mask_function,
        interpret=interpret,
    )


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:
    def __init__(
        self,
        fwd_mask_info: mask_info_lib.MaskInfo,
        dq_mask_info: mask_info_lib.MaskInfo | None,
        dkv_mask_info: mask_info_lib.MaskInfo | None,
        **kwargs,
    ):
        """Initialize the SplashAttentionKernel with mask metadata.

        Args:
            fwd_mask_info: Sparse mask information for the forward pass.
            dq_mask_info: Sparse mask information for the dQ backward pass,
                or None if backward is not needed.
            dkv_mask_info: Sparse mask information for the dKV backward pass,
                or None if backward is not needed.
            **kwargs: Additional kernel configuration (block_sizes, is_mqa,
                save_residuals, mask_value, logits_soft_cap, etc.).
        """
        self.kwargs = kwargs
        self.fwd_mask_info = fwd_mask_info
        self.dq_mask_info = dq_mask_info
        self.dkv_mask_info = dkv_mask_info

    def __call__(self, *args, **kwargs) -> SplashCustomReturnType:
        """Execute the Splash Attention kernel.

        Dispatches to the internal ``_splash_attention`` function with the
        stored mask metadata and kernel configuration.

        Args:
            *args: Positional arguments (q, k, v, segment_ids, sinks).
            **kwargs: Keyword arguments overriding stored configuration.

        Returns:
            Attention output tensor, or (output, (log_sum_exp,)) if
            save_residuals was set to True during construction.
        """
        return _splash_attention(
            self.fwd_mask_info,
            self.dq_mask_info,
            self.dkv_mask_info,
            *args,
            **kwargs,
            **self.kwargs,
        )

    def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
        """Returns a value that can be used as a shard_map partition spec for the kernel."""
        if self.fwd_mask_info.data_next is not None:
            block_mask_shape = self.fwd_mask_info.data_next.shape
            try:
                shard_shape = sharding.shard_shape(block_mask_shape)
            except ValueError as exc:
                raise ValueError("The sharding must divide the mask blocks evenly between devices") from exc
            if block_mask_shape[-1] != shard_shape[-1]:
                raise ValueError("Sharding the kv sequence dimension is not supported")
        spec = sharding.spec
        assert len(spec) == 2
        replicated = jax.sharding.PartitionSpec()
        partial_mask_blocks_spec = spec if self.fwd_mask_info.is_dynamic_mask else replicated

        q_sequence_spec = jax.sharding.PartitionSpec(spec[1])
        mask_info_specs = mask_info_lib.MaskInfo(
            data_next=spec if self.fwd_mask_info.data_next is not None else None,
            mask_next=spec if self.fwd_mask_info.mask_next is not None else None,
            block_mask=spec if self.fwd_mask_info.block_mask is not None else None,
            partial_mask_blocks=partial_mask_blocks_spec if self.fwd_mask_info.partial_mask_blocks is not None else None,
            q_sequence=q_sequence_spec if self.fwd_mask_info.q_sequence is not None else None,
        )
        return SplashAttentionKernel(
            mask_info_specs,
            mask_info_specs if self.dq_mask_info is not None else None,
            mask_info_specs if self.dkv_mask_info is not None else None,
            **self.kwargs,
        )

    def tree_flatten(self):
        return (
            (self.fwd_mask_info, self.dq_mask_info, self.dkv_mask_info),
            self.kwargs,
        )

    @classmethod
    def tree_unflatten(cls, kwargs, values):
        fwd_mask_info, dq_mask_info, dkv_mask_info = values

        dq_mask_info = mask_info_lib.MaskInfo(*dq_mask_info) if dq_mask_info is not None else None
        dkv_mask_info = mask_info_lib.MaskInfo(*dkv_mask_info) if dkv_mask_info is not None else None
        return SplashAttentionKernel(
            mask_info_lib.MaskInfo(*fwd_mask_info),
            dq_mask_info,
            dkv_mask_info,
            **kwargs,
        )


def _make_splash_attention(
    mask: np.ndarray | jax.Array | mask_lib.MultiHeadMask,
    *,
    block_sizes: BlockSizes | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    logits_soft_cap: float | None = None,
    downcast_smem_data: bool = True,
    head_shards: int,
    q_seq_shards: int,
    residual_checkpoint_name: str | None = None,
    interpret: bool = False,
):
    """Create a SplashAttentionKernel from a dense or lazy mask.

    Processes the input mask into sparse MaskInfo representations for forward
    and backward passes, then constructs a callable kernel object. Supports
    both static (compile-time) and dynamic (traced) masks.

    Args:
        mask: Attention mask as a 3D array [num_heads, q_seq_len, kv_seq_len]
            or a MultiHeadMask object.
        block_sizes: Tile sizes for the kernel. Uses defaults if None.
        is_mqa: If True, build for multi-query attention.
        save_residuals: If True, kernel returns log-sum-exp for backward pass.
        mask_value: Value for masked positions.
        logits_soft_cap: Optional soft capping value for attention logits.
        downcast_smem_data: If True, minimize scalar memory by downcasting.
        head_shards: Number of shards along the head dimension.
        q_seq_shards: Number of shards along the query sequence dimension.
        residual_checkpoint_name: Optional name for activation checkpointing.
        interpret: If True, run kernel in interpret mode for debugging.

    Returns:
        A SplashAttentionKernel instance ready for execution.

    Raises:
        ValueError: If mask shape is not 3-dimensional.
    """
    if len(mask.shape) != 3:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    if isinstance(mask, np.ndarray):
        mask = mask_lib.MultiHeadMask([mask_lib.NumpyMask(head_mask) for head_mask in mask])

    if block_sizes is None:
        block_sizes = BlockSizes.get_default()

    process_mask_fn = mask_info_lib.process_dynamic_mask if isinstance(mask, jax.Array) else mask_info_lib.process_mask

    process_mask_dvk_fn = (
        mask_info_lib.process_dynamic_mask_dkv if isinstance(mask, jax.Array) else mask_info_lib.process_mask_dkv
    )

    fwd_mask_info, mask_function_fwd = process_mask_fn(
        mask,
        (block_sizes.block_q, block_sizes.block_kv),
        downcast_smem_data=downcast_smem_data,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
    )
    fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

    dq_mask_info = None
    dkv_mask_info = None
    if block_sizes.has_backward_blocks:
        if block_sizes.use_fused_bwd_kernel:
            dq_mask_info = None
        else:
            bq_dq, bkv_dq = block_sizes.block_q_dq, block_sizes.block_kv_dq
            dq_mask_info, mask_function_dq = process_mask_fn(
                mask,
                (bq_dq, bkv_dq),
                downcast_smem_data=downcast_smem_data,
                head_shards=head_shards,
                q_seq_shards=q_seq_shards,
            )
            assert (mask_function_fwd is None) == (mask_function_dq is None)
            dq_mask_info = tree_util.tree_map(jnp.array, dq_mask_info)
        bq_dkv, bkv_dkv = block_sizes.block_q_dkv, block_sizes.block_kv_dkv
        dkv_mask_info, mask_function_dkv = process_mask_dvk_fn(
            mask,
            (bq_dkv, bkv_dkv),
            downcast_smem_data=downcast_smem_data,
            head_shards=head_shards,
            q_seq_shards=q_seq_shards,
            shrink_grid=not block_sizes.use_fused_bwd_kernel,
        )
        assert (mask_function_fwd is None) == (mask_function_dkv is None)

        dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)

    return SplashAttentionKernel(
        fwd_mask_info,
        dq_mask_info,
        dkv_mask_info,
        block_sizes=block_sizes,
        is_mqa=is_mqa,
        save_residuals=save_residuals,
        mask_value=mask_value,
        logits_soft_cap=logits_soft_cap,
        residual_checkpoint_name=residual_checkpoint_name,
        mask_function=mask_function_fwd,
        interpret=interpret,
    )


make_splash_mha = partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = partial(_make_splash_attention, is_mqa=True)

make_splash_mha_single_device = partial(make_splash_mha, is_mqa=False, head_shards=1, q_seq_shards=1)
make_splash_mqa_single_device = partial(make_splash_mha, is_mqa=True, head_shards=1, q_seq_shards=1)


@ejit(
    static_argnames=(
        "softmax_scale",
        "mask_builder",
        "sliding_window",
        "chunk_size",
        "causal",
        "fused_backward",
        "fwd_params",
        "bwd_params",
        "logits_soft_cap",
    )
)
def blocksparse_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch kv_num_heads kv_len head_dim"],
    value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
    q_segment_ids: Int[Array, "batch seq_len"] | None = None,
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    q_positions: Int[Array, "batch seq_len"] | None = None,
    kv_positions: Int[Array, "batch kv_len"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | Int[Array, "batch num_heads_or_1 seq_len kv_len"] | None
    ) = None,
    sequence_parallelism_mesh_axis_name: str | None = None,
    logits_soft_cap: float | None = None,
    qkv_layouts: tuple["SparseMask"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    mask_builder: Callable[[int, int, int, int, int], "Mask"] | Callable[[], "SparseMask"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
) -> Float[Array, "batch num_heads seq_len vhead_dim"]:
    """Pallas TPU block-sparse attention kernel implementation.

    Computes attention over sparse block patterns using Pallas kernels optimized for TPU execution.

    Args:
        query: Query tensor [batch num_heads seq_len head_dim]
        key: Key tensor [batch kv_num_heads kv_len head_dim]
        value: Value tensor [batch kv_num_heads kv_len vhead_dim]
        q_segment_ids: Optional query segment ids [batch, seq_len]
        kv_segment_ids: Optional KV segment ids [batch, kv_len]
        q_positions: Optional query position indices [batch, seq_len] (not implemented for TPU)
        kv_positions: Optional KV position indices [batch, kv_len] (not implemented for TPU)
        softmax_aux: Optional auxiliary softmax values for attention sinks
        bias: Optional attention bias [batch num_heads seq_len head_dim]
        sequence_parallelism_mesh_axis_name: Optional mesh axis name for sequence parallelism
        logits_soft_cap: Optional soft capping value for attention logits. When specified,
            applies tanh-based soft capping: logits_soft_cap * tanh(logits / logits_soft_cap).
            This prevents attention scores from becoming too large, improving numerical
            stability (Gemma-2 style). Gradients are computed with proper Jacobian.
        qkv_layouts: Optional pre-computed attention mask layouts
        softmax_scale: Attention score scaling factor (default: 1/sqrt(head_dim))
        mask_builder: Custom mask builder function
        sliding_window: Sliding window size. Can be:
            - int: symmetric window (same size left and right)
            - tuple[int, int]: (left_window, right_window) for asymmetric
            - None: no sliding window
        chunk_size: Size of chunks for chunked causal attention (like Llama4)
            - int: enable chunked causal mask with specified chunk size
            - None: no chunking
        causal: Whether to use causal masking (default True)
        fused_backward: Whether to use fused backward kernel

    Returns:
        Attention output [batch num_heads seq_len vhead_dim]
    """
    if q_positions is not None and q_segment_ids is None:
        raise NotImplementedError("`q_positions` is not implemented for tpu-pallas (gpu-triton and xla only).")
    if kv_positions is not None and kv_segment_ids is None:
        raise NotImplementedError("`kv_positions` is not implemented for tpu-pallas (gpu-triton and xla only).")
    if bias is not None:
        raise NotImplementedError("`bias` is not implemented for tpu-pallas (gpu-triton and xla only).")
    if sequence_parallelism_mesh_axis_name is not None:
        raise NotImplementedError(
            "`sequence_parallelism_mesh_axis_name` is not implemented for tpu-pallas (gpu-triton and xla only)."
        )
    if qkv_layouts is not None:
        raise NotImplementedError("`qkv_layouts` is not implemented for tpu-pallas (gpu-triton and xla only).")

    query_length = query.shape[2]
    kv_length = value.shape[2]
    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=512, kv_blocksize=512, num_stages=2, num_warps=4)
    if bwd_params is None:
        bwd_params = BwdParams(q_blocksize=1024, kv_blocksize=1024, num_stages=2, num_warps=4)

    if attention_mask is not None and (q_segment_ids is None or kv_segment_ids is None):
        from ejkernel.types.mask import mask_to_segment_ids

        inferred_q_seg, inferred_kv_seg = mask_to_segment_ids(attention_mask)
        if q_segment_ids is None:
            q_segment_ids = inferred_q_seg
        if kv_segment_ids is None:
            kv_segment_ids = inferred_kv_seg

    if q_segment_ids is not None and kv_segment_ids is None:
        raise ValueError("If `q_segment_ids` is provided, `kv_segment_ids` must also be provided.")
    if kv_segment_ids is not None and q_segment_ids is None:
        raise ValueError("If `kv_segment_ids` is provided, `q_segment_ids` must also be provided.")

    if mask_builder is None:

        def mask_builder(q_len: int, kv_len: int, num_heads: int, head_idx: int, num_reps: int) -> Mask:
            if chunk_size is not None:
                return ChunkedCausalMask((q_len, kv_len), chunk_size=chunk_size)

            elif sliding_window is not None:
                if isinstance(sliding_window, int):
                    left_window = right_window = sliding_window
                else:
                    left_window, right_window = sliding_window

                local_mask = LocalMask(shape=(q_len, kv_len), window_size=(left_window, right_window), offset=0)

                if causal:
                    causal_mask = CausalMask((q_len, kv_len))
                    return causal_mask & local_mask
                else:
                    return local_mask

            elif causal:
                return CausalMask((q_len, kv_len))
            else:
                return FullMask((q_len, kv_len))

    block_sizes = BlockSizes(
        block_q=min(fwd_params.q_blocksize, query_length),
        block_kv_compute=min(fwd_params.kv_blocksize, kv_length),
        block_kv=min(fwd_params.kv_blocksize, kv_length),
        block_q_dkv=min(bwd_params.q_blocksize, query_length),
        block_kv_dkv=min(bwd_params.kv_blocksize, kv_length),
        block_kv_dkv_compute=min(bwd_params.kv_blocksize, kv_length),
        block_q_dq=min(bwd_params.q_blocksize, query_length),
        block_kv_dq=min(bwd_params.kv_blocksize, kv_length),
        use_fused_bwd_kernel=fused_backward,
    )
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    assert query_length != 1
    mask = MultiHeadMask(
        [mask_builder(query_length, kv_length, query.shape[-3], ox, query.shape[-2]) for ox in range(query.shape[-3])]
    )

    def attn_static_fn(
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
        softmax_aux,
    ):
        segment_ids = None

        if kv_segment_ids is not None and q_segment_ids is not None:
            segment_ids = SegmentIds(q_segment_ids, kv_segment_ids)

        return make_splash_mha(
            mask=mask,
            block_sizes=block_sizes,
            logits_soft_cap=logits_soft_cap,
            head_shards=1,
            q_seq_shards=1,
        )(
            q=q,
            k=k,
            v=v,
            segment_ids=segment_ids,
            sinks=softmax_aux,
        )

    attn_fn = jax.vmap(
        attn_static_fn,
        in_axes=(
            0,
            0,
            0,
            0 if q_segment_ids is not None else None,
            0 if kv_segment_ids is not None else None,
            None,
        ),
    )

    return attn_fn(
        query * softmax_scale,
        key,
        value,
        q_segment_ids,
        kv_segment_ids,
        softmax_aux,
    )
