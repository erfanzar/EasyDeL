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

"""Public interface for the CUDA block-sparse attention kernel.

This module exposes :func:`blocksparse_attention`, which is registered in the
ejkernel kernel registry under the ``CUDA`` platform and ``GPU`` backend.  It
also implements the JAX custom-VJP rule so that:

* The **forward pass** is dispatched to the compiled CUDA kernel via
  :func:`._cuda_impl.blocksparse_attention_cuda`.
* The **backward pass** uses a CUDA-side dense analytical fallback that
  preserves block-sparse masking semantics.

Helper utilities for converting sparsity layouts to dense boolean masks and
for creating empty placeholder layouts are also defined here.
"""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from beartype.typing import Callable
from jax.interpreters import ad
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from ejkernel.callib import ejit
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.ops import BwdParams, FwdParams

from ._cuda_impl import blocksparse_attention_cuda

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask
    from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask


_TRITON_SPARSE_MASK_TYPE: type[typing.Any] | None = None

try:
    from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask as _TRITON_SPARSE_MASK_TYPE
    from ejkernel.kernels._triton.blocksparse_attention._mask import create_sparsity_mask
except ModuleNotFoundError as _triton_mask_import_error:  # pragma: no cover
    if _triton_mask_import_error.name != "triton":
        raise
    _triton_mask_import_exc = _triton_mask_import_error

    def create_sparsity_mask(*args: typing.Any, _err: Exception = _triton_mask_import_exc, **kwargs: typing.Any):  # type: ignore[override]
        del args, kwargs
        raise ValueError(
            "`blocksparse_attention` mask auto-generation requires Triton "
            "(install `ejkernel[gpu]`), or pass precomputed `qkv_layouts`."
        ) from _err


def _build_fwd_params(q_blocksize: int, kv_blocksize: int, num_stages: int | None, num_warps: int | None) -> FwdParams:
    """Construct a :class:`FwdParams` instance from individual block-size arguments.

    Args:
        q_blocksize: Query block size for the CUDA kernel tiling.
        kv_blocksize: Key/value block size for the CUDA kernel tiling.
        num_stages: Number of pipeline stages, or ``None`` to leave unset.
        num_warps: Number of GPU warps per thread block, or ``None`` to leave unset.

    Returns:
        A :class:`~ejkernel.ops.FwdParams` populated with the given values.
    """
    return FwdParams(
        q_blocksize=int(q_blocksize),
        kv_blocksize=int(kv_blocksize),
        num_stages=None if num_stages is None else int(num_stages),
        num_warps=None if num_warps is None else int(num_warps),
    )


def _build_bwd_params(q_blocksize: int, kv_blocksize: int, num_stages: int | None, num_warps: int | None) -> BwdParams:
    """Construct a :class:`BwdParams` instance from individual block-size arguments.

    Args:
        q_blocksize: Query block size for the backward kernel tiling.
        kv_blocksize: Key/value block size for the backward kernel tiling.
        num_stages: Number of pipeline stages, or ``None`` to leave unset.
        num_warps: Number of GPU warps per thread block, or ``None`` to leave unset.

    Returns:
        A :class:`~ejkernel.ops.BwdParams` populated with the given values.
    """
    return BwdParams(
        q_blocksize=int(q_blocksize),
        kv_blocksize=int(kv_blocksize),
        num_stages=None if num_stages is None else int(num_stages),
        num_warps=None if num_warps is None else int(num_warps),
    )


def _normalize_softmax_aux(
    softmax_aux: ArrayLike | None,
    *,
    num_heads: int,
    num_kv_heads: int,
) -> Float[Array, "num_heads num_sinks"] | None:
    """Normalize and broadcast softmax auxiliary (attention-sink) weights.

    Converts *softmax_aux* into a ``float32`` array of shape
    ``(num_heads, num_sinks)`` suitable for the backward dense-attention path.
    The conversion rules mirror those of
    :func:`._cuda_impl._normalize_softmax_aux`:

    * **1-D** ``(num_heads,)`` -- reshaped to ``(num_heads, 1)``.
    * **1-D** ``(num_kv_heads,)`` -- repeated by the GQA group factor
      ``num_heads // num_kv_heads``, then reshaped to ``(num_heads, 1)``.
    * **1-D** ``(num_sinks,)`` -- broadcast to ``(num_heads, num_sinks)``.
    * **2-D** ``(num_heads, num_sinks)`` -- returned unchanged.
    * **2-D** ``(num_kv_heads, num_sinks)`` -- repeated to
      ``(num_heads, num_sinks)`` via the GQA group factor.

    Args:
        softmax_aux: Optional auxiliary weights. ``None`` disables
            attention-sink logic.
        num_heads: Total number of query heads.
        num_kv_heads: Number of key/value heads (may differ under GQA).

    Returns:
        A ``float32`` array of shape ``(num_heads, num_sinks)``, or
        ``None`` if *softmax_aux* is ``None``.

    Raises:
        ValueError: If the first dimension of a 2-D input does not match
            either *num_heads* or *num_kv_heads*.
        ValueError: If the input has more than two dimensions.
    """
    if softmax_aux is None:
        return None

    aux = jnp.asarray(softmax_aux, dtype=jnp.float32)
    if aux.ndim == 1:
        if aux.shape[0] == num_heads:
            return aux[:, None]
        if aux.shape[0] == num_kv_heads:
            return jnp.repeat(aux, repeats=(num_heads // num_kv_heads), axis=0)[:, None]
        return jnp.broadcast_to(aux[None, :], (num_heads, aux.shape[0]))

    if aux.ndim == 2:
        if aux.shape[0] == num_heads:
            return aux
        if aux.shape[0] == num_kv_heads:
            return jnp.repeat(aux, repeats=(num_heads // num_kv_heads), axis=0)
        raise ValueError(
            f"softmax_aux first dim must be num_kv_heads ({num_kv_heads}) or num_heads ({num_heads}); got {aux.shape[0]}"
        )
    raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux.shape}")


def _build_token_mask(
    *,
    q_positions: Int[Array, "batch q_len"],
    q_segment_ids: Int[Array, "batch q_len"],
    kv_positions: Int[Array, "batch kv_len"],
    kv_segment_ids: Int[Array, "batch kv_len"],
    qkv_layouts: tuple[SparseMask] | None,
    q_blocksize: int,
    kv_blocksize: int,
    causal: bool,
    window_left: int,
    window_right: int,
) -> Bool[Array, "batch q_len kv_len"]:
    """Build a dense boolean attention mask for the backward pass.

    Constructs a ``(batch, q_len, kv_len)`` boolean mask that encodes which
    query-key token pairs may attend to each other, replicating the masking
    semantics of the CUDA forward kernel. The mask is the intersection of:

    1. **Block sparsity** (if *qkv_layouts* is provided): a token pair is
       allowed if the KV block index falls within the ``[lower_bound,
       upper_bound)`` range for the corresponding query block.
    2. **Segment IDs**: query and key tokens must share the same segment ID.
    3. **Causal mask** (if *causal*): ``q_position >= kv_position``.
    4. **Sliding window** (if ``window_left >= 0`` or ``window_right >= 0``):
       token distance is bounded by the window limits.

    Args:
        q_positions: Integer position indices for queries, shape
            ``(batch, q_len)``.
        q_segment_ids: Segment IDs for queries, shape ``(batch, q_len)``.
        kv_positions: Integer position indices for keys/values, shape
            ``(batch, kv_len)``.
        kv_segment_ids: Segment IDs for keys/values, shape
            ``(batch, kv_len)``.
        qkv_layouts: Block-level sparsity layout; only the first element's
            ``lower_bounds`` and ``upper_bounds`` are used. ``None`` skips
            block-sparsity masking.
        q_blocksize: Query block size used to map token indices to block
            indices.
        kv_blocksize: Key/value block size for the same purpose.
        causal: Whether causal masking is applied.
        window_left: Maximum allowed left distance (``q_pos - kv_pos``).
            ``-1`` disables the left window constraint.
        window_right: Maximum allowed right distance. ``-1`` disables the
            right window constraint.

    Returns:
        Boolean mask of shape ``(batch, q_len, kv_len)`` where ``True``
        means the token pair is allowed to attend.
    """
    batch_size, q_len = q_positions.shape
    _, kv_len = kv_positions.shape

    valid = jnp.ones((batch_size, q_len, kv_len), dtype=bool)

    if qkv_layouts is not None and len(qkv_layouts) > 0 and qkv_layouts[0].lower_bounds is not None:
        layout = qkv_layouts[0]
        lower = jnp.asarray(layout.lower_bounds, dtype=jnp.int32)
        upper = jnp.asarray(layout.upper_bounds, dtype=jnp.int32)
        if lower.ndim == 2:
            lower = lower[:, None, :]
            upper = upper[:, None, :]
        q_block_idx = (jnp.arange(q_len, dtype=jnp.int32) // int(q_blocksize)).astype(jnp.int32)
        kv_block_idx = (jnp.arange(kv_len, dtype=jnp.int32) // int(kv_blocksize)).astype(jnp.int32)
        lb_tok = lower[:, 0, q_block_idx]
        ub_tok = upper[:, 0, q_block_idx]
        valid = (
            valid
            & (kv_block_idx[None, None, :] >= lb_tok[:, :, None])
            & (kv_block_idx[None, None, :] < ub_tok[:, :, None])
        )

    valid = valid & (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :])

    if causal:
        valid = valid & (q_positions[:, :, None] >= kv_positions[:, None, :])

    if window_left >= 0 or window_right >= 0:
        dist = q_positions[:, :, None] - kv_positions[:, None, :]
        if window_left >= 0:
            valid = valid & (dist <= int(window_left))
        if window_right >= 0:
            valid = valid & (dist >= -int(window_right))

    return valid


def _blocksparse_dense_backward(
    *,
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    q_positions: Int[Array, "batch q_len"],
    q_segment_ids: Int[Array, "batch q_len"],
    kv_positions: Int[Array, "batch kv_len"],
    kv_segment_ids: Int[Array, "batch kv_len"],
    qkv_layouts: tuple[SparseMask] | None,
    softmax_scale: float,
    softmax_aux: ArrayLike | None,
    window_left: int,
    window_right: int,
    causal: bool,
    logits_soft_cap: float | None,
    q_blocksize: int,
    kv_blocksize: int,
    dout: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Compute dense gradients for block-sparse attention.

    Implements the full analytical backward pass in JAX (no custom CUDA kernel).
    Dequantizes/replicates KV heads for GQA, reconstructs the attention
    probability matrix with block-sparse masking and optional soft-capping,
    then applies the standard softmax VJP to yield ``dq``, ``dk``, and ``dv``.

    All computations are performed in ``float32`` regardless of the input
    dtypes; the output gradients are cast back to the original dtypes before
    returning.

    Note:
        When *softmax_aux* is provided (attention-sink mode), a small set of
        learned sink-token logits is appended to the standard logits before
        the softmax and their gradients are discarded (not returned).

    Args:
        query: Query tensor of shape ``(batch, num_heads, q_len, head_dim)``.
        key: Key tensor of shape ``(batch, num_kv_heads, kv_len, head_dim)``.
        value: Value tensor of shape
            ``(batch, num_kv_heads, kv_len, v_head_dim)``.
        q_positions: Integer position indices for queries, shape
            ``(batch, q_len)``.
        q_segment_ids: Segment IDs for queries, shape ``(batch, q_len)``.
        kv_positions: Integer position indices for keys/values, shape
            ``(batch, kv_len)``.
        kv_segment_ids: Segment IDs for keys/values, shape
            ``(batch, kv_len)``.
        qkv_layouts: Block-level sparsity layouts or ``None``.
        softmax_scale: Scaling factor applied to raw dot-product logits.
        softmax_aux: Optional attention-sink auxiliary weights; passed to
            :func:`_normalize_softmax_aux`.
        window_left: Left sliding-window size (``-1`` to disable).
        window_right: Right sliding-window size (``-1`` to disable).
        causal: Whether causal masking is applied.
        logits_soft_cap: Optional soft-cap for logits (``tanh``-based).
            ``None`` disables capping.
        q_blocksize: Query block size for mask reconstruction.
        kv_blocksize: Key/value block size for mask reconstruction.
        dout: Upstream gradient of the attention output, shape
            ``(batch, num_heads, q_len, v_head_dim)``.

    Returns:
        A 3-tuple ``(dq, dk, dv)`` of gradients:

        * **dq** -- shape ``(batch, num_heads, q_len, head_dim)``,
          same dtype as *query*.
        * **dk** -- shape ``(batch, num_kv_heads, kv_len, head_dim)``,
          same dtype as *key*.
        * **dv** -- shape ``(batch, num_kv_heads, kv_len, v_head_dim)``,
          same dtype as *value*.

    Raises:
        ValueError: If *num_heads* is not divisible by *num_kv_heads*.
    """
    q_f = jnp.asarray(query, dtype=jnp.float32)
    k_f = jnp.asarray(key, dtype=jnp.float32)
    v_f = jnp.asarray(value, dtype=jnp.float32)
    do_f = jnp.asarray(dout, dtype=jnp.float32)

    batch_size, num_heads, q_len, _ = q_f.shape
    _, num_kv_heads, kv_len, _ = k_f.shape
    repeats = num_heads // num_kv_heads
    if repeats * num_kv_heads != num_heads:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}) in CUDA blocksparse backward."
        )

    k_full = jnp.repeat(k_f, repeats=repeats, axis=1)
    v_full = jnp.repeat(v_f, repeats=repeats, axis=1)

    raw_logits = jnp.einsum("bhqd,bhkd->bhqk", q_f, k_full)
    if logits_soft_cap is not None and logits_soft_cap > 0:
        cap = float(logits_soft_cap)
        scaled = raw_logits * softmax_scale / cap
        logits = cap * jnp.tanh(scaled)
        dlogits_draw = softmax_scale * (1.0 - jnp.tanh(scaled) ** 2)
    else:
        logits = raw_logits * softmax_scale
        dlogits_draw = softmax_scale

    token_mask = _build_token_mask(
        q_positions=jnp.asarray(q_positions, dtype=jnp.int32),
        q_segment_ids=jnp.asarray(q_segment_ids, dtype=jnp.int32),
        kv_positions=jnp.asarray(kv_positions, dtype=jnp.int32),
        kv_segment_ids=jnp.asarray(kv_segment_ids, dtype=jnp.int32),
        qkv_layouts=qkv_layouts,
        q_blocksize=q_blocksize,
        kv_blocksize=kv_blocksize,
        causal=causal,
        window_left=window_left,
        window_right=window_right,
    )
    logits = jnp.where(token_mask[:, None, :, :], logits, -jnp.inf)

    aux = _normalize_softmax_aux(softmax_aux, num_heads=num_heads, num_kv_heads=num_kv_heads)
    if aux is None:
        probs_ext = jax.nn.softmax(logits, axis=-1)
        probs_ext = jnp.where(jnp.isfinite(probs_ext), probs_ext, 0.0)
        probs = probs_ext
        dprobs_ext = jnp.einsum("bhqv,bhkv->bhqk", do_f, v_full)
    else:
        aux_logits = jnp.broadcast_to(aux[None, :, None, :], (batch_size, num_heads, q_len, aux.shape[-1]))
        logits_ext = jnp.concatenate([logits, aux_logits], axis=-1)
        probs_ext = jax.nn.softmax(logits_ext, axis=-1)
        probs_ext = jnp.where(jnp.isfinite(probs_ext), probs_ext, 0.0)
        probs = probs_ext[..., :kv_len]
        dprobs = jnp.einsum("bhqv,bhkv->bhqk", do_f, v_full)
        dprobs_ext = jnp.concatenate([dprobs, jnp.zeros_like(aux_logits)], axis=-1)

    dv_full = jnp.einsum("bhqk,bhqv->bhkv", probs, do_f)

    softmax_dot = jnp.sum(dprobs_ext * probs_ext, axis=-1, keepdims=True)
    dlogits_ext = probs_ext * (dprobs_ext - softmax_dot)
    dlogits = dlogits_ext[..., :kv_len]
    dlogits = jnp.where(token_mask[:, None, :, :], dlogits, 0.0)

    draw = dlogits * dlogits_draw
    dq = jnp.einsum("bhqk,bhkd->bhqd", draw, k_full)
    dk_full = jnp.einsum("bhqk,bhqd->bhkd", draw, q_f)

    dk = jnp.sum(dk_full.reshape(batch_size, num_kv_heads, repeats, kv_len, key.shape[-1]), axis=2)
    dv = jnp.sum(dv_full.reshape(batch_size, num_kv_heads, repeats, kv_len, value.shape[-1]), axis=2)

    return dq.astype(query.dtype), dk.astype(key.dtype), dv.astype(value.dtype)


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=[8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "softmax_scale",
        "apply_load_balance",
        "sequence_parallelism_mesh_axis_name",
        "window_left",
        "window_right",
        "causal",
        "fwd_q_blocksize",
        "fwd_kv_blocksize",
        "fwd_num_stages",
        "fwd_num_warps",
        "bwd_q_blocksize",
        "bwd_kv_blocksize",
        "bwd_num_stages",
        "bwd_num_warps",
        "logits_soft_cap",
    ],
)
def _blocksparse_attention_bhtd(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    q_positions: ArrayLike,
    q_segment_ids: ArrayLike,
    kv_positions: ArrayLike,
    kv_segment_ids: ArrayLike,
    qkv_layouts: tuple[SparseMask] | None = None,
    softmax_scale: float = 1.0,
    softmax_aux: ArrayLike | None = None,
    bias: ArrayLike | None = None,
    apply_load_balance: bool = True,
    sequence_parallelism_mesh_axis_name: str | None = None,
    window_left: int = -1,
    window_right: int = -1,
    causal: bool = True,
    fwd_q_blocksize: int = 64,
    fwd_kv_blocksize: int = 64,
    fwd_num_stages: int | None = 2,
    fwd_num_warps: int | None = 4,
    bwd_q_blocksize: int = 32,
    bwd_kv_blocksize: int = 32,
    bwd_num_stages: int | None = 2,
    bwd_num_warps: int | None = 4,
    logits_soft_cap: float | None = None,
) -> ArrayLike:
    """Compute block-sparse attention in BHTD layout with a custom VJP rule.

    This is the core differentiable primitive. The forward evaluation
    delegates to the CUDA kernel, while JAX's custom VJP machinery routes
    gradient computation through :func:`_blocksparse_attention_bhtd_fwd`
    and :func:`_blocksparse_attention_bhtd_bwd`.

    Args:
        query: Query tensor, shape ``(batch, num_heads, q_len, head_dim)``.
        key: Key tensor, shape ``(batch, num_kv_heads, kv_len, head_dim)``.
        value: Value tensor, shape
            ``(batch, num_kv_heads, kv_len, v_head_dim)``.
        q_positions: Query position indices, shape ``(batch, q_len)``.
        q_segment_ids: Query segment identifiers, shape ``(batch, q_len)``.
        kv_positions: Key/value position indices, shape ``(batch, kv_len)``.
        kv_segment_ids: Key/value segment identifiers, shape
            ``(batch, kv_len)``.
        qkv_layouts: Block-level sparsity layouts.
        softmax_scale: Logit scaling factor.
        softmax_aux: Optional attention-sink auxiliary weights.
        bias: Explicit additive bias (not supported; raises if provided).
        apply_load_balance: Unused; reserved for API compatibility.
        sequence_parallelism_mesh_axis_name: Unused; reserved for API
            compatibility.
        window_left: Left sliding-window size (``-1`` to disable).
        window_right: Right sliding-window size (``-1`` to disable).
        causal: Whether to apply causal masking.
        fwd_q_blocksize: Forward query block size.
        fwd_kv_blocksize: Forward key/value block size.
        fwd_num_stages: Forward kernel stages.
        fwd_num_warps: Forward kernel warps.
        bwd_q_blocksize: Backward query block size (unused in forward).
        bwd_kv_blocksize: Backward key/value block size (unused in forward).
        bwd_num_stages: Backward kernel stages (unused in forward).
        bwd_num_warps: Backward kernel warps (unused in forward).
        logits_soft_cap: Optional soft-cap for attention logits.

    Returns:
        Attention output of shape
        ``(batch, num_heads, q_len, v_head_dim)``.

    Raises:
        NotImplementedError: If *bias* is not ``None``.
    """
    del (
        apply_load_balance,
        sequence_parallelism_mesh_axis_name,
        bwd_q_blocksize,
        bwd_kv_blocksize,
        bwd_num_stages,
        bwd_num_warps,
    )

    if bias is not None:
        raise NotImplementedError("Bias is not supported in CUDA block-sparse attention.")
    fwd_params = _build_fwd_params(fwd_q_blocksize, fwd_kv_blocksize, fwd_num_stages, fwd_num_warps)

    return blocksparse_attention_cuda(
        query=query,
        key=key,
        value=value,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        fwd_params=fwd_params,
        logits_soft_cap=logits_soft_cap,
    )


def _blocksparse_attention_bhtd_fwd(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    q_positions: ArrayLike,
    q_segment_ids: ArrayLike,
    kv_positions: ArrayLike,
    kv_segment_ids: ArrayLike,
    qkv_layouts: tuple[SparseMask] | None,
    softmax_scale: float,
    softmax_aux: ArrayLike | None,
    bias: ArrayLike | None,
    apply_load_balance: bool,
    sequence_parallelism_mesh_axis_name: str | None,
    window_left: int,
    window_right: int,
    causal: bool,
    fwd_q_blocksize: int,
    fwd_kv_blocksize: int,
    fwd_num_stages: int | None,
    fwd_num_warps: int | None,
    bwd_q_blocksize: int,
    bwd_kv_blocksize: int,
    bwd_num_stages: int | None,
    bwd_num_warps: int | None,
    logits_soft_cap: float | None,
):
    """Forward pass for the custom VJP of block-sparse attention.

    Runs the CUDA forward kernel and returns both the output and a residuals
    tuple that the backward pass (:func:`_blocksparse_attention_bhtd_bwd`)
    will consume.

    Args:
        query: Query tensor, shape ``(batch, num_heads, q_len, head_dim)``.
        key: Key tensor, shape ``(batch, num_kv_heads, kv_len, head_dim)``.
        value: Value tensor, shape
            ``(batch, num_kv_heads, kv_len, v_head_dim)``.
        q_positions: Query position indices.
        q_segment_ids: Query segment identifiers.
        kv_positions: Key/value position indices.
        kv_segment_ids: Key/value segment identifiers.
        qkv_layouts: Block-level sparsity layouts.
        softmax_scale: Logit scaling factor.
        softmax_aux: Optional attention-sink auxiliary weights.
        bias: Explicit additive bias (not supported).
        apply_load_balance: Unused; reserved for API compatibility.
        sequence_parallelism_mesh_axis_name: Unused; reserved for API
            compatibility.
        window_left: Left sliding-window size.
        window_right: Right sliding-window size.
        causal: Whether to apply causal masking.
        fwd_q_blocksize: Forward query block size.
        fwd_kv_blocksize: Forward key/value block size.
        fwd_num_stages: Forward kernel stages.
        fwd_num_warps: Forward kernel warps.
        bwd_q_blocksize: Backward query block size.
        bwd_kv_blocksize: Backward key/value block size.
        bwd_num_stages: Backward kernel stages.
        bwd_num_warps: Backward kernel warps.
        logits_soft_cap: Optional soft-cap for attention logits.

    Returns:
        A tuple ``(out, res)`` where *out* is the attention output tensor
        and *res* is a tuple of values saved for the backward pass.

    Raises:
        NotImplementedError: If *bias* is not ``None``.
    """
    del apply_load_balance, sequence_parallelism_mesh_axis_name

    if bias is not None:
        raise NotImplementedError("Bias is not supported in CUDA block-sparse attention.")
    fwd_params = _build_fwd_params(fwd_q_blocksize, fwd_kv_blocksize, fwd_num_stages, fwd_num_warps)

    out = blocksparse_attention_cuda(
        query=query,
        key=key,
        value=value,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        fwd_params=fwd_params,
        logits_soft_cap=logits_soft_cap,
    )

    res = (
        query,
        key,
        value,
        q_positions,
        q_segment_ids,
        kv_positions,
        kv_segment_ids,
        qkv_layouts,
        softmax_scale,
        softmax_aux,
        window_left,
        window_right,
        causal,
        logits_soft_cap,
        fwd_q_blocksize,
        fwd_kv_blocksize,
        fwd_num_stages,
        fwd_num_warps,
        bwd_q_blocksize,
        bwd_kv_blocksize,
        bwd_num_stages,
        bwd_num_warps,
    )
    return out, res


def _blocksparse_attention_bhtd_bwd(
    softmax_scale: float,
    apply_load_balance: bool,
    sequence_parallelism_mesh_axis_name: str | None,
    window_left: int,
    window_right: int,
    causal: bool,
    fwd_q_blocksize: int,
    fwd_kv_blocksize: int,
    fwd_num_stages: int | None,
    fwd_num_warps: int | None,
    bwd_q_blocksize: int,
    bwd_kv_blocksize: int,
    bwd_num_stages: int | None,
    bwd_num_warps: int | None,
    logits_soft_cap: float | None,
    res,
    dout: ArrayLike,
):
    """Backward pass for the custom VJP of block-sparse attention.

    Unpacks the residual tuple produced by
    :func:`_blocksparse_attention_bhtd_fwd` and delegates gradient
    computation to :func:`_blocksparse_dense_backward`. Returns gradients
    only for *query*, *key*, and *value*; all other differentiable inputs
    receive ``None``.

    Note:
        ``bwd_q_blocksize``, ``bwd_kv_blocksize``, ``bwd_num_stages``,
        ``bwd_num_warps``, ``fwd_num_stages``, ``fwd_num_warps``,
        ``apply_load_balance``, and
        ``sequence_parallelism_mesh_axis_name`` are all silently discarded
        in the backward path. The backward uses *fwd_q_blocksize* and
        *fwd_kv_blocksize* for block-sparsity mask reconstruction.

    Args:
        softmax_scale: Logit scaling factor (received as a non-differentiable
            argument via ``nondiff_argnums``).
        apply_load_balance: Unused; reserved for API compatibility.
        sequence_parallelism_mesh_axis_name: Unused; reserved for API
            compatibility.
        window_left: Left sliding-window size (non-differentiable).
        window_right: Right sliding-window size (non-differentiable).
        causal: Whether causal masking is applied (non-differentiable).
        fwd_q_blocksize: Forward query block size (used for mask
            reconstruction in the dense backward).
        fwd_kv_blocksize: Forward key/value block size.
        fwd_num_stages: Unused; forward pipeline stages.
        fwd_num_warps: Unused; forward warp count.
        bwd_q_blocksize: Unused in the backward path.
        bwd_kv_blocksize: Unused in the backward path.
        bwd_num_stages: Unused in the backward path.
        bwd_num_warps: Unused in the backward path.
        logits_soft_cap: Optional soft-cap for attention logits
            (non-differentiable).
        res: Residuals tuple saved by
            :func:`_blocksparse_attention_bhtd_fwd`. Elements at indices
            0--9 are: query, key, value, q_positions, q_segment_ids,
            kv_positions, kv_segment_ids, qkv_layouts, softmax_scale
            (float), softmax_aux.
        dout: Upstream gradient tensor of the same shape as the forward
            output ``(batch, num_heads, q_len, v_head_dim)``.

    Returns:
        A 10-tuple ``(dq, dk, dv, None, None, None, None, None, None, None)``
        where only the first three elements are non-``None`` gradients for
        *query*, *key*, and *value*.
    """
    del (
        apply_load_balance,
        sequence_parallelism_mesh_axis_name,
        fwd_num_stages,
        fwd_num_warps,
        bwd_q_blocksize,
        bwd_kv_blocksize,
        bwd_num_stages,
        bwd_num_warps,
    )

    query = res[0]
    key = res[1]
    value = res[2]
    q_positions = res[3]
    q_segment_ids = res[4]
    kv_positions = res[5]
    kv_segment_ids = res[6]
    qkv_layouts = res[7]
    out = res[8]
    softmax_aux = res[9]

    dout = ad.instantiate_zeros(dout)
    if dout is None:
        dout = jnp.zeros_like(out)

    dq, dk, dv = _blocksparse_dense_backward(
        query=query,
        key=key,
        value=value,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        q_blocksize=fwd_q_blocksize,
        kv_blocksize=fwd_kv_blocksize,
        dout=dout,
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
    )


_blocksparse_attention_bhtd.defvjp(_blocksparse_attention_bhtd_fwd, _blocksparse_attention_bhtd_bwd)


@kernel_registry.register("blocksparse_attention", Platform.CUDA, Backend.GPU)
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
@jaxtyping.jaxtyped(typechecker=beartype)
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
    """Compute block-sparse multi-head attention on a CUDA GPU.

    This is the high-level entry point registered in the ejkernel kernel
    registry for the ``CUDA`` platform / ``GPU`` backend.  It validates and
    defaults all parameters, constructs sparsity layouts if they are not
    supplied, and delegates to the CUDA forward kernel.

    Grouped-query attention (GQA) is supported: *key* and *value* may have
    fewer heads than *query*, and the kernel will broadcast internally.

    Args:
        query: Query tensor of shape
            ``(batch, num_heads, seq_len, head_dim)``.
        key: Key tensor of shape
            ``(batch, kv_num_heads, kv_len, head_dim)``.
        value: Value tensor of shape
            ``(batch, kv_num_heads, kv_len, vhead_dim)``.
        q_segment_ids: Segment IDs for the query sequence, shape
            ``(batch, seq_len)``.  Tokens with different segment IDs do not
            attend to each other.  Defaults to all-ones.
        kv_segment_ids: Segment IDs for the key/value sequence, shape
            ``(batch, kv_len)``.  Defaults to all-ones.
        q_positions: Integer position indices for queries, shape
            ``(batch, seq_len)``.  Defaults to ``arange(seq_len)``
            broadcast across the batch.
        kv_positions: Integer position indices for keys/values, shape
            ``(batch, kv_len)``.  Defaults to ``arange(kv_len)``.
        softmax_aux: Optional attention-sink auxiliary weights for the
            softmax computation.
        bias: Explicit additive attention bias.  **Not supported** by the
            CUDA kernel; passing a non-``None`` value raises
            :class:`NotImplementedError`.
        attention_mask: Optional dense boolean or integer attention mask of
            shape ``(batch, num_heads_or_1, seq_len, kv_len)``.  When
            segment IDs are not provided, the mask is converted to segment
            IDs internally.
        sequence_parallelism_mesh_axis_name: Mesh axis name for sequence
            parallelism (reserved; currently unused).
        logits_soft_cap: Optional soft-cap applied to attention logits
            before the softmax.  ``None`` disables capping.
        qkv_layouts: Pre-computed block-level sparsity layouts as a tuple
            of :class:`SparseMask` instances.  When ``None``, layouts are
            built automatically via *mask_builder* or
            :func:`create_sparsity_mask`.
        softmax_scale: Scaling factor for dot-product logits.  Defaults to
            ``head_dim ** -0.5``.
        fwd_params: Forward-pass kernel parameters (:class:`FwdParams`).
            Must specify ``q_blocksize`` and ``kv_blocksize``.  Defaults
            to ``FwdParams(q_blocksize=64, kv_blocksize=64, num_stages=2,
            num_warps=4)``.
        bwd_params: Backward-pass kernel parameters (:class:`BwdParams`).
            Defaults to ``BwdParams(q_blocksize=32, kv_blocksize=32,
            num_stages=2, num_warps=4)``.
        mask_builder: A callable that returns a :class:`SparseMask` (or
            compatible ``Mask`` object) when *qkv_layouts* is ``None``.
        sliding_window: Sliding-window size.  Can be a single ``int``
            (symmetric left/right window) or a ``(left, right)`` tuple.
            ``None`` disables windowing.
        chunk_size: Unused; reserved for API compatibility.
        causal: Whether to apply lower-triangular causal masking.
            Defaults to ``True``.
        fused_backward: Unused; reserved for API compatibility.

    Returns:
        The attention output tensor of shape
        ``(batch, num_heads, seq_len, vhead_dim)``.

    Raises:
        NotImplementedError: If *bias* is not ``None``.
        ValueError: If *fwd_params* does not specify ``q_blocksize`` and
            ``kv_blocksize``.
    """
    del fused_backward, chunk_size

    if bias is not None:
        raise NotImplementedError("Bias is not supported in CUDA block-sparse attention.")

    qlen = query.shape[2]
    kvlen = key.shape[2]

    if mask_builder is not None and qkv_layouts is None:
        qkv_layouts = mask_builder()
    if _TRITON_SPARSE_MASK_TYPE is not None and isinstance(qkv_layouts, _TRITON_SPARSE_MASK_TYPE):
        qkv_layouts = (qkv_layouts,)

    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=64, kv_blocksize=64, num_stages=2, num_warps=4)
    if bwd_params is None:
        bwd_params = BwdParams(q_blocksize=32, kv_blocksize=32, num_stages=2, num_warps=4)
    if fwd_params.q_blocksize is None or fwd_params.kv_blocksize is None:
        raise ValueError("CUDA blocksparse_attention requires q_blocksize and kv_blocksize in fwd_params.")

    if sliding_window is None:
        window_left = window_right = -1
    elif isinstance(sliding_window, int):
        window_left = window_right = sliding_window
    else:
        window_left, window_right = sliding_window

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    if q_positions is None:
        q_positions = jnp.arange(0, qlen).reshape(1, -1).repeat(query.shape[0], 0)
    if kv_positions is None:
        kv_positions = jnp.arange(0, kvlen).reshape(1, -1).repeat(key.shape[0], 0)

    if attention_mask is not None and (q_segment_ids is None or kv_segment_ids is None):
        from ejkernel.types.mask import mask_to_segment_ids

        inferred_q_seg, inferred_kv_seg = mask_to_segment_ids(attention_mask)
        if q_segment_ids is None:
            q_segment_ids = inferred_q_seg
        if kv_segment_ids is None:
            kv_segment_ids = inferred_kv_seg

    if q_segment_ids is None:
        q_segment_ids = jnp.ones_like(q_positions)
    if kv_segment_ids is None:
        kv_segment_ids = jnp.ones_like(kv_positions)

    if qkv_layouts is None:
        qkv_layouts = create_sparsity_mask(
            q_blocksize=fwd_params.q_blocksize,
            kv_blocksize=fwd_params.kv_blocksize,
            kv_positions=kv_positions,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            q_segment_ids=q_segment_ids,
            causal=causal,
            window_left=window_left,
            window_right=window_right,
        )

    return _blocksparse_attention_bhtd(
        query=query,
        key=key,
        value=value,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
        bias=bias,
        apply_load_balance=True,
        sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        fwd_q_blocksize=int(fwd_params.q_blocksize),
        fwd_kv_blocksize=int(fwd_params.kv_blocksize),
        fwd_num_stages=None if fwd_params.num_stages is None else int(fwd_params.num_stages),
        fwd_num_warps=None if fwd_params.num_warps is None else int(fwd_params.num_warps),
        bwd_q_blocksize=int(bwd_params.q_blocksize),
        bwd_kv_blocksize=int(bwd_params.kv_blocksize),
        bwd_num_stages=None if bwd_params.num_stages is None else int(bwd_params.num_stages),
        bwd_num_warps=None if bwd_params.num_warps is None else int(bwd_params.num_warps),
        logits_soft_cap=logits_soft_cap,
    )
