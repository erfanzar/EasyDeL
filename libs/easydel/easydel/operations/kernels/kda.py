# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""
Kernel Delta Attention (KDA) linear attention implementation for EasyDeL.

This module provides the KDA operation used in Kimi Linear models (moonshotai).
KDA is a variant of the gated delta rule with:

1. Separate convolutions for Q, K, V
2. Different decay gate computation using A_log and dt_bias
3. Two-layer MLP for decay gate (f_a_proj -> f_b_proj)

Key characteristics:
- Linear complexity O(N) in sequence length
- Maintains recurrent state for efficient inference
- Supports chunked computation for efficient training
- Compatible with HybridCache for state management

The algorithm:
    Training (chunked):
        - Process sequence in chunks for parallelism
        - Intra-chunk: parallel computation within each chunk
        - Inter-chunk: sequential state propagation via scan

    Inference (recurrent):
        - Single-step state update
        - h_t = decay * h_{t-1} + beta_t * (v_t ⊗ k_t)
        - o_t = h_t @ q_t

References:
    - Kimi Linear: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
    - FLA: https://github.com/fla-org/flash-linear-attention
"""

import jax
import jax.numpy as jnp
import numpy as np
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from ejkernel.kernels import Backend, Platform, kernel_registry  # pyright: ignore[reportMissingTypeStubs]
from ejkernel.modules.operations import KdaChunkConfig  # pyright: ignore[reportMissingTypeStubs]
from ejkernel.modules.operations import kda_chunk as ejkernel_kda_chunk  # pyright: ignore[reportMissingTypeStubs]
from jax import lax
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float
from spectrax import sanitize_partition_spec_for_mesh_and_shape, with_sharding_constraint

from easydel.caching import KDACacheView
from easydel.infra.sequence_packing import normalize_packed_segment_ids

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry, run_mosaic_kernel
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)

logger = get_logger(__name__)

_MATMUL_PRECISION = lax.Precision.HIGHEST


def l2norm(x, axis=-1, eps=1e-6):
    """Normalize ``x`` to unit L2 norm along ``axis``.

    Implementation uses ``rsqrt`` for numerical stability:
    ``inv_norm = rsqrt(sum(x*x, axis=axis) + eps)`` and
    ``x_normalized = x * inv_norm``. The KDA forward pass invokes this
    on Q and K to keep the recurrent state norms bounded across long
    decode loops.

    Args:
        x: Input tensor; any shape and floating dtype is accepted.
        axis: Reduction axis. Defaults to the trailing axis.
        eps: Stability epsilon mixed in before ``rsqrt`` to bound the
            output magnitude on zero-magnitude rows.

    Returns:
        jnp.ndarray: Same shape and dtype as ``x``, with elements along
        ``axis`` summing in squared magnitude to approximately 1.
    """
    inv_norm = lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def fused_kda_gate(gate: Float[Array, "..."], A_log: Float[Array, "num_heads"], dt_bias: Float[Array, "num_heads"]):
    """Materialise the per-token KDA log-decay from raw f_b / A_log / dt_bias.

    Implements the fused expression
    ``decay = -exp(A_log) * softplus(gate + dt_bias)`` used by the Kimi
    Linear / KDA recurrence. The negative ``-exp(A_log)`` factor keeps
    the recurrent state contractive; the softplus ensures the
    per-token rate is strictly positive even when ``gate + dt_bias``
    goes negative.

    All arithmetic is performed in float32 and the output keeps that
    dtype so the downstream ``exp(decay)`` stays numerically stable
    even for long sequences. Shapes use broadcasting: ``A_log`` and
    ``dt_bias`` are per-head ``(num_heads,)``, ``gate`` is
    ``(batch, seq, num_heads)``.

    Args:
        gate: Per-token gate from the ``f_b_proj`` head, shape
            ``(batch, seq, num_heads)``.
        A_log: Per-head log-decay parameter, shape ``(num_heads,)``.
        dt_bias: Per-head time-discretisation bias, shape
            ``(num_heads,)``.

    Returns:
        Float[Array, "batch seq num_heads"]: The float32 log-decay
        ready to be exponentiated inside the KDA state update.
    """
    A = -jnp.exp(A_log.astype(jnp.float32))
    gate = gate.astype(jnp.float32)
    dt_bias = dt_bias.astype(jnp.float32)
    # softplus(gate + dt_bias)
    gate_biased = gate + dt_bias
    # Overflow-safe softplus (log1p(exp(x)) is inf for x > ~88 and NaN in the gradient).
    softplus_gate = jax.nn.softplus(gate_biased)
    decay = A * softplus_gate
    return decay


@auto_pytree
class KDAOutput(AttentionOutput):
    """Output container for KDA operation.

    Extends AttentionOutput with recurrent state fields needed for
    hybrid attention models.

    Attributes:
        attention_outputs: Output tensor [batch, seq_len, num_heads, head_dim]
        attention_weights: Always None for linear attention (no explicit weights)
        q_conv_state: Updated Q convolution state [batch, key_dim, d_conv]
        k_conv_state: Updated K convolution state [batch, key_dim, d_conv]
        v_conv_state: Updated V convolution state [batch, value_dim, d_conv]
        recurrent_state: Updated recurrent state [batch, num_heads, head_dim, d_state]
    """

    q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None
    k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None
    v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None
    recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None


def _recurrent_kda_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
    segment_ids: Array | None = None,
    continue_initial_state: bool = False,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Sequential per-head-decay KDA recurrence over a full sequence.

    Thin wrapper over :func:`_recurrent_kda_per_channel_fwd` (the reference
    path used by :func:`_chunk_kda_fwd` and as a fallback for tiny sequences
    where chunking adds overhead): the per-head log-decay is broadcast with a
    trailing singleton channel axis, ``decay[..., None]``, so the shared
    per-channel scan applies ``exp(g)``
    per head. The bodies are identical except that broadcast axis. All math
    is done in float32.

    Args:
        query: ``(batch, num_heads, seq_len, head_dim)``.
        key: ``(batch, num_heads, seq_len, head_dim)``.
        value: ``(batch, num_heads, seq_len, d_state)``.
        beta: Per-token gating coefficient,
            ``(batch, num_heads, seq_len)``.
        decay: Per-token log-decay, ``(batch, num_heads, seq_len)``;
            pass ``None`` to disable decay (equivalent to all-zeros).
        initial_state: Optional starting recurrent state,
            ``(batch, num_heads, head_dim, d_state)``. Defaults to
            zeros.
        use_qk_l2norm: Whether to L2-normalize ``query`` and ``key``
            before the loop.
        segment_ids: Optional packed-sequence ids,
            ``(batch, seq_len)``; state resets at segment boundaries and
            padded steps emit zero outputs.
        continue_initial_state: Preserve incoming memory into the first live segment
            and retain memory over padding. Later segment changes still reset state.
            False preserves the packed-training first-segment reset/padding behavior.

    Returns:
        tuple: ``(outputs, final_state)`` where ``outputs`` has shape
        ``(batch, num_heads, seq_len, d_state)`` and ``final_state``
        has shape ``(batch, num_heads, head_dim, d_state)``.
    """
    batch, num_heads, seq_len = beta.shape
    if decay is None:
        # The per-head contract zero-fills a missing decay (state carried
        # forward untouched); the per-channel kernel requires an explicit g.
        decay = jnp.zeros((batch, num_heads, seq_len), dtype=jnp.float32)
    return _recurrent_kda_per_channel_fwd(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay[..., None],
        initial_state=initial_state,
        use_qk_l2norm=use_qk_l2norm,
        segment_ids=segment_ids,
        continue_initial_state=continue_initial_state,
    )


def _chunk_kda_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Chunked KDA forward pass for training / long-context decode.

    Implements the FLA-style two-level decomposition: each contiguous
    block of ``chunk_size`` tokens is processed in parallel via the
    intra-chunk attention pattern (computed at ``Precision.HIGHEST``),
    while the recurrent state is propagated *between* chunks using
    ``lax.scan``. The intra-chunk path solves a unit lower-triangular
    system to mix the gated delta updates inside the chunk; the
    inter-chunk path applies the cumulative-decay matmul plus the
    delta correction.

    The sequence is right-padded to a multiple of ``chunk_size``; the
    pad is stripped from the final output before returning. All math
    runs in float32 internally; outputs are returned in float32 too -
    callers cast back to runtime dtype as needed.

    Args:
        query: ``(batch, num_heads, seq_len, head_dim)``.
        key: ``(batch, num_heads, seq_len, head_dim)``.
        value: ``(batch, num_heads, seq_len, d_state)``.
        beta: Per-token gating, ``(batch, num_heads, seq_len)``.
        decay: Per-token log-decay, ``(batch, num_heads, seq_len)``;
            pass ``None`` to disable decay.
        chunk_size: Chunk granularity used for the intra-chunk parallel
            kernel. Must divide the (padded) sequence length.
        initial_state: Optional starting recurrent state,
            ``(batch, num_heads, head_dim, d_state)``.
        use_qk_l2norm: Whether to L2-normalize ``query`` and ``key``
            before processing.

    Returns:
        tuple: ``(outputs, final_state)`` with outputs of shape
        ``(batch, num_heads, seq_len, d_state)`` (the padding is
        sliced away) and final state of shape
        ``(batch, num_heads, head_dim, d_state)``.
    """
    B, H, L, K_dim = query.shape
    V_dim = value.shape[-1]

    if use_qk_l2norm:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)

    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    if decay is None:
        decay = jnp.zeros((B, H, L), dtype=jnp.float32)
    else:
        decay = decay.astype(jnp.float32)

    pad_size = (chunk_size - L % chunk_size) % chunk_size
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        decay = jnp.pad(decay, ((0, 0), (0, 0), (0, pad_size)))

    total_len = L + pad_size
    num_chunks = total_len // chunk_size

    scale = 1.0 / (K_dim**0.5)
    query = query * scale

    v_beta = value * beta[:, :, :, None]
    k_beta = key * beta[:, :, :, None]

    query = query.reshape(B, H, num_chunks, chunk_size, K_dim)
    key = key.reshape(B, H, num_chunks, chunk_size, K_dim)
    value = value.reshape(B, H, num_chunks, chunk_size, V_dim)
    k_beta = k_beta.reshape(B, H, num_chunks, chunk_size, K_dim)
    v_beta = v_beta.reshape(B, H, num_chunks, chunk_size, V_dim)
    g = decay.reshape(B, H, num_chunks, chunk_size)

    mask_triu = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)

    g_cumsum = jnp.cumsum(g, axis=-1)

    g_diff = g_cumsum[:, :, :, :, None] - g_cumsum[:, :, :, None, :]  # (B, H, C, cs, cs)
    g_diff = jnp.tril(g_diff)
    decay_mask = jnp.exp(g_diff)
    decay_mask = jnp.tril(decay_mask)

    attn = jnp.einsum("bhcik,bhcjk->bhcij", k_beta, key, precision=_MATMUL_PRECISION)
    attn = -(attn * decay_mask)
    attn = jnp.where(mask_triu, 0.0, attn)

    def resolve_intra_chunk_row(attn_chunk, i):
        """Resolve the i-th row of an intra-chunk attention block in place.

        Implements the partial-sum update used by the KDA chunked
        algorithm: each row is augmented with contributions from earlier
        rows so the final ``(I + L)`` lower-triangular system is solved
        iteratively without an explicit triangular solve.

        Args:
            attn_chunk: ``(chunk_size, chunk_size)`` float matrix being
                resolved row-by-row inside :func:`lax.scan`.
            i: Current row index (scalar tracer fed by ``lax.scan``).

        Returns:
            tuple: ``(updated_attn_chunk, None)`` where the carry is the
            chunk after the i-th row has been updated and the scan output
            is unused.
        """
        row = attn_chunk[i, :]
        idx = jnp.arange(chunk_size)
        mask_lt_i = idx < i
        contribution = jnp.sum(row[:, None] * attn_chunk * mask_lt_i[:, None] * mask_lt_i[None, :], axis=0)

        new_row = row + contribution
        new_row = jnp.where(mask_lt_i, new_row, row)

        return attn_chunk.at[i].set(new_row), None

    def resolve_single_chunk(attn_single):
        """Run :func:`resolve_intra_chunk_row` over every row of one chunk.

        Args:
            attn_single: ``(chunk_size, chunk_size)`` matrix for a single
                ``(batch, head, chunk)`` triple.

        Returns:
            jax.Array: Same-shaped matrix after all rows have been resolved.
        """
        resolved, _ = lax.scan(resolve_intra_chunk_row, attn_single, jnp.arange(chunk_size))
        return resolved

    attn_flat = attn.reshape(-1, chunk_size, chunk_size)
    attn_resolved = jax.vmap(resolve_single_chunk)(attn_flat)
    attn = attn_resolved.reshape(B, H, num_chunks, chunk_size, chunk_size)
    eye = jnp.eye(chunk_size, dtype=attn.dtype)
    attn = attn + eye
    value_local = jnp.einsum("bhcij,bhcjv->bhciv", attn, v_beta, precision=_MATMUL_PRECISION)
    k_beta_scaled = k_beta * jnp.exp(g_cumsum)[:, :, :, :, None]
    k_cumdecay = jnp.einsum("bhcij,bhcjk->bhcik", attn, k_beta_scaled, precision=_MATMUL_PRECISION)

    if initial_state is None:
        initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
    else:
        initial_state = initial_state.astype(jnp.float32)

    mask_triu_inner = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

    xs = {
        "query": query.transpose(2, 0, 1, 3, 4),  # (C, B, H, cs, K)
        "key": key.transpose(2, 0, 1, 3, 4),  # (C, B, H, cs, K)
        "value": value_local.transpose(2, 0, 1, 3, 4),  # (C, B, H, cs, V)
        "k_cumdecay": k_cumdecay.transpose(2, 0, 1, 3, 4),  # (C, B, H, cs, K)
        "g_cumsum": g_cumsum.transpose(2, 0, 1, 3),  # (C, B, H, cs)
        "decay_mask": decay_mask.transpose(2, 0, 1, 3, 4),  # (C, B, H, cs, cs)
    }

    def chunk_step(state, inputs):
        """Inter-chunk KDA recurrence step driven by ``lax.scan``.

        Computes the per-chunk output by combining the carried-over
        recurrent ``state`` with the local intra-chunk attention pattern,
        then advances the state through the chunk by applying the
        accumulated gated decay.

        Args:
            state: Recurrent state of shape ``(B, H, K_dim, V_dim)``.
            inputs: Dict with ``query``, ``key``, ``value``, ``k_cumdecay``,
                ``g_cumsum`` and ``decay_mask`` entries for the current
                chunk (see the surrounding ``xs`` construction for shapes).

        Returns:
            tuple: ``(new_state, core_out)``. ``new_state`` carries the
            recurrent buffer to the next chunk; ``core_out`` is the
            chunk's output of shape ``(B, H, chunk_size, V_dim)``.
        """
        q_i = inputs["query"]  # (B, H, cs, K)
        k_i = inputs["key"]  # (B, H, cs, K)
        v_i = inputs["value"]  # (B, H, cs, V)
        k_cumdecay_i = inputs["k_cumdecay"]  # (B, H, cs, K)
        g_cumsum_i = inputs["g_cumsum"]  # (B, H, cs)
        decay_mask_i = inputs["decay_mask"]  # (B, H, cs, cs)

        attn_qk = jnp.einsum("bhik,bhjk->bhij", q_i, k_i, precision=_MATMUL_PRECISION)
        attn_qk = attn_qk * decay_mask_i
        attn_qk = jnp.where(mask_triu_inner, 0.0, attn_qk)

        v_prime = jnp.einsum("bhik,bhkv->bhiv", k_cumdecay_i, state, precision=_MATMUL_PRECISION)

        v_new = v_i - v_prime

        q_scaled = q_i * jnp.exp(g_cumsum_i)[:, :, :, None]
        attn_inter = jnp.einsum("bhik,bhkv->bhiv", q_scaled, state, precision=_MATMUL_PRECISION)

        core_out = attn_inter + jnp.einsum("bhij,bhjv->bhiv", attn_qk, v_new, precision=_MATMUL_PRECISION)

        g_end = g_cumsum_i[:, :, -1]  # (B, H)
        state_decayed = state * jnp.exp(g_end)[:, :, None, None]

        g_diff_state = g_end[:, :, None] - g_cumsum_i  # (B, H, cs)
        k_scaled = k_i * jnp.exp(g_diff_state)[:, :, :, None]  # (B, H, cs, K)

        state_update = jnp.einsum("bhik,bhiv->bhkv", k_scaled, v_new, precision=_MATMUL_PRECISION)

        new_state = state_decayed + state_update

        return new_state, core_out

    final_state, core_attn_out = lax.scan(chunk_step, initial_state, xs)

    # Transpose back and reshape: (C, B, H, cs, V) -> (B, H, C, cs, V) -> (B, H, L, V)
    core_attn_out = core_attn_out.transpose(1, 2, 0, 3, 4)
    core_attn_out = core_attn_out.reshape(B, H, -1, V_dim)
    core_attn_out = core_attn_out[:, :, :L, :]  # Remove padding

    return core_attn_out, final_state


def _single_step_kda_fwd_bthd(
    query: Float[Array, "batch 1 num_heads head_dim"],
    key: Float[Array, "batch 1 num_heads head_dim"],
    value: Float[Array, "batch 1 num_heads d_state"],
    beta: Float[Array, "batch 1 num_heads"],
    decay: Float[Array, "batch 1 num_heads"] | None,
    recurrent_state: Float[Array, "batch num_heads head_dim d_state"],
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch 1 num_heads d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """One-step KDA recurrence for inputs in BTHD layout (decode path).

    Thin shim around :func:`_single_step_kda_core` for the
    ``(batch, seq, num_heads, head_dim)`` layout used by the public
    :class:`KernelDeltaAttnOp` forward. Slices the ``seq=1`` axis,
    delegates to :func:`_single_step_kda_core`, and reattaches the
    sequence axis on the output.

    Args:
        query: ``(batch, 1, num_heads, head_dim)``.
        key: ``(batch, 1, num_heads, head_dim)``.
        value: ``(batch, 1, num_heads, d_state)``.
        beta: ``(batch, 1, num_heads)`` gating coefficient.
        decay: Optional ``(batch, 1, num_heads)`` log-decay; ``None``
            disables the exponential state decay this step.
        recurrent_state: ``(batch, num_heads, head_dim, d_state)``.
        use_qk_l2norm: Whether to L2-normalize Q and K before the update.

    Returns:
        tuple: ``(output, new_state)`` with output of shape
        ``(batch, 1, num_heads, d_state)``.
    """
    output, new_state = _single_step_kda_core(
        query=query[:, 0, :, :],
        key=key[:, 0, :, :],
        value=value[:, 0, :, :],
        beta=beta[:, 0, :],
        decay=None if decay is None else decay[:, 0, :],
        recurrent_state=recurrent_state,
        use_qk_l2norm=use_qk_l2norm,
    )
    return output[:, None, :, :], new_state


def _single_step_kda_core(
    query: Float[Array, "batch num_heads head_dim"],
    key: Float[Array, "batch num_heads head_dim"],
    value: Float[Array, "batch num_heads d_state"],
    beta: Float[Array, "batch num_heads"],
    decay: Float[Array, "batch num_heads"] | None,
    recurrent_state: Float[Array, "batch num_heads head_dim d_state"],
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Shared single-step KDA recurrence over squeezed decode tensors.

    Backbone of the BHTD and BTHD decode shims; the layout-specific
    wrappers squeeze and re-expand the time axis around this call.
    Computes one step of the Kernel Delta Attention recurrence:

    1. Optional L2-normalisation of Q and K (matches FLA convention).
    2. Cast to float32 for stable accumulation; scale ``query`` by
       ``head_dim ** -0.5``.
    3. Apply exponential decay to the recurrent state when ``decay`` is
       provided: ``state *= exp(decay)[..., None, None]``.
    4. Compute ``kv_mem = sum(state * key[..., None], axis=-2)`` and
       the gated delta ``delta = (value - kv_mem) * beta[..., None]``.
    5. Update the state: ``state += key[..., None] * delta[..., None, :]``.
    6. Read out the output: ``output = sum(state * query[..., None],
       axis=-2)``.
    7. Cast the output back to the input dtype and return both the
       output and the float32 ``new_state`` for the caller to persist.

    Args:
        query: ``(batch, num_heads, head_dim)``.
        key: ``(batch, num_heads, head_dim)``.
        value: ``(batch, num_heads, d_state)``.
        beta: ``(batch, num_heads)`` gating coefficient (already
            sigmoid-activated upstream).
        decay: Optional ``(batch, num_heads)`` log-decay; pass ``None``
            to skip the multiplicative state decay (equivalent to
            ``decay=0``).
        recurrent_state: ``(batch, num_heads, head_dim, d_state)``.
        use_qk_l2norm: Whether to L2-normalize ``query`` and ``key``
            before the update. Match the chunked path's setting to
            keep training and inference consistent.

    Returns:
        tuple: ``(output, new_state)`` with output of shape
        ``(batch, num_heads, d_state)`` cast to ``query.dtype`` and
        ``new_state`` retained in float32 to feed the next decode step.
    """
    if use_qk_l2norm:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)

    input_dtype = query.dtype
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    recurrent_state = recurrent_state.astype(jnp.float32)

    head_dim = query.shape[-1]
    scale = 1.0 / (head_dim**0.5)
    query = query * scale

    if decay is not None:
        decay = decay.astype(jnp.float32)
        recurrent_state = recurrent_state * jnp.exp(decay)[:, :, None, None]

    kv_mem = jnp.sum(recurrent_state * key[:, :, :, None], axis=-2)
    delta = (value - kv_mem) * beta[:, :, None]
    new_state = recurrent_state + key[:, :, :, None] * delta[:, :, None, :]
    output = jnp.sum(new_state * query[:, :, :, None], axis=-2)
    return output.astype(input_dtype), new_state


def fused_kda_gate_per_channel(
    gate: Float[Array, "batch seq qkv_dim"],
    A_log: Float[Array, "num_heads"],
    dt_bias: Float[Array, "qkv_dim"],  # noqa: F821
    lower_bound: float | None = -5.0,
) -> Float[Array, "batch seq num_heads head_dim"]:
    """Materialise the GLM-5-Next per-channel KDA log-decay.

    GLM-5-Next (``Glm5NextTextForgetGate``) computes the forget gate as a
    two-layer MLP ``f_b(f_a(x))`` whose ``qkv_dim`` output is viewed as
    ``(num_heads, head_dim)`` — the decay varies per *channel*, unlike the
    per-head Kimi gate handled by :func:`fused_kda_gate`. With the safe-gate
    lower bound configured (GLM-5.3-Flash ships ``-5.0``) the log-decay is

        ``g = lower_bound * sigmoid(exp(A_log_head) * (gate + dt_bias))``

    so ``g`` lies in ``[lower_bound, 0)`` and ``exp(g)`` contracts the
    recurrent state. Without the bound it falls back to the softplus form
    ``-exp(A_log_head) * softplus(gate + dt_bias)`` (values above 20 pass
    through unscaled to avoid overflow), matching the HF reference.

    Args:
        gate: Raw ``f_b_proj`` output before head reshaping,
            ``(batch, seq, num_heads * head_dim)``.
        A_log: Per-head log-decay magnitude, ``(num_heads,)`` — broadcast
            across each head's channels.
        dt_bias: Per-channel time-discretisation bias,
            ``(num_heads * head_dim,)``.
        lower_bound: Safe lower bound for the sigmoid gate, or ``None`` to
            use the softplus formulation.

    Returns:
        Float[Array, "batch seq num_heads head_dim"]: Float32 per-channel
        log-decay ready for the per-channel KDA kernels.
    """
    qkv_dim = dt_bias.shape[-1]
    num_heads = A_log.shape[-1]
    head_dim = qkv_dim // num_heads
    gate = gate.astype(jnp.float32) + dt_bias.astype(jnp.float32).reshape(1, 1, qkv_dim)
    gate = gate.reshape(gate.shape[0], gate.shape[1], num_heads, head_dim)
    decay_rate = jnp.exp(A_log.astype(jnp.float32)).reshape(1, 1, num_heads, 1)
    if lower_bound is not None:
        return jnp.asarray(lower_bound, jnp.float32) * jax.nn.sigmoid(decay_rate * gate)
    # ``jax.nn.softplus`` (logaddexp) is finite in value AND gradient for any finite input.
    # ``where(gate > 20, gate, log1p(exp(gate)))`` still differentiated the dead branch:
    # exp overflowed to inf and its cotangent became 0 * inf = NaN.
    softplus_gate = jax.nn.softplus(gate)
    return -decay_rate * softplus_gate


def _recurrent_kda_per_channel_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len head_dim"],
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
    segment_ids: Array | None = None,
    continue_initial_state: bool = False,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Sequential per-channel-decay KDA recurrence (reference path).

    GLM-5-Next variant of :func:`_recurrent_kda_fwd`: the only difference is
    that ``decay`` carries a trailing ``head_dim`` axis, so each step scales
    the recurrent state per channel, ``state *= exp(g)[:, :, :, None]``,
    instead of per head. Everything else — float32 math, L2-normed Q/K, the
    gated delta update, and the packed-segment reset logic — mirrors the
    per-head implementation exactly.

    Args:
        query: ``(batch, num_heads, seq_len, head_dim)``.
        key: ``(batch, num_heads, seq_len, head_dim)``.
        value: ``(batch, num_heads, seq_len, d_state)``.
        beta: Per-token gating, ``(batch, num_heads, seq_len)``.
        decay: Per-token **per-channel** log-decay,
            ``(batch, num_heads, seq_len, head_dim)``.
        initial_state: Optional starting recurrent state,
            ``(batch, num_heads, head_dim, d_state)``.
        use_qk_l2norm: Whether to L2-normalize Q and K before the loop.
        segment_ids: Optional packed-sequence ids,
            ``(batch, seq_len)``; state resets at segment boundaries and
            padded steps emit zero outputs.
        continue_initial_state: Preserve incoming memory into the first live segment
            and retain memory over padding. Later segment changes still reset state.
            False preserves the packed-training first-segment reset/padding behavior.

    Returns:
        tuple: ``(outputs, final_state)`` — outputs of shape
        ``(batch, num_heads, seq_len, d_state)`` and final state of shape
        ``(batch, num_heads, head_dim, d_state)``, both float32.
    """
    B, H, L, K_dim = query.shape
    V_dim = value.shape[-1]

    if use_qk_l2norm:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)

    scale = 1.0 / (K_dim**0.5)
    query = query * scale

    if initial_state is None:
        initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
    else:
        initial_state = initial_state.astype(jnp.float32)
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    decay = decay.astype(jnp.float32)

    q_seq = query.transpose(2, 0, 1, 3)  # (L, B, H, K)
    k_seq = key.transpose(2, 0, 1, 3)  # (L, B, H, K)
    v_seq = value.transpose(2, 0, 1, 3)  # (L, B, H, V)
    g_seq = decay.transpose(2, 0, 1, 3)  # (L, B, H, K)
    b_seq = beta.transpose(2, 0, 1)  # (L, B, H)
    if segment_ids is not None:
        segment_ids = normalize_packed_segment_ids(segment_ids, L, pad_from_last=False)
        seg_seq = segment_ids.swapaxes(0, 1)  # (L, B)
        initial_segment = jnp.full((B,), -1, dtype=seg_seq.dtype)

    def step_fn(carry, inputs):
        """Single-token per-channel KDA step used by ``lax.scan``.

        Args:
            state: Recurrent state ``(B, H, K, V)`` (plus the segment
                carry when ``segment_ids`` is given).
            inputs: Tuple ``(q_t, k_t, v_t, g_t, beta_t)`` with per-channel
                ``g_t`` of shape ``(B, H, K)`` (plus the segment id).

        Returns:
            tuple: ``(new_state, output)`` — state ``(B, H, K, V)`` and
            output ``(B, H, V)``.
        """
        if segment_ids is None:
            state = carry
            q_t, k_t, v_t, g_t, beta_t = inputs
            valid = None
            next_segment = None
        else:
            state, prev_segment = carry
            q_t, k_t, v_t, g_t, beta_t, segment_t = inputs
            valid = segment_t >= 0
            old_state = state
            new_segment = valid & (segment_t != prev_segment)
            if continue_initial_state:
                # The first live segment continues incoming memory, even after leading padding.
                new_segment = new_segment & (prev_segment >= 0)
            state = jnp.where(new_segment[:, None, None, None], jnp.zeros_like(state), state)
            next_segment = jnp.where(valid, segment_t, prev_segment if continue_initial_state else -1)
        g_exp = jnp.exp(g_t)[:, :, :, None]  # (B, H, K, 1) — per-channel decay
        beta_scaled = beta_t[:, :, None]
        state = state * g_exp
        kv_mem = jnp.sum(state * k_t[:, :, :, None], axis=-2)

        delta = (v_t - kv_mem) * beta_scaled
        state = state + k_t[:, :, :, None] * delta[:, :, None, :]
        output = jnp.sum(state * q_t[:, :, :, None], axis=-2)
        if valid is not None:
            output = jnp.where(valid[:, None, None], output, jnp.zeros_like(output))
            padding_state = old_state if continue_initial_state else jnp.zeros_like(state)
            state = jnp.where(valid[:, None, None, None], state, padding_state)
            return (state, next_segment), output

        return state, output

    if segment_ids is None:
        final_state, outputs = lax.scan(step_fn, initial_state, (q_seq, k_seq, v_seq, g_seq, b_seq))
    else:
        (final_state, _), outputs = lax.scan(
            step_fn,
            (initial_state, initial_segment),
            (q_seq, k_seq, v_seq, g_seq, b_seq, seg_seq),
        )
    outputs = outputs.transpose(1, 2, 0, 3)

    return outputs, final_state


def _kda_chunk_local(k_beta, key, query, g_cumsum, *, mesh=None, lead_specs=None, sub_block: int = 16):
    """Dispatch ejkernel ``kda_chunk_local`` on ``(C, R, H, cs, K)`` chunk tensors.

    A Mosaic (Pallas) call cannot be SPMD-partitioned, so it is used only when that is safe:
    TPU backend and either a single device (``mesh`` of size 1, or no mesh on a 1-device
    host) or a multi-device ``mesh`` with ``lead_specs = (batch_axes, head_axes)`` for the
    inputs, in which case it runs under ``shard_map`` (chunks are independent on every
    leading axis). Everything else uses the XLA implementation, and each such fallback
    logs its reason once.

    When traced inside an outer ``shard_map`` with manual axes (e.g. an SPMD pipeline body),
    a nested ``shard_map`` over the full mesh is invalid: the Pallas kernel then runs
    directly if every remaining auto axis has size 1, and falls back to XLA otherwise.
    """
    args = (k_beta, key, query, g_cumsum)
    xla = kernel_registry.get("kda_chunk_local", platform=Platform.XLA, backend=Backend.ANY)
    if jax.default_backend() != "tpu":
        return xla(*args)
    pallas = kernel_registry.get("kda_chunk_local", platform=Platform.PALLAS, backend=Backend.TPU)

    def fallback(reason):
        logger.warning_once(f"kda_chunk_local: {reason}; using the XLA implementation.")
        return xla(*args)

    def specs():
        if lead_specs is None:
            return None
        spec = PartitionSpec(None, lead_specs[0], lead_specs[1], None, None)
        return (spec,) * 4, (spec, spec)

    return run_mosaic_kernel(lambda *a: pallas(*a, sub_block=sub_block), fallback, args, mesh=mesh, specs=specs)


def _chunk_kda_per_channel_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len head_dim"],
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
    segment_ids: Array | None = None,
    kernel_mesh=None,
    kernel_lead_specs: tuple | None = None,
    continue_initial_state: bool = False,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Chunked per-channel-decay KDA forward for training / long decode.

    ``segment_ids`` (packed training) resets the state at segment boundaries and treats
    negative ids as padding, matching :func:`_recurrent_kda_per_channel_fwd`.

    Thin wrapper over :func:`_chunked_scan_per_channel_rows` (rows = batch,
    ``token_valid`` all-ones). The GLM-5-Next chunked algorithm mirrors HF
    ``chunk_kimi_delta_attention``: the cumulative log-decay ``g`` carries a
    trailing ``head_dim`` axis, so every decay factor becomes per-channel —

    - ``decay_mask[i, j, :] = exp(g_cumsum[i, :] - g_cumsum[j, :])`` enters
      the decay-weighted ``k_beta . key`` dot products inside the chunk,
    - ``k_cumdecay = attn @ (k_beta * exp(g_cumsum))`` scales each key
      channel by its own decay,
    - the inter-chunk query scale and the state's end-of-chunk decay apply
      ``exp(g)`` per channel, and the outer-state update uses
      ``k * exp(g_end - g)``.

    The chunk-local terms (``g_cumsum``, ``attn``, ``k_cumdecay``, ...) are
    materialized for every chunk of the whole sequence at once, and only the
    inter-chunk recurrent state is carried by the scan, so transient memory
    scales with sequence length (rely on outer remat for large batch x
    heads x length). All math runs in float32.

    Args:
        query: ``(batch, num_heads, seq_len, head_dim)``.
        key: ``(batch, num_heads, seq_len, head_dim)``.
        value: ``(batch, num_heads, seq_len, d_state)``.
        beta: Per-token gating, ``(batch, num_heads, seq_len)``.
        decay: Per-token **per-channel** log-decay,
            ``(batch, num_heads, seq_len, head_dim)``.
        chunk_size: Chunk granularity for the intra-chunk parallel kernel.
        initial_state: Optional starting recurrent state,
            ``(batch, num_heads, head_dim, d_state)``.
        use_qk_l2norm: Whether to L2-normalize Q and K before processing.
        segment_ids: Optional packed document IDs; negative IDs denote padding.
        kernel_mesh: Optional mesh for the chunk-local kernel.
        kernel_lead_specs: Optional row/head mesh axes for the chunk-local kernel.
        continue_initial_state: Continue incoming memory through the first valid
            segment and padding; False preserves packed-training resets.

    Returns:
        tuple: ``(outputs, final_state)`` with outputs of shape
        ``(batch, num_heads, seq_len, d_state)`` (padding removed) and
        final state ``(batch, num_heads, head_dim, d_state)``.
    """
    batch, _, seq_len = beta.shape
    return _chunked_scan_per_channel_rows(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        initial_state=initial_state,
        token_valid=jnp.ones((batch, seq_len), dtype=query.dtype),
        chunk_size=chunk_size,
        use_qk_l2norm=use_qk_l2norm,
        segment_ids=segment_ids,
        kernel_mesh=kernel_mesh,
        kernel_lead_specs=kernel_lead_specs,
        continue_initial_state=continue_initial_state,
    )


KDA_CHUNK_IMPLS = ("scan", "kda_chunk")


def _chunk_kda_per_channel_fused(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len head_dim"],
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
    segment_ids: Array | None = None,
    kernel_mesh=None,
    kernel_lead_specs: tuple | None = None,
    continue_initial_state: bool = False,
    precision: str = "highest",
    cfg: KdaChunkConfig | None = None,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Per-channel chunked KDA through the fused ejkernel ``kda_chunk`` operation.

    Same contract as :func:`_chunk_kda_per_channel_fwd` (packed ``segment_ids`` with negative padding,
    ``continue_initial_state``). The Mosaic kernel cannot be SPMD-partitioned, so on TPU it runs directly on one
    device, under ``shard_map`` over ``kernel_lead_specs = (batch_axes, head_axes)`` on a multi-device mesh, or
    falls back to the ejkernel XLA implementation when neither is safe (see :func:`run_mosaic_kernel`).

    Args:
        query: ``(batch, num_heads, seq_len, head_dim)``.
        key: ``(batch, num_heads, seq_len, head_dim)``.
        value: ``(batch, num_heads, seq_len, d_state)``.
        beta: ``(batch, num_heads, seq_len)``.
        decay: Per-channel log-decay ``(batch, num_heads, seq_len, head_dim)``.
        chunk_size: Chunk length (a power of two in ``[8, 128]`` for the Pallas kernel).
        initial_state: Optional ``(batch, num_heads, head_dim, d_state)``.
        use_qk_l2norm: L2-normalise queries and keys.
        segment_ids: Optional packed ids ``(batch, seq_len)``; negative ids are padding.
        kernel_mesh: Optional mesh for the kernel's ``shard_map``.
        kernel_lead_specs: Optional ``(batch_axes, head_axes)`` of the inputs.
        continue_initial_state: Cached-continuation segment semantics.
        precision: ``"highest"`` or ``"default"`` for the non-inverse dots (the chunk inverse stays HIGHEST).
        cfg: Optional ejkernel :class:`KdaChunkConfig` (``block_h``, ``backward``, ``bwd_reduce``).

    Returns:
        tuple: ``(outputs, final_state)`` float32.
    """
    seq_len = query.shape[2]
    seg = None
    if segment_ids is not None:
        seg = normalize_packed_segment_ids(segment_ids, seq_len, pad_from_last=False).astype(jnp.int32)
    arrays = (query, key, value, beta, decay, initial_state, seg)
    present = tuple(i for i, a in enumerate(arrays) if a is not None)
    args = tuple(arrays[i] for i in present)

    def call(*local, platform=None):
        full = [None] * len(arrays)
        for i, a in zip(present, local, strict=True):
            full[i] = a
        return ejkernel_kda_chunk(
            *full,
            chunk_size=chunk_size,
            use_qk_l2norm=use_qk_l2norm,
            continue_initial_state=continue_initial_state,
            precision=precision,
            platform=platform,
            cfg=cfg,
        )

    if jax.default_backend() != "tpu":
        return call(*args, platform="xla")

    def fallback(reason):
        logger.warning_once(f"kda_chunk: {reason}; using the XLA implementation.")
        return call(*args, platform="xla")

    def specs():
        if kernel_lead_specs is None:
            return None
        b_axes, h_axes = kernel_lead_specs
        bhld = PartitionSpec(b_axes, h_axes, None, None)
        full = (bhld, bhld, bhld, PartitionSpec(b_axes, h_axes, None), bhld, bhld, PartitionSpec(b_axes, None))
        return tuple(full[i] for i in present), (bhld, bhld)

    return run_mosaic_kernel(call, fallback, args, mesh=kernel_mesh, specs=specs)


def _chunked_scan_per_channel_rows(
    query: Float[Array, "rows heads seq_len head_dim"],
    key: Float[Array, "rows heads seq_len head_dim"],
    value: Float[Array, "rows heads seq_len d_state"],
    beta: Float[Array, "rows heads seq_len"],
    decay: Float[Array, "rows heads seq_len head_dim"],
    initial_state: Float[Array, "rows heads head_dim d_state"] | None,
    token_valid: Float[Array, "rows seq_len"],
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    segment_ids: Array | None = None,
    kernel_mesh=None,
    kernel_lead_specs: tuple | None = None,
    continue_initial_state: bool = False,
) -> tuple[Float[Array, "rows heads seq_len d_state"], Float[Array, "rows heads head_dim d_state"]]:
    """Chunked per-channel KDA over independent rows (training and non-packed prefill).

    ``kernel_mesh`` / ``kernel_lead_specs`` (the batch and head mesh axes of the inputs) let
    the TPU Pallas chunk-local kernel run under ``shard_map``; see :func:`_kda_chunk_local`.

    Three phases, all float32:

    1. ``lax.map`` over chunks builds the chunk-local ``attn`` / ``attn_qk`` matrices; the
       ``(cs, cs, K)`` per-channel decay tensor exists for one chunk at a time.
    2. ``(I - attn)^-1`` is solved for **every chunk in one batched** ``triangular_solve``.
       This replaces a ``chunk_size``-step serial forward substitution inside every chunk
       step (TPU, B=2 H=16 L=8192 K=V=128: fwd+bwd 395 -> 59 ms, temp 44 -> 37 GB).
    3. ``lax.scan`` over chunks carries only the recurrent state.

    ``token_valid`` masks positions that contribute nothing (their ``q``/``k``/``v``/``g``/
    ``beta`` are zeroed). With ``segment_ids`` (packed training), the state resets whenever
    the segment id changes and positions with a negative id are padding, exactly like
    :func:`_recurrent_kda_per_channel_fwd`'s segment mode (``initial_state`` is then
    reset for the first run unless ``continue_initial_state=True``). Continuation preserves
    memory across padding and into the first valid segment, while later live segment
    changes reset it. Boundaries may fall anywhere inside a chunk: pairs across runs are
    masked out of the intra-chunk matrices, the incoming state only reaches the chunk's
    first run, and the outgoing state only collects the chunk's last run.

    Args:
        query: ``(rows, heads, seq_len, head_dim)`` window queries.
        key: ``(rows, heads, seq_len, head_dim)``.
        value: ``(rows, heads, seq_len, d_state)``.
        beta: ``(rows, heads, seq_len)`` gating.
        decay: ``(rows, heads, seq_len, head_dim)`` per-channel log-decay.
        initial_state: ``(rows, heads, head_dim, d_state)`` starting state.
        token_valid: ``(rows, seq_len)`` validity mask (1.0/0.0 or bool).
        chunk_size: Chunk granularity (must divide nothing in particular;
            the tail chunk is zero-padded).
        use_qk_l2norm: Whether to L2-normalize Q and K first.
        segment_ids: Optional packed-sequence ids ``(rows, seq_len)``; negative ids are
            padding. See above.
        kernel_mesh: Optional mesh for the chunk-local kernel.
        kernel_lead_specs: Optional row/head mesh axes for the chunk-local kernel.
        continue_initial_state: Continue incoming memory through the first valid
            segment and padding; False preserves packed-training resets.

    Returns:
        tuple: ``(outputs, final_state)`` — outputs
        ``(rows, heads, seq_len, d_state)`` float32 and final state
        ``(rows, heads, head_dim, d_state)`` float32.
    """
    R, H, L, K_dim = query.shape
    V_dim = value.shape[-1]
    cs = chunk_size

    if use_qk_l2norm:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)

    tok_valid = jnp.asarray(token_valid) > 0  # (R, L)
    run = None
    if segment_ids is not None:
        seg = normalize_packed_segment_ids(segment_ids, L, pad_from_last=False).astype(jnp.int32)
        tok_valid = tok_valid & (seg >= 0)
        # Run ids (>= 1): a new run starts whenever the segment id changes, padding included,
        # which is exactly when the recurrent reference resets its state.
        if continue_initial_state:
            # Padding is inert: compare each live token to the preceding live segment,
            # not to a padding sentinel, and let the first live run inherit memory.
            indices = jnp.broadcast_to(jnp.arange(L), (R, L))
            last_valid = lax.associative_scan(jnp.maximum, jnp.where(tok_valid, indices, -1), axis=1)
            previous = jnp.concatenate([jnp.full((R, 1), -1), last_valid[:, :-1]], axis=1)
            previous_segment = jnp.take_along_axis(seg, jnp.maximum(previous, 0), axis=1)
            change = tok_valid & (previous >= 0) & (seg != previous_segment)
            run = 1 + jnp.cumsum(change.astype(jnp.int32), axis=1)
        else:
            change = jnp.concatenate([jnp.ones((R, 1), dtype=bool), seg[:, 1:] != seg[:, :-1]], axis=1)
            run = jnp.cumsum(change.astype(jnp.int32), axis=1)
    valid4 = tok_valid[:, None, :, None]
    query = jnp.where(valid4, query.astype(jnp.float32), 0.0)
    key = jnp.where(valid4, key.astype(jnp.float32), 0.0)
    value = jnp.where(valid4, value.astype(jnp.float32), 0.0)
    decay = jnp.where(valid4, decay.astype(jnp.float32), 0.0)
    beta = jnp.where(tok_valid[:, None, :], beta.astype(jnp.float32), 0.0)

    pad_size = (cs - L % cs) % cs
    if pad_size > 0:
        p4 = ((0, 0), (0, 0), (0, pad_size), (0, 0))
        query, key, value, decay = (jnp.pad(x, p4) for x in (query, key, value, decay))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        if run is not None:
            # Chunk padding continues the last run (its inputs are zero, so it is inert).
            run = jnp.pad(run, ((0, 0), (0, pad_size)), mode="edge")
    num_chunks = (L + pad_size) // cs
    query = query * (1.0 / (K_dim**0.5))

    def chunks(x):  # (R, H, T, D) -> (C, R, H, cs, D)
        return x.reshape(R, H, num_chunks, cs, x.shape[-1]).transpose(2, 0, 1, 3, 4)

    q_c, k_c, v_c, g_c = chunks(query), chunks(key), chunks(value), chunks(decay)
    b_c = beta.reshape(R, H, num_chunks, cs).transpose(2, 0, 1, 3)  # (C, R, H, cs)
    run_c = None if run is None else run.reshape(R, num_chunks, cs).transpose(1, 0, 2)  # (C, R, cs)

    g_cumsum = jnp.cumsum(g_c, axis=-2)  # (C, R, H, cs, K)
    k_beta = k_c * b_c[..., None]
    v_beta = v_c * b_c[..., None]

    eye = jnp.eye(cs, dtype=jnp.float32)

    # Chunk-local per-channel matrices via ejkernel ``kda_chunk_local``: Pallas on TPU
    # (sub-block rescaled MXU tiles, closed-form backward), rematerialized XLA elsewhere.
    attn, attn_qk = _kda_chunk_local(
        k_beta, k_c, q_c, g_cumsum, mesh=kernel_mesh, lead_specs=kernel_lead_specs
    )  # (C, R, H, cs, cs)
    if run_c is not None:  # segment mode: no pairs across runs
        same = (run_c[:, :, :, None] == run_c[:, :, None, :])[:, :, None]  # (C, R, 1, cs, cs)
        attn = jnp.where(same, attn, 0.0)
        attn_qk = jnp.where(same, attn_qk, 0.0)
    # (I - attn)^-1 for every chunk in one call (TPU solve latency is ~fixed per call).
    lower = eye - attn
    attn_resolved = lax.linalg.triangular_solve(
        lower, jnp.broadcast_to(eye, lower.shape), left_side=True, lower=True, unit_diagonal=True
    )
    value_local = jnp.einsum("crhij,crhjv->crhiv", attn_resolved, v_beta, precision=_MATMUL_PRECISION)
    k_cumdecay = jnp.einsum("crhij,crhjk->crhik", attn_resolved, k_beta * jnp.exp(g_cumsum), precision=_MATMUL_PRECISION)
    q_scaled = q_c * jnp.exp(g_cumsum)
    g_end = g_cumsum[:, :, :, -1, :]  # (C, R, H, K)
    k_scaled = k_c * jnp.exp(g_end[:, :, :, None, :] - g_cumsum)
    decay_end = jnp.exp(g_end)

    if run_c is not None:
        run_last = run_c[:, :, -1]  # (C, R)
        # Packed training starts from a reset; cached continuation inherits the first run.
        first_run = run_c[:1, :, 0] if continue_initial_state else jnp.full((1, R), -2, dtype=run_c.dtype)
        run_prev = jnp.concatenate([first_run, run_last[:-1]], axis=0)
        continues = (run_c == run_prev[:, :, None]).astype(jnp.float32)[:, :, None, :, None]
        in_last = (run_c == run_last[:, :, None]).astype(jnp.float32)[:, :, None, :, None]
        keep = (run_last == run_prev).astype(jnp.float32)[:, :, None, None]
        k_cumdecay = k_cumdecay * continues  # incoming state reaches only the first run
        q_scaled = q_scaled * continues
        k_scaled = k_scaled * in_last  # outgoing state collects only the last run
        decay_end = decay_end * keep  # carried state survives only a run spanning the chunk

    if initial_state is None:
        state0 = jnp.zeros((R, H, K_dim, V_dim), dtype=jnp.float32)
    else:
        state0 = initial_state.astype(jnp.float32)

    def state_step(state, x):
        kcd, vl, qs, aqk, ks, de = x
        v_new = vl - jnp.einsum("rhik,rhkv->rhiv", kcd, state, precision=_MATMUL_PRECISION)
        out = jnp.einsum("rhik,rhkv->rhiv", qs, state, precision=_MATMUL_PRECISION) + jnp.einsum(
            "rhij,rhjv->rhiv", aqk, v_new, precision=_MATMUL_PRECISION
        )
        state = state * de[..., None] + jnp.einsum("rhik,rhiv->rhkv", ks, v_new, precision=_MATMUL_PRECISION)
        return state, out

    final_state, core_attn_out = lax.scan(
        state_step, state0, (k_cumdecay, value_local, q_scaled, attn_qk, k_scaled, decay_end)
    )
    core_attn_out = core_attn_out.transpose(1, 2, 0, 3, 4)  # (R, H, C, cs, V)
    core_attn_out = core_attn_out.reshape(R, H, -1, V_dim)[:, :, :L, :]
    return core_attn_out, final_state


def _packed_segment_kda_per_channel(
    query: Float[Array, "rows heads seq_len head_dim"],
    key: Float[Array, "rows heads seq_len head_dim"],
    value: Float[Array, "rows heads seq_len d_state"],
    beta: Float[Array, "rows heads seq_len"],
    decay: Float[Array, "rows heads seq_len head_dim"],
    initial_state: Float[Array, "rows heads head_dim d_state"] | None,
    token_valid: Float[Array, "rows seq_len"],
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    max_tokens: int | None = None,
) -> tuple[Float[Array, "rows heads seq_len d_state"], Float[Array, "rows heads head_dim d_state"]]:
    """Serving prefill for per-channel KDA over packed rows, without per-row bucket padding.

    Same contract and math as :func:`_chunked_scan_per_channel_rows`, restructured for
    eSurge prefill steps:

    - **Segment-aligned chunks.** The row layout pads every row to the full bucket, so the
      row scan processes ``rows * seq_len`` token slots for a step holding at most
      ``max_tokens`` real ones. Here each row's valid tokens are laid out back-to-back, every
      row starting on a chunk boundary, giving a static budget of
      ``ceil(max_tokens / chunk_size) + rows`` chunks. Each chunk belongs to exactly one row.
    - **One batched triangular solve.** The chunk-local ``(I - A)^-1`` is solved for all
      chunks in a single ``triangular_solve`` instead of a ``chunk_size``-step serial
      forward substitution inside every chunk step.
    - **State-only scan.** The chunk scan carries only the recurrent state; it restarts from
      the row's initial state at the row's first chunk and records the row's final state at
      its last chunk. Rows with no valid tokens keep their initial state.

    Differs from the row scan only by float32 reassociation. Not intended for differentiation.

    Args:
        query: ``(rows, heads, seq_len, head_dim)`` window queries.
        key: ``(rows, heads, seq_len, head_dim)``.
        value: ``(rows, heads, seq_len, d_state)``.
        beta: ``(rows, heads, seq_len)`` gating.
        decay: ``(rows, heads, seq_len, head_dim)`` per-channel log-decay.
        initial_state: ``(rows, heads, head_dim, d_state)`` starting state, or ``None``.
        token_valid: ``(rows, seq_len)`` validity mask. Must be prefix-valid per row (each row
            owns its first ``lens[r]`` positions), as in eSurge packed rows.
        chunk_size: Chunk granularity.
        use_qk_l2norm: Whether to L2-normalize Q and K first.
        max_tokens: Static upper bound on the total valid tokens over all rows. Serving passes
            the packed stream length, since rows are segments of that stream. Tokens beyond
            this budget are dropped, so callers must honour the bound. ``None`` means
            ``rows * seq_len`` (no saving from segment packing).

    Returns:
        tuple: ``(outputs, final_state)`` — outputs ``(rows, heads, seq_len, d_state)``
        float32 (zero at invalid positions) and final state
        ``(rows, heads, head_dim, d_state)`` float32.
    """
    R, H, L, K_dim = query.shape
    V_dim = value.shape[-1]
    cs = chunk_size
    if max_tokens is None:
        max_tokens = R * L
    num_chunks = min(-(-max_tokens // cs) + R, R * (-(-L // cs)))  # static chunk budget

    lens = jnp.sum(jnp.asarray(token_valid) > 0, axis=1).astype(jnp.int32)  # (R,)
    row_chunks = (lens + cs - 1) // cs
    chunk_end = jnp.cumsum(row_chunks)  # exclusive end chunk per row
    chunk_start = chunk_end - row_chunks
    c = jnp.arange(num_chunks)
    chunk_row = jnp.clip(jnp.sum(chunk_end[None, :] <= c[:, None], axis=1), 0, R - 1)  # (C,)
    chunk_in_row = c - chunk_start[chunk_row]
    live = c < chunk_end[-1]
    pos = chunk_in_row[:, None] * cs + jnp.arange(cs)[None, :]  # (C, cs) position within the row
    tok = live[:, None] & (pos >= 0) & (pos < lens[chunk_row][:, None])  # (C, cs)
    pos = jnp.clip(pos, 0, L - 1)
    tok4 = tok[:, None, :, None]

    def gather(x):  # (R, H, L, D) -> (C, H, cs, D), zero outside the row's valid tokens
        g = x[chunk_row[:, None], :, pos].transpose(0, 2, 1, 3)
        return jnp.where(tok4, g.astype(jnp.float32), 0.0)

    q = gather(query)
    k = gather(key)
    v = gather(value)
    g = gather(decay)
    b = jnp.where(tok[:, None, :], beta[chunk_row[:, None], :, pos].transpose(0, 2, 1).astype(jnp.float32), 0.0)
    if use_qk_l2norm:
        q = l2norm(q, axis=-1, eps=1e-6)
        k = l2norm(k, axis=-1, eps=1e-6)
    q = q * (1.0 / (K_dim**0.5))

    mask_tril = jnp.tril(jnp.ones((cs, cs), dtype=bool), k=0)
    mask_triu = jnp.triu(jnp.ones((cs, cs), dtype=bool), k=0)
    mask_triu_inner = jnp.triu(jnp.ones((cs, cs), dtype=bool), k=1)
    eye = jnp.eye(cs, dtype=jnp.float32)

    g_cumsum = jnp.cumsum(g, axis=-2)  # (C, H, cs, K)
    k_beta = k * b[..., None]
    v_beta = v * b[..., None]

    def chunk_local(args):
        # Pair difference before exp, masked on both sides: see the overflow note in
        # :func:`_chunked_scan_per_channel_rows`. The (cs, cs, K) decay tensor stays chunk-local.
        kb, kk, qq, gc = args  # (H, cs, K)
        g_diff = jnp.where(mask_tril[:, :, None], gc[:, :, None, :] - gc[:, None, :, :], 0.0)
        decay_mask = jnp.where(mask_tril[:, :, None], jnp.exp(g_diff), 0.0)
        attn = -jnp.einsum("hik,hjk,hijk->hij", kb, kk, decay_mask, precision=_MATMUL_PRECISION)
        attn = jnp.where(mask_triu, 0.0, attn)
        attn_qk = jnp.einsum("hik,hjk,hijk->hij", qq, kk, decay_mask, precision=_MATMUL_PRECISION)
        attn_qk = jnp.where(mask_triu_inner, 0.0, attn_qk)
        return attn, attn_qk

    attn, attn_qk = lax.map(chunk_local, (k_beta, k, q, g_cumsum))  # (C, H, cs, cs)
    # (I - attn)^-1 for every chunk at once; equals the forward-substitution resolve + I.
    lower = eye - attn
    attn_resolved = lax.linalg.triangular_solve(
        lower, jnp.broadcast_to(eye, lower.shape), left_side=True, lower=True, unit_diagonal=True
    )
    value_local = jnp.einsum("chij,chjv->chiv", attn_resolved, v_beta, precision=_MATMUL_PRECISION)
    k_cumdecay = jnp.einsum("chij,chjk->chik", attn_resolved, k_beta * jnp.exp(g_cumsum), precision=_MATMUL_PRECISION)
    q_scaled = q * jnp.exp(g_cumsum)
    g_end = g_cumsum[:, :, -1, :]  # (C, H, K)
    k_scaled = k * jnp.exp(g_end[:, :, None, :] - g_cumsum)
    decay_end = jnp.exp(g_end)

    is_first = live & (chunk_in_row == 0)
    is_last = live & (c == chunk_end[chunk_row] - 1)
    if initial_state is None:
        init = jnp.zeros((R, H, K_dim, V_dim), dtype=jnp.float32)
    else:
        init = initial_state.astype(jnp.float32)

    def state_step(carry, xs):
        state, finals = carry
        kcd, vl, qs, aqk, ks, de, row, first, last = xs
        state = jnp.where(first, init[row], state)
        v_new = vl - jnp.einsum("hik,hkv->hiv", kcd, state, precision=_MATMUL_PRECISION)
        out = jnp.einsum("hik,hkv->hiv", qs, state, precision=_MATMUL_PRECISION) + jnp.einsum(
            "hij,hjv->hiv", aqk, v_new, precision=_MATMUL_PRECISION
        )
        state = state * de[..., None] + jnp.einsum("hik,hiv->hkv", ks, v_new, precision=_MATMUL_PRECISION)
        finals = finals.at[row].set(jnp.where(last, state, finals[row]))
        return (state, finals), out

    (_, final_state), chunk_out = lax.scan(
        state_step,
        (jnp.zeros((H, K_dim, V_dim), jnp.float32), init),
        (k_cumdecay, value_local, q_scaled, attn_qk, k_scaled, decay_end, chunk_row, is_first, is_last),
    )  # chunk_out (C, H, cs, V)

    j = jnp.arange(L)
    out_chunk = jnp.clip(chunk_start[:, None] + j[None, :] // cs, 0, num_chunks - 1)  # (R, L)
    out = chunk_out[out_chunk, :, (j % cs)[None, :]]  # (R, L, H, V)
    out = jnp.where((j[None, :] < lens[:, None])[:, :, None, None], out, 0.0)
    return out.transpose(0, 2, 1, 3), final_state


def _single_step_kda_per_channel_fwd_bthd(
    query: Float[Array, "batch 1 num_heads head_dim"],
    key: Float[Array, "batch 1 num_heads head_dim"],
    value: Float[Array, "batch 1 num_heads d_state"],
    beta: Float[Array, "batch 1 num_heads"],
    decay: Float[Array, "batch 1 num_heads head_dim"] | None,
    recurrent_state: Float[Array, "batch num_heads head_dim d_state"],
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch 1 num_heads d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """One-step per-channel-decay KDA recurrence (decode path, BTHD).

    GLM-5-Next variant of :func:`_single_step_kda_core` for decode: identical
    gated delta update, except the decay scales the recurrent state per
    channel — ``state *= exp(g)[..., :, None]`` with ``g`` of shape
    ``(batch, num_heads, head_dim)``.

    Args:
        query: ``(batch, 1, num_heads, head_dim)``.
        key: ``(batch, 1, num_heads, head_dim)``.
        value: ``(batch, 1, num_heads, d_state)``.
        beta: ``(batch, 1, num_heads)`` gating coefficient.
        decay: Optional ``(batch, 1, num_heads, head_dim)`` per-channel
            log-decay; ``None`` skips the multiplicative decay.
        recurrent_state: ``(batch, num_heads, head_dim, d_state)``.
        use_qk_l2norm: Whether to L2-normalize Q and K first.

    Returns:
        tuple: ``(output, new_state)`` — output
        ``(batch, 1, num_heads, d_state)`` cast to the input dtype and
        ``new_state`` retained in float32.
    """
    if use_qk_l2norm:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)

    input_dtype = query.dtype
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    recurrent_state = recurrent_state.astype(jnp.float32)

    head_dim = query.shape[-1]
    scale = 1.0 / (head_dim**0.5)
    query = query * scale

    if decay is not None:
        decay = decay.astype(jnp.float32)
        recurrent_state = recurrent_state * jnp.exp(decay)[:, 0, :, :, None]  # per-channel

    kv_mem = jnp.sum(recurrent_state * key[:, 0, :, :, None], axis=-2)
    delta = (value[:, 0, :, :] - kv_mem) * beta[:, 0, :, None]
    new_state = recurrent_state + key[:, 0, :, :, None] * delta[:, :, None, :]
    output = jnp.sum(new_state * query[:, 0, :, :, None], axis=-2)
    return output.astype(input_dtype)[:, None, :, :], new_state


_SHORT_WINDOW_MAX = 8


def _short_window_kda_per_channel(
    query: Float[Array, "rows heads seq_len head_dim"],
    key: Float[Array, "rows heads seq_len head_dim"],
    value: Float[Array, "rows heads seq_len d_state"],
    beta: Float[Array, "rows heads seq_len"],
    decay: Float[Array, "rows heads seq_len head_dim"],
    initial_state: Float[Array, "rows heads head_dim d_state"],
    lens: Array,
    max_steps: int,
    stack_steps: int = 0,
) -> tuple[Array, ...]:
    """Advance per-channel KDA rows of at most ``max_steps`` tokens with the per-token decode recurrence.

    Applies :func:`_single_step_kda_per_channel_fwd_bthd` once per position (a ``fori_loop`` whose trip count is
    the longest row, capped at ``max_steps``), masking each row past its own length. Between steps the carried
    state is stored in ``initial_state``'s dtype, exactly as decode stores it in the cache, so every token sees
    plain decode's arithmetic. Used for short windows (speculative verify), where the chunked prefill kernel would
    build full ``chunk_size`` chunks for a handful of tokens per row.

    Args:
        query: ``(rows, heads, seq_len, head_dim)`` post-conv queries (row ``r`` owns its first ``lens[r]`` slots).
        key: ``(rows, heads, seq_len, head_dim)``.
        value: ``(rows, heads, seq_len, d_state)``.
        beta: ``(rows, heads, seq_len)`` gating.
        decay: ``(rows, heads, seq_len, head_dim)`` per-channel log-decay.
        initial_state: ``(rows, heads, head_dim, d_state)`` carried state (its dtype is the storage dtype).
        lens: ``(rows,)`` valid tokens per row.
        max_steps: Static upper bound on ``lens``; positions at or past it are not processed.
        stack_steps: When positive, also return the state after each of the first ``stack_steps`` tokens
            (speculative candidate rows), stored in ``initial_state``'s dtype.

    Returns:
        tuple: ``(outputs, final_state)`` — outputs ``(rows, heads, seq_len, d_state)`` float32 (zero at
        positions past a row's length) and final state float32. Rows with ``lens == 0`` keep their state. With
        ``stack_steps > 0`` a third element ``(stack_steps, rows, heads, head_dim, d_state)`` holds the state after
        ``t + 1`` tokens at index ``t`` (entries past a row's length are unspecified).
    """
    rows, heads, seq, _ = query.shape
    d_state = value.shape[-1]
    state_dtype = initial_state.dtype
    trips = jnp.minimum(jnp.max(lens), max_steps).astype(jnp.int32)
    stack0 = jnp.zeros((stack_steps, *initial_state.shape), state_dtype) if stack_steps > 0 else None

    def body(t, carry):
        state, out, stack = carry

        def at(x):
            return lax.dynamic_slice_in_dim(x, t, 1, axis=2)

        o, s = _single_step_kda_per_channel_fwd_bthd(
            query=at(query).transpose(0, 2, 1, 3),
            key=at(key).transpose(0, 2, 1, 3),
            value=at(value).transpose(0, 2, 1, 3),
            beta=at(beta).transpose(0, 2, 1),
            decay=at(decay).transpose(0, 2, 1, 3),
            recurrent_state=state,
            use_qk_l2norm=True,
        )
        live = (t < lens)[:, None, None, None]
        state = jnp.where(live, s.astype(state_dtype), state)
        o = jnp.where(live, o.astype(jnp.float32), 0.0).transpose(0, 2, 1, 3)  # (R, H, 1, V)
        out = lax.dynamic_update_slice_in_dim(out, o, t, axis=2)
        if stack is not None:
            idx = jnp.minimum(t, stack_steps - 1)
            entry = jnp.where(t < stack_steps, state, lax.dynamic_index_in_dim(stack, idx, axis=0, keepdims=False))
            stack = lax.dynamic_update_index_in_dim(stack, entry, idx, axis=0)
        return state, out, stack

    out0 = jnp.zeros((rows, heads, seq, d_state), jnp.float32)
    state, out, stack = lax.fori_loop(0, trips, body, (initial_state, out0, stack0))
    if stack is None:
        return out, state.astype(jnp.float32)
    return out, state.astype(jnp.float32), stack


def kda_packed_rows_forward(
    query: Float[Array, "1 seq_len key_dim"],
    key: Float[Array, "1 seq_len key_dim"],
    value: Float[Array, "1 seq_len value_dim"],
    beta: Float[Array, "1 seq_len num_heads"],
    decay: Float[Array, "1 seq_len num_heads head_dim"],
    conv_kernels: tuple[Array, Array, Array],
    cache_view: KDACacheView,
    query_start_loc: Array,
    context_lens: Array | None = None,
    conv_dtype: jnp.dtype = jnp.bfloat16,
    chunk_size: int = 64,
    spec_recurrent_commit: Array | None = None,
) -> tuple[Float[Array, "1 seq_len num_heads d_state"], KDACacheView]:
    """Advance per-request per-channel KDA state over an eSurge packed token stream.

    Model-agnostic serving driver shared by every per-channel KDA layer (GLM-5-Next,
    Kimi Linear). The runner flattens all scheduled requests into one token stream
    ``[1, T, D]``; ``query_start_loc`` gives each request's (row's) segment as
    ``[start_i, start_{i+1})``, and per-request conv windows and recurrent memory live
    in per-row cache slots. This function splits the stream into rows, runs the causal
    depthwise conv per row (with the carried window as prefix for rows that already
    hold context), advances the delta-rule state per row -- single-step on pure decode
    steps, :func:`_packed_segment_kda_per_channel` otherwise -- and gathers the per-row
    outputs back into the packed stream.

    The caller owns the model-specific parts: projections, the decay gate, the output
    gate/norm and the output projection.

    Args:
        query: Packed Q projection (pre-conv) ``[1, T, key_dim]``.
        key: Packed K projection (pre-conv) ``[1, T, key_dim]``.
        value: Packed V projection (pre-conv) ``[1, T, value_dim]``.
        beta: Packed per-head gating ``[1, T, heads]`` (sigmoid applied).
        decay: Packed per-channel log-decay ``[1, T, heads, head_dim]``.
        conv_kernels: Depthwise causal conv kernels for q, k, v, each ``[dim, d_conv]``
            (``conv1d.weight.value.squeeze(1).T`` for a grouped ``nn.Conv1d``).
        cache_view: Per-layer KDA cache view with one slot per row.
        query_start_loc: Cumulative segment offsets ``[rows + 1]``. Rank-major DP metadata
            (``[dp, rows + 1]``) is rejected with ``ValueError``.
        context_lens: Tokens per row *including* this step; a row with
            ``context_lens > scheduled`` continues from its carried conv window. When
            ``None``, only single-token rows are treated as continuations.
        conv_dtype: Output dtype of the causal conv (matches the dense path).
        chunk_size: Chunk size of the prefill kernel.
        spec_recurrent_commit: Optional ``[2, >= rows]`` int32 speculative commit (row 0: active flag, row 1:
            accepted prefix length). Used only when the cache pool holds speculative candidate rows (pool size
            ``rows * (1 + C)``, prefix-major layout ``rows + (prefix - 1) * rows + row``): the accepted prefix's
            candidate row is copied into the live row before this step runs. Candidate rows are rewritten for every
            speculative row (``2 <= lens <= C`` with prior context) with the decode-exact state after each prefix.

    Returns:
        tuple: ``(outputs, new_cache_view)`` -- outputs ``[1, T, heads, head_v_dim]``
        float32 (zero at bucket padding), and the cache view with updated conv windows
        and recurrent states.
    """
    if query_start_loc.ndim != 1:
        # Rank-major DP metadata ([dp, rows + 1]) would need per-rank state scatter and a
        # ``context_lens`` slice; using only rank 0 would leave every other rank's rows
        # un-advanced with no error, so refuse instead.
        raise ValueError(
            "kda_packed_rows_forward requires 1-D query_start_loc [rows + 1]; rank-major DP metadata "
            f"(got shape {tuple(query_start_loc.shape)}) is not supported for per-channel KDA layers."
        )
    rows = query_start_loc.shape[0] - 1
    seq = query.shape[1]
    starts = query_start_loc[:-1]
    lens = query_start_loc[1:] - starts  # [rows]
    ar = jnp.arange(seq)

    # Index-clamped gather (NOT start-clamped slicing): the token bucket is padded to the
    # compile width, so `starts[i] + seq` routinely overruns the buffer and a start-clamped
    # dynamic_slice would clamp back to 0, silently feeding row 0's tokens to later rows.
    def gather(x):
        idx = jnp.clip(starts[:, None] + ar[None, :], 0, x.shape[0] - 1)
        return x[idx]

    q_r, k_r, v_r = gather(query[0]), gather(key[0]), gather(value[0])
    b_r, g_r = gather(beta[0]), gather(decay[0])
    valid = (ar[None, :] < lens[:, None]).astype(q_r.dtype)  # [rows, T]

    def mask(x):
        shape = valid.shape + (1,) * (x.ndim - 2)
        return jnp.where(valid.reshape(shape), x, 0.0)

    q_r, k_r, v_r, b_r, g_r = mask(q_r), mask(k_r), mask(v_r), mask(b_r), mask(g_r)

    # The pool may hold ``C`` candidate rows per request row after the live rows (prefix-major). Apply a pending
    # commit (accepted prefix -> live row) first, then work on the live rows only.
    pool_leaves = (cache_view.q_conv_state, cache_view.k_conv_state, cache_view.v_conv_state, cache_view.recurrent_state)
    pool = int(cache_view.recurrent_state.shape[0])
    cand = (pool - rows) // rows if rows > 0 and pool > rows else 0
    if cand > 0 and spec_recurrent_commit is not None:
        commit = jnp.asarray(spec_recurrent_commit, dtype=jnp.int32)
        if commit.shape[1] < rows:
            commit = jnp.pad(commit, ((0, 0), (0, rows - commit.shape[1])))
        prefix_sel = commit[1, :rows]
        commit_active = (commit[0, :rows] > 0) & (prefix_sel > 0) & (prefix_sel <= cand)
        src_rows = jnp.clip(rows + (prefix_sel - 1) * rows + jnp.arange(rows), 0, pool - 1)

        def _apply_commit(leaf):
            mask_c = commit_active.reshape((rows,) + (1,) * (leaf.ndim - 1))
            live_rows = jnp.where(mask_c, leaf[src_rows], leaf[:rows]).astype(leaf.dtype)
            return lax.dynamic_update_slice_in_dim(leaf, live_rows, 0, axis=0)

        pool_leaves = tuple(_apply_commit(leaf) for leaf in pool_leaves)
    base_leaves = tuple(leaf[:rows] for leaf in pool_leaves) if pool != rows else pool_leaves
    if context_lens is None:
        context_before = jnp.zeros_like(lens)
    else:
        context_before = jnp.asarray(context_lens, dtype=jnp.int32)[:rows] - lens
    spec_mask = (lens >= 2) & (lens <= cand) & (context_before > 0)

    conv_states = base_leaves[:3]
    cand_conv = []
    if context_lens is None:
        has_prefix = lens <= 1
    else:
        has_prefix = context_lens > lens
    conv_outs = []
    new_conv = []
    for x_in, kern, state in zip((q_r, k_r, v_r), conv_kernels, conv_states, strict=True):
        x_rows = x_in.astype(jnp.float32)
        st = state.astype(jnp.float32)  # [rows, d, d_conv]
        d_conv = st.shape[-1]
        prefix = jnp.where(has_prefix[:, None, None], st[:, :, 1:], 0.0).transpose(0, 2, 1)  # [rows, dc-1, d]
        stream = jnp.concatenate([prefix, x_rows], axis=1)  # [rows, T + dc - 1, d]
        out = jnp.zeros_like(stream[:, :seq, :])
        kern = kern.astype(jnp.float32)
        for m in range(d_conv):
            out = out + stream[:, m : m + seq, :] * kern[:, m][None, None, :]
        conv_outs.append(jax.nn.silu(out).astype(conv_dtype))
        # New state = the trailing window ending at this row's LAST real token
        # (stream offset dc-1+lens), not the padded bucket tail.
        win_start = jnp.clip(lens - 1, 0, seq - 1)
        win_idx = win_start[:, None] + jnp.arange(d_conv)[None, :]
        candidate = stream[jnp.arange(rows)[:, None], win_idx].transpose(0, 2, 1)  # [rows, d, d_conv]
        keep = (lens > 0)[:, None, None]
        new_conv.append(jnp.where(keep, candidate, st).astype(state.dtype))
        if cand > 0:
            # Window after prefix ``t + 1``: stream offsets ``[t, t + d_conv)`` (decode-exact; raw inputs).
            c_idx = jnp.clip(jnp.arange(cand)[:, None] + jnp.arange(d_conv)[None, :], 0, stream.shape[1] - 1)
            cand_conv.append(stream[:, c_idx].transpose(1, 0, 3, 2))  # [cand, rows, d, d_conv]

    num_heads = b_r.shape[-1]
    head_k = q_r.shape[-1] // num_heads
    head_v = v_r.shape[-1] // num_heads
    qh = conv_outs[0].reshape(rows, seq, num_heads, head_k).transpose(0, 2, 1, 3)  # [R, H, T, K]
    kh = conv_outs[1].reshape(rows, seq, num_heads, head_k).transpose(0, 2, 1, 3)
    vh = conv_outs[2].reshape(rows, seq, num_heads, head_v).transpose(0, 2, 1, 3)
    bh = b_r.transpose(0, 2, 1)  # [R, H, T]
    gh = g_r.reshape(rows, seq, num_heads, head_k).transpose(0, 2, 1, 3)
    init_state = base_leaves[3]  # [R, H, K, V]
    rec_dtype = init_state.dtype
    zero_stack = jnp.zeros((cand, *init_state.shape), rec_dtype) if cand > 0 else None

    def _spec_stack():
        # Decode-exact per-prefix states of the speculative rows, for branches whose recurrence does not produce them.
        spec_lens = jnp.where(spec_mask, jnp.minimum(lens, cand), 0)
        return _short_window_kda_per_channel(qh, kh, vh, bh, gh, init_state, spec_lens, cand, stack_steps=cand)[2]

    def _single_path():
        out1, new_state = _single_step_kda_per_channel_fwd_bthd(
            query=qh[:, :, :1, :].transpose(0, 2, 1, 3),
            key=kh[:, :, :1, :].transpose(0, 2, 1, 3),
            value=vh[:, :, :1, :].transpose(0, 2, 1, 3),
            beta=bh[:, :, :1].transpose(0, 2, 1),
            decay=gh[:, :, :1, :].transpose(0, 2, 1, 3),
            recurrent_state=init_state,
            use_qk_l2norm=True,
        )
        out1 = jnp.transpose(out1, (0, 2, 1, 3)).astype(jnp.float32)  # [R, H, 1, V]
        out1 = jnp.pad(out1, ((0, 0), (0, 0), (0, seq - 1), (0, 0)))
        return out1, new_state.astype(jnp.float32), zero_stack

    def _chunk_path():
        # Rows are segments of one packed ``seq``-token stream, so their valid tokens total
        # at most ``seq``: segment-aligned chunks skip the per-row bucket padding.
        out, new_state = _packed_segment_kda_per_channel(
            query=qh,
            key=kh,
            value=vh,
            beta=bh,
            decay=gh,
            initial_state=init_state,
            token_valid=valid,
            chunk_size=chunk_size,
            max_tokens=seq,
        )
        return out, new_state.astype(jnp.float32), (_spec_stack() if cand > 0 else None)

    short_max = min(_SHORT_WINDOW_MAX, seq)
    if short_max > 1:

        def _short_path():
            # Short windows (speculative verify: 1 + draft tokens per row) run the decode recurrence per token:
            # cheaper than building full chunks, and arithmetically identical to plain decode.
            if cand > 0:
                return _short_window_kda_per_channel(qh, kh, vh, bh, gh, init_state, lens, short_max, stack_steps=cand)
            out_s, state_s = _short_window_kda_per_channel(qh, kh, vh, bh, gh, init_state, lens, short_max)
            return out_s, state_s, None

        max_len = jnp.max(lens)
        branch = jnp.where(max_len <= 1, 0, jnp.where(max_len <= short_max, 1, 2))
        out_h, new_recurrent, cand_states = lax.switch(branch, (_single_path, _short_path, _chunk_path))
    else:
        out_h, new_recurrent, cand_states = lax.cond(jnp.all(lens <= 1), _single_path, _chunk_path)

    # Gather per-row outputs back into the packed stream.
    out_rows = out_h.transpose(0, 2, 1, 3)  # [R, T, H, V]
    seg_ids = jnp.clip(jnp.sum(query_start_loc[1:, None] <= ar[None, :], axis=0), 0, rows - 1)  # [T]
    pos_in_row = jnp.clip(ar - query_start_loc[seg_ids], 0, seq - 1)
    in_stream = ar < query_start_loc[-1]
    out_packed = jnp.where(in_stream[:, None, None], out_rows[seg_ids, pos_in_row], 0.0)[None]  # [1, T, H, V]

    def _merge(pool_leaf, new_live, cand_vals):
        # Live rows <- this step's final state; speculative rows' candidate rows <- per-prefix states; rest kept.
        new_live = new_live.astype(pool_leaf.dtype)
        if pool_leaf.shape[0] == rows:
            return new_live
        used = rows * (1 + cand)
        tail_shape = pool_leaf.shape[1:]
        parts = [new_live]
        if cand > 0:
            old = pool_leaf[rows:used].reshape((cand, rows, *tail_shape))
            m = spec_mask.reshape((1, rows) + (1,) * len(tail_shape))
            parts.append(jnp.where(m, cand_vals.astype(pool_leaf.dtype), old).reshape((cand * rows, *tail_shape)))
        if pool_leaf.shape[0] > used:
            parts.append(pool_leaf[used:])
        return jnp.concatenate(parts, axis=0)

    new_cache_view = cache_view.update_kda_states(
        new_q_conv_state=_merge(pool_leaves[0], new_conv[0], cand_conv[0] if cand > 0 else None),
        new_k_conv_state=_merge(pool_leaves[1], new_conv[1], cand_conv[1] if cand > 0 else None),
        new_v_conv_state=_merge(pool_leaves[2], new_conv[2], cand_conv[2] if cand > 0 else None),
        new_recurrent_state=_merge(pool_leaves[3], new_recurrent, cand_states),
    )
    return out_packed, new_cache_view


@OperationRegistry.register
class KernelDeltaAttnOp(OperationImpl):
    """Kernel Delta Attention (KDA) operation for Kimi Linear.

    Implements the KDA mechanism for efficient linear attention:
    - Training mode: Uses chunked algorithm for O(N) complexity
    - Inference mode: Uses recurrent update for single-token generation

    The KDA updates state as:
        h_t = decay * h_{t-1} + beta_t * (v_t - k_t @ h_{t-1}) * k_t
        o_t = h_t @ q_t

    Where:
    - beta_t is a learned gating signal (sigmoid)
    - decay = -exp(A_log) * softplus(gate + dt_bias)

    Two decay layouts are supported: the Kimi-Linear **per-head** decay of
    shape ``(batch, seq_len, num_heads)`` (default), and the GLM-5-Next
    **per-channel** decay of shape ``(batch, seq_len, num_heads, head_dim)``
    — detected from ``decay.ndim == 4`` (or forced with
    ``per_channel_decay=True``) and dispatched to dedicated recurrent /
    chunked / single-step kernels that scale the recurrent state per channel.

    Registered under the names "kda" and "kernel_delta_attention".

    Example:
        >>> from easydel.operations import OperationMetadata, OperationRegistry
        >>> metadata = OperationMetadata(runtime_dtype=jnp.float16)
        >>> kda_op = OperationRegistry.create("kda", metadata)
        >>> output = kda_op(
        ...     query=query,
        ...     key=key,
        ...     value=value,
        ...     beta=beta,
        ...     decay=decay,
        ...     chunk_size=64,
        ... )
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Returns the registered name of this operation.

        Returns:
            Tuple of names: ("kda", "kernel_delta_attention")
        """
        return ("kda", "kernel_delta_attention")

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for KernelDeltaAttnOp (KDA).

        KDA is a recurrent/linear attention mechanism similar to GDR that requires:
        - Basic metadata plus state management fields
        - Recurrent or Hybrid cache types for state persistence
        - Uses KDACacheView for state management
        """
        return (
            RequirementsBuilder("kernel_delta_attention")
            .require_metadata(
                MetadataField.SEQ_LENS
                | MetadataField.POSITIONS
                | MetadataField.HAS_INITIAL_STATE
                | MetadataField.STATE_INDICES
            )
            .optional_metadata(MetadataField.LOGITS_INDICES)
            .support_cache(CacheType.RECURRENT | CacheType.HYBRID)
            .use_cache_view(KDACacheView)
            .build()
        )

    @jax.named_scope("easydel-kda-native")
    def forward_native(
        self,
        query: Float[Array, "batch seq_len num_heads head_dim"],
        key: Float[Array, "batch seq_len num_heads head_dim"],
        value: Float[Array, "batch seq_len num_heads d_state"],
        beta: Float[Array, "batch seq_len num_heads"],
        decay: Float[Array, "batch seq_len num_heads"] | Float[Array, "batch seq_len num_heads head_dim"] | None = None,
        q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None,
        recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
        segment_ids: Array | None = None,
        chunk_size: int = 64,
        per_channel_decay: bool = False,
        continue_initial_state: bool = False,
        chunk_impl: str = "scan",
        chunk_precision: str = "highest",
        **kwargs,
    ) -> KDAOutput:
        """Forward pass for KDA attention.

        Automatically selects between chunked (training) and recurrent (inference)
        modes based on sequence length.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, d_state]
            beta: Gating tensor [batch, seq_len, num_heads]
            decay: Optional decay factors — per-head
                [batch, seq_len, num_heads] (Kimi Linear) or per-channel
                [batch, seq_len, num_heads, head_dim] (GLM-5-Next).
            q_conv_state: Optional Q convolution state (passed through, not used here)
            k_conv_state: Optional K convolution state (passed through, not used here)
            v_conv_state: Optional V convolution state (passed through, not used here)
            recurrent_state: Optional recurrent state for inference
            chunk_size: Chunk size for training mode (default: 64)
            per_channel_decay: Force the per-channel-decay kernels; inferred
                automatically when ``decay.ndim == 4``.
            continue_initial_state: For cached multi-token segmented continuation,
                preserve incoming state into the first valid segment and over padding.
                Real within-chunk segment changes still reset memory. False keeps
                the packed-training reset behavior.
            chunk_impl: Per-channel chunked (multi-token) implementation, one of
                :data:`KDA_CHUNK_IMPLS`: ``"scan"`` (default, the XLA chunk scan) or
                ``"kda_chunk"`` (fused ejkernel kernel: Pallas TPU forward + backward,
                its ``block_h`` / ``backward`` / ``bwd_reduce`` taken from the
                ``"kda_chunk"`` entry of ``OperationMetadata.operation_configs``).
            chunk_precision: Precision of the non-inverse dots on the ``"kda_chunk"``
                path: ``"highest"`` (default) or ``"default"`` (bf16 MXU passes).
            **kwargs: Additional ignored arguments

        Returns:
            KDAOutput containing attention outputs and updated states
        """
        if chunk_impl not in KDA_CHUNK_IMPLS:
            raise ValueError(f"chunk_impl must be one of {KDA_CHUNK_IMPLS}, got {chunk_impl!r}")
        if chunk_precision not in ("highest", "default"):
            raise ValueError(f"chunk_precision must be 'highest' or 'default', got {chunk_precision!r}")
        per_channel = per_channel_decay or (decay is not None and decay.ndim == 4)
        if per_channel and decay is None:
            # The per-channel kernels have no zero-decay fallback (zero decay
            # would mean "no memory decay", a different operation). The
            # per-head family zero-fills None; keep that contract there.
            raise ValueError(
                "per_channel_decay=True requires explicit `decay` [batch, seq_len, num_heads, head_dim]; got None."
            )
        seq_len = query.shape[1]
        shardings = None
        mesh = self.metadata.mesh

        if mesh is not None:
            with mesh:
                mode = self.get_mode(query=query, BTHD=True)
                shardings = self.metadata.get_shardings(mode, layout="bthd")
                # The constraints below sanitize abstract policy axes against the
                # actual mesh and tensor shape. Use that same effective query
                # layout for the chunk-local shard_map, not the raw policy spec.
                query_spec = sanitize_partition_spec_for_mesh_and_shape(shardings.query, mesh=mesh, shape=query.shape)
                query = with_sharding_constraint(arr=query, sharding=query_spec, mesh=mesh)
                key = with_sharding_constraint(arr=key, sharding=shardings.key, mesh=mesh)
                value = with_sharding_constraint(arr=value, sharding=shardings.value, mesh=mesh)
                if segment_ids is not None and segment_ids.ndim == 2 and len(tuple(query_spec)) > 0:
                    # Match the query's batch sharding once so the chunk loop never reshards the ids.
                    segment_ids = with_sharding_constraint(
                        arr=segment_ids, sharding=PartitionSpec(next(iter(query_spec)), None), mesh=mesh
                    )

        runtime_dtype = self.metadata.runtime_dtype
        query = query.astype(runtime_dtype)
        key = key.astype(runtime_dtype)
        value = value.astype(runtime_dtype)

        beta = beta.astype(runtime_dtype)
        if decay is not None:
            # Keep the log-decay in fp32 (HF passes it un-cast): g in [-5, 0)
            # at bf16 precision biases exp(g_cumsum) by up to ~0.02 absolute
            # per chunk, compounding over long recurrences. The per-channel
            # kernels upcast to f32 internally regardless.
            decay = decay.astype(jnp.float32)

        is_inference = seq_len == 1

        if is_inference and recurrent_state is not None:
            if per_channel:
                outputs, new_recurrent_state = _single_step_kda_per_channel_fwd_bthd(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    decay=decay,
                    recurrent_state=recurrent_state,
                    use_qk_l2norm=True,
                )
            else:
                outputs, new_recurrent_state = _single_step_kda_fwd_bthd(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    decay=decay,
                    recurrent_state=recurrent_state,
                    use_qk_l2norm=True,
                )
            if segment_ids is not None:
                # A padding token (negative segment id) must neither advance the cached state nor
                # emit output; the single-step kernels above ignore segment ids entirely.
                valid = normalize_packed_segment_ids(segment_ids, 1, pad_from_last=False)[:, 0] >= 0
                new_recurrent_state = jnp.where(
                    valid.reshape((-1,) + (1,) * (new_recurrent_state.ndim - 1)),
                    new_recurrent_state,
                    recurrent_state.astype(new_recurrent_state.dtype),
                )
                outputs = jnp.where(valid.reshape((-1,) + (1,) * (outputs.ndim - 1)), outputs, 0).astype(outputs.dtype)
        else:
            query = query.transpose(0, 2, 1, 3)
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)

            if beta.ndim == 3:
                beta = beta.transpose(0, 2, 1)

            if decay is not None and decay.ndim == 3:
                decay = decay.transpose(0, 2, 1)
            elif decay is not None and decay.ndim == 4:
                decay = decay.transpose(0, 2, 1, 3)

            if segment_ids is None or per_channel:
                # The per-channel chunked kernel handles packed segments (state resets at
                # boundaries) directly; only the per-head family falls back to the recurrence.
                chunk_fwd = _chunk_kda_per_channel_fwd if per_channel else _chunk_kda_fwd
                extra = {}
                if per_channel:
                    extra = {
                        "segment_ids": segment_ids,
                        "kernel_mesh": mesh,
                        "continue_initial_state": continue_initial_state,
                    }
                    if chunk_impl == "kda_chunk":
                        chunk_fwd = _chunk_kda_per_channel_fused
                        extra["precision"] = chunk_precision
                        extra["cfg"] = self.metadata.get_operation_config("kda_chunk")
                    if shardings is not None and mesh is not None:

                        def _eff(axes):
                            # Axes spanning a single device (e.g. ``sp`` of size 1) are not sharded.
                            if axes is None:
                                return None
                            names = axes if isinstance(axes, tuple) else (axes,)
                            n = 1
                            for name in names:
                                n *= int(mesh.shape[name])
                            return None if n == 1 else axes

                        qs = tuple(_eff(a) for a in tuple(query_spec) + (None,) * (4 - len(tuple(query_spec))))
                        # BTHD spec -> (batch, head) axes; seq / head_dim must be unsharded.
                        if qs[1] is None and qs[3] is None:
                            extra["kernel_lead_specs"] = (qs[0], qs[2])
                outputs, new_recurrent_state = chunk_fwd(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    decay=decay,
                    chunk_size=chunk_size,
                    initial_state=recurrent_state,
                    use_qk_l2norm=True,
                    **extra,
                )
            else:
                recurrent_fwd = _recurrent_kda_per_channel_fwd if per_channel else _recurrent_kda_fwd
                outputs, new_recurrent_state = recurrent_fwd(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    decay=decay,
                    initial_state=recurrent_state,
                    use_qk_l2norm=True,
                    segment_ids=segment_ids,
                    continue_initial_state=continue_initial_state,
                )

            outputs = outputs.transpose(0, 2, 1, 3)

        if mesh is not None and shardings is not None:
            with mesh:
                outputs = with_sharding_constraint(arr=outputs, sharding=shardings.output, mesh=mesh)

        return KDAOutput(
            attention_outputs=outputs,
            attention_weights=None,
            q_conv_state=q_conv_state,
            k_conv_state=k_conv_state,
            v_conv_state=v_conv_state,
            recurrent_state=new_recurrent_state,
        )

    def forward_tpu(self, *args, **kwargs) -> KDAOutput:
        """TPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> KDAOutput:
        """GPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> KDAOutput:
        """CPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> KDAOutput:
        """CUDA forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> KDAOutput:
        """ROCm forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch seq_len num_heads head_dim"],
        key: Float[Array, "batch seq_len num_heads head_dim"],
        value: Float[Array, "batch seq_len num_heads d_state"],
        beta: Float[Array, "batch seq_len num_heads"],
        decay: Float[Array, "batch seq_len num_heads"] | Float[Array, "batch seq_len num_heads head_dim"] | None = None,
        q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None,
        recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
        segment_ids: Array | None = None,
        chunk_size: int = 64,
        per_channel_decay: bool = False,
        continue_initial_state: bool = False,
        chunk_impl: str = "scan",
        chunk_precision: str = "highest",
        **kwargs,
    ) -> KDAOutput:
        """Execute the KDA operation.

        Dispatches to appropriate backend via parent __call__.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, d_state]
            beta: Gating tensor [batch, seq_len, num_heads]
            decay: Optional decay factors — per-head
                [batch, seq_len, num_heads] or per-channel
                [batch, seq_len, num_heads, head_dim].
            q_conv_state: Optional Q convolution state
            k_conv_state: Optional K convolution state
            v_conv_state: Optional V convolution state
            recurrent_state: Optional recurrent state
            chunk_size: Chunk size for training mode
            per_channel_decay: Force the per-channel-decay kernels; inferred
                automatically when ``decay.ndim == 4``.
            continue_initial_state: For cached multi-token segmented continuation,
                preserve incoming state into the first valid segment and over padding.
                Real within-chunk segment changes still reset memory. False keeps
                the packed-training reset behavior.
            chunk_impl: ``"scan"`` (default) or ``"kda_chunk"``; see :meth:`forward_native`.
            chunk_precision: ``"highest"`` (default) or ``"default"``; see :meth:`forward_native`.
            **kwargs: Additional arguments

        Returns:
            KDAOutput with attention outputs and updated states
        """
        return super().__call__(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            q_conv_state=q_conv_state,
            k_conv_state=k_conv_state,
            v_conv_state=v_conv_state,
            recurrent_state=recurrent_state,
            segment_ids=segment_ids,
            chunk_size=chunk_size,
            per_channel_decay=per_channel_decay,
            continue_initial_state=continue_initial_state,
            chunk_impl=chunk_impl,
            chunk_precision=chunk_precision,
            **kwargs,
        )


if __name__ == "__main__":
    from jax import random as jr

    from easydel.infra import EasyDeLBaseConfig

    print("Testing KernelDeltaAttnOp...")

    batch, seq_len, num_heads, head_dim, d_state = 2, 128, 8, 64, 64

    key = jr.PRNGKey(0)
    k1, k2, k3, k4, k5 = jr.split(key, 5)

    query = jr.normal(k1, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.1
    key_tensor = jr.normal(k2, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.1
    value = jr.normal(k3, (batch, seq_len, num_heads, d_state), dtype=jnp.float32) * 0.1
    beta = jax.nn.sigmoid(jr.normal(k4, (batch, seq_len, num_heads), dtype=jnp.float32))
    decay = jr.normal(k5, (batch, seq_len, num_heads), dtype=jnp.float32) * 0.01

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )

    kda_op = KernelDeltaAttnOp(metadata)

    print("Testing training mode (chunked)...")
    output = kda_op(
        query=query,
        key=key_tensor,
        value=value,
        beta=beta,
        decay=decay,
        chunk_size=32,
    )
    print(f"  Output shape: {output.attention_outputs.shape}")
    print(f"  Recurrent state shape: {output.recurrent_state.shape}")
    print(f"  Output range: [{output.attention_outputs.min():.4f}, {output.attention_outputs.max():.4f}]")

    print("Testing inference mode (recurrent)...")
    query_single = query[:, :1, :, :]
    key_single = key_tensor[:, :1, :, :]
    value_single = value[:, :1, :, :]
    beta_single = beta[:, :1, :]
    decay_single = decay[:, :1, :]

    output_infer = kda_op(
        query=query_single,
        key=key_single,
        value=value_single,
        beta=beta_single,
        decay=decay_single,
        recurrent_state=output.recurrent_state,
    )
    print(f"  Output shape: {output_infer.attention_outputs.shape}")
    print(f"  Recurrent state shape: {output_infer.recurrent_state.shape}")

    print("Testing chunked vs recurrent consistency...")

    out_recurrent, state_recurrent = _recurrent_kda_fwd(
        query.transpose(0, 2, 1, 3)[:, :, :32, :],
        key_tensor.transpose(0, 2, 1, 3)[:, :, :32, :],
        value.transpose(0, 2, 1, 3)[:, :, :32, :],
        beta.transpose(0, 2, 1)[:, :, :32],
        decay.transpose(0, 2, 1)[:, :, :32],
        use_qk_l2norm=True,
    )

    out_chunk, state_chunk = _chunk_kda_fwd(
        query.transpose(0, 2, 1, 3)[:, :, :32, :],
        key_tensor.transpose(0, 2, 1, 3)[:, :, :32, :],
        value.transpose(0, 2, 1, 3)[:, :, :32, :],
        beta.transpose(0, 2, 1)[:, :, :32],
        decay.transpose(0, 2, 1)[:, :, :32],
        chunk_size=16,
        use_qk_l2norm=True,
    )

    import numpy as np

    max_diff = np.abs(np.array(out_chunk) - np.array(out_recurrent)).max()
    print(f"  Max diff between chunked and recurrent: {max_diff:.2e}")

    if max_diff < 1e-4:
        print("  Consistency check: PASS")
    else:
        print("  Consistency check: FAIL")

    print("All tests passed!")
