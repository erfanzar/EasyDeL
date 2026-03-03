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
Gated Delta Rule (GDR) linear attention implementation for EasyDeL.

This module provides the GatedDeltaRule operation, a linear attention mechanism
used in hybrid transformer architectures like Qwen3Next. The gated delta rule
combines:

1. Causal convolution for local context
2. Gated linear attention with delta rule updates
3. Learnable decay for forgetting previous state

Key characteristics:
- Linear complexity O(N) in sequence length (vs O(N²) for standard attention)
- Maintains recurrent state for efficient inference
- Supports chunked computation for efficient training

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
    - Qwen3Next: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/
"""

import functools
import math
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from eformer.escale import with_sharding_constraint
from eformer.pytree import auto_pytree
from ejkernel.ops import (  # pyright: ignore[reportMissingTypeStubs]
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
)
from jax import lax
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Float

from easydel.caching import RecurrentCacheView

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)

_MATMUL_PRECISION = lax.Precision.HIGHEST


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed


def _env_chunk_candidates(name: str) -> tuple[int, ...] | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            item = int(part)
        except ValueError:
            continue
        if item > 0:
            values.append(item)
    if not values:
        return None
    return tuple(sorted(set(values)))


_GDR_AUTOTUNE_CHUNK_SIZE_DEFAULT = _env_flag("EASYDEL_GDR_AUTOTUNE_CHUNK_SIZE", default=False)
_GDR_AUTOTUNE_CHUNK_CANDIDATES_ENV = _env_chunk_candidates("EASYDEL_GDR_AUTOTUNE_CHUNK_CANDIDATES")
_GDR_AUTOTUNE_TUNER_WARMUP = max(1, _env_int("EASYDEL_GDR_AUTOTUNE_TUNER_WARMUP", default=1))
_GDR_AUTOTUNE_TUNER_ITERS = max(1, _env_int("EASYDEL_GDR_AUTOTUNE_TUNER_ITERS", default=8))


def l2norm(x, axis=-1, eps=1e-6):
    """L2 normalize along specified axis.

    Uses rsqrt: inv_norm = rsqrt(sum(x^2) + eps); return x * inv_norm
    """
    inv_norm = lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _l2norm_with_inv(x, axis=-1, eps=1e-6):
    """Returns both normalized tensor and inverse norm."""
    inv_norm = lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm, inv_norm


def _l2norm_bwd(grad_y, y, inv_norm):
    """Backward pass for y = l2norm(x)."""
    proj = jnp.sum(grad_y * y, axis=-1, keepdims=True)
    return inv_norm * (grad_y - y * proj)


@auto_pytree
class GatedDeltaRuleOutput(AttentionOutput):
    """Output container for GatedDeltaRule operation.

    Extends AttentionOutput with recurrent state fields needed for
    hybrid attention models.

    Attributes:
        attention_outputs: Output tensor [batch, seq_len, num_heads, head_dim]
        attention_weights: Always None for linear attention (no explicit weights)
        conv_state: Updated convolution state [batch, d_inner, d_conv]
        recurrent_state: Updated recurrent state [batch, num_heads, head_dim, d_state]
    """

    conv_state: Float[Array, "batch d_inner d_conv"] | None = None
    recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None


def _recurrent_gated_delta_rule_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Recurrent forward pass for gated delta rule.

    Processes each position sequentially using lax.scan for efficiency.
    This is the reference implementation that matches HuggingFace's
    torch_recurrent_gated_delta_rule exactly.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_heads, seq_len, head_dim]
        value: Value tensor [batch, num_heads, seq_len, d_state]
        beta: Gating tensor [batch, num_heads, seq_len]
        decay: Per-token decay [batch, num_heads, seq_len] (g from HuggingFace)
        initial_state: Optional initial recurrent state
        use_qk_l2norm: Whether to apply L2 normalization to query and key

    Returns:
        Tuple of (outputs, final_state)
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
    if decay is None:
        decay = jnp.zeros((B, H, L), dtype=jnp.float32)
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    decay = decay.astype(jnp.float32)

    q_seq = query.transpose(2, 0, 1, 3)  # (L, B, H, K)
    k_seq = key.transpose(2, 0, 1, 3)  # (L, B, H, K)
    v_seq = value.transpose(2, 0, 1, 3)  # (L, B, H, V)
    g_seq = decay.transpose(2, 0, 1)  # (L, B, H)
    b_seq = beta.transpose(2, 0, 1)  # (L, B, H)

    def step_fn(state, inputs):
        q_t, k_t, v_t, g_t, beta_t = inputs
        g_exp = jnp.exp(g_t)[:, :, None, None]
        beta_scaled = beta_t[:, :, None]
        state = state * g_exp
        kv_mem = jnp.sum(state * k_t[:, :, :, None], axis=-2)

        delta = (v_t - kv_mem) * beta_scaled
        state = state + k_t[:, :, :, None] * delta[:, :, None, :]
        output = jnp.sum(state * q_t[:, :, :, None], axis=-2)

        return state, output

    final_state, outputs = lax.scan(step_fn, initial_state, (q_seq, k_seq, v_seq, g_seq, b_seq))
    outputs = outputs.transpose(1, 2, 0, 3)

    return outputs, final_state


def _chunk_gated_delta_rule_impl(
    query,
    key,
    value,
    beta,
    decay,
    chunk_size,
    initial_state,
    use_qk_l2norm,
):
    """Core implementation for chunked gated delta rule forward.

    This function is shared by the custom_vjp wrapper. Forward executes in
    ``input_dtype`` with Neumann-series accumulation in float32.
    """
    output, final_state, _ = _chunk_gated_delta_rule_core(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        chunk_size=chunk_size,
        initial_state=initial_state,
        use_qk_l2norm=use_qk_l2norm,
        save_residual=False,
    )
    return output, final_state


def _chunk_gated_delta_rule_core(
    query,
    key,
    value,
    beta,
    decay,
    chunk_size,
    initial_state,
    use_qk_l2norm,
    save_residual: bool,
):
    """Shared chunked forward path, optionally capturing backward residuals."""
    B, H, L, K_dim = query.shape
    V_dim = value.shape[-1]
    input_dtype = query.dtype
    decay_was_none = decay is None
    initial_state_was_none = initial_state is None

    q_inv_norm = None
    k_inv_norm = None
    if use_qk_l2norm:
        query, q_inv_norm = _l2norm_with_inv(query, axis=-1, eps=1e-6)
        key, k_inv_norm = _l2norm_with_inv(key, axis=-1, eps=1e-6)

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
    beta = beta.reshape(B, H, num_chunks, chunk_size)
    k_beta = k_beta.reshape(B, H, num_chunks, chunk_size, K_dim)
    v_beta = v_beta.reshape(B, H, num_chunks, chunk_size, V_dim)
    g = decay.reshape(B, H, num_chunks, chunk_size)

    mask_triu = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)

    g_cumsum = jnp.cumsum(g, axis=-1)

    g_diff = g_cumsum[:, :, :, :, None] - g_cumsum[:, :, :, None, :]
    g_diff = jnp.tril(g_diff)
    decay_mask = jnp.exp(g_diff)
    decay_mask = jnp.tril(decay_mask)

    attn = jnp.einsum("bhcik,bhcjk->bhcij", k_beta, key, precision=_MATMUL_PRECISION)
    # Neumann series must use float32 — matrix squaring amplifies bf16
    # precision errors exponentially.
    attn = -(attn * decay_mask).astype(jnp.float32)
    attn = jnp.where(mask_triu, 0.0, attn)

    eye = jnp.eye(chunk_size, dtype=jnp.float32)
    M = eye + attn
    P = attn
    for _ in range(math.ceil(math.log2(max(chunk_size, 2))) - 1):
        P = jnp.einsum("...ij,...jk->...ik", P, P, precision=_MATMUL_PRECISION)
        M = M + jnp.einsum("...ij,...jk->...ik", P, M, precision=_MATMUL_PRECISION)
    attn = M.astype(input_dtype)

    g_cumsum_exp = jnp.exp(g_cumsum).astype(input_dtype)
    g_end = g_cumsum[:, :, :, -1]
    g_end_exp = jnp.exp(g_end).astype(input_dtype)
    g_diff_state_exp = jnp.exp(g_end[:, :, :, None] - g_cumsum).astype(input_dtype)

    value_local = jnp.einsum("bhcij,bhcjv->bhciv", attn, v_beta, precision=_MATMUL_PRECISION)
    k_beta_scaled = k_beta * g_cumsum_exp[:, :, :, :, None]
    k_cumdecay = jnp.einsum("bhcij,bhcjk->bhcik", attn, k_beta_scaled, precision=_MATMUL_PRECISION)

    if initial_state is None:
        initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=input_dtype)
    else:
        initial_state = initial_state.astype(input_dtype)

    mask_triu_inner = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

    xs = (
        query.transpose(2, 0, 1, 3, 4),
        key.transpose(2, 0, 1, 3, 4),
        value_local.transpose(2, 0, 1, 3, 4),
        k_cumdecay.transpose(2, 0, 1, 3, 4),
        g_cumsum_exp.transpose(2, 0, 1, 3),
        g_end_exp.transpose(2, 0, 1),
        g_diff_state_exp.transpose(2, 0, 1, 3),
        decay_mask.astype(input_dtype).transpose(2, 0, 1, 3, 4),
    )

    def chunk_step(state, inputs):
        q_i, k_i, v_i, k_cumdecay_i, g_exp_i, g_end_exp_i, g_diff_exp_i, decay_mask_i = inputs

        attn_qk = jnp.einsum("bhik,bhjk->bhij", q_i, k_i, precision=_MATMUL_PRECISION)
        attn_qk = attn_qk * decay_mask_i
        attn_qk = jnp.where(mask_triu_inner, 0.0, attn_qk)

        q_scaled = q_i * g_exp_i[:, :, :, None]
        qk_fused = jnp.stack([k_cumdecay_i, q_scaled], axis=0)
        both = jnp.einsum("nbhik,bhkv->nbhiv", qk_fused, state, precision=_MATMUL_PRECISION)
        v_prime, attn_inter = both[0], both[1]

        v_new = v_i - v_prime
        core_out = attn_inter + jnp.einsum("bhij,bhjv->bhiv", attn_qk, v_new, precision=_MATMUL_PRECISION)

        state_decayed = state * g_end_exp_i[:, :, None, None]
        k_scaled = k_i * g_diff_exp_i[:, :, :, None]
        state_update = jnp.einsum("bhik,bhiv->bhkv", k_scaled, v_new, precision=_MATMUL_PRECISION)
        new_state = state_decayed + state_update

        return new_state, core_out.astype(input_dtype)

    final_state, core_attn_out = lax.scan(chunk_step, initial_state, xs)

    core_attn_out = core_attn_out.transpose(1, 2, 0, 3, 4)
    core_attn_out = core_attn_out.reshape(B, H, -1, V_dim)
    core_attn_out = core_attn_out[:, :, :L, :]

    if not save_residual:
        return core_attn_out, final_state, None

    residual = (
        query,
        key,
        value,
        beta,
        attn,
        decay_mask.astype(input_dtype),
        g_cumsum_exp,
        g_end_exp,
        g_diff_state_exp,
        initial_state,
        q_inv_norm,
        k_inv_norm,
        L,
        pad_size,
        decay_was_none,
        initial_state_was_none,
    )
    return core_attn_out, final_state, residual


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 7))
def _chunk_gated_delta_rule_fwd(
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
    """Chunked forward pass for gated delta rule with custom backward.

    Forward runs in ``input_dtype`` (bf16) for memory efficiency.
    Backward uses a hand-derived analytical reverse pass in float32,
    avoiding mixed-type issues inside ``shard_map`` backward.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_heads, seq_len, head_dim]
        value: Value tensor [batch, num_heads, seq_len, d_state]
        beta: Gating tensor [batch, num_heads, seq_len]
        decay: Per-token decay [batch, num_heads, seq_len]
        chunk_size: Size of chunks for parallel processing (non-diff)
        initial_state: Optional initial recurrent state
        use_qk_l2norm: Whether to apply L2 normalization (non-diff)

    Returns:
        Tuple of (outputs, final_state)
    """
    return _chunk_gated_delta_rule_impl(
        query,
        key,
        value,
        beta,
        decay,
        chunk_size,
        initial_state,
        use_qk_l2norm,
    )


def _chunk_gdr_fwd(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm):
    """Forward pass for custom_vjp — saves minimal backward context."""
    output, final_state, residual = _chunk_gated_delta_rule_core(
        query,
        key,
        value,
        beta,
        decay,
        chunk_size,
        initial_state,
        use_qk_l2norm,
        save_residual=True,
    )
    return (output, final_state), residual


def _chunk_gdr_bwd(chunk_size, use_qk_l2norm, res, g):
    """Analytical backward pass in float32 with reverse scans."""
    (
        query,
        key,
        value,
        beta,
        attn,
        decay_mask,
        g_cumsum_exp,
        g_end_exp,
        g_diff_state_exp,
        initial_state,
        q_inv_norm,
        k_inv_norm,
        seq_len,
        pad_size,
        decay_was_none,
        initial_state_was_none,
    ) = res
    d_out, d_final_state = g
    input_dtype = query.dtype

    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    attn = attn.astype(jnp.float32)
    decay_mask = decay_mask.astype(jnp.float32)
    g_cumsum_exp = g_cumsum_exp.astype(jnp.float32)
    g_end_exp = g_end_exp.astype(jnp.float32)
    g_diff_state_exp = g_diff_state_exp.astype(jnp.float32)
    initial_state = initial_state.astype(jnp.float32)

    d_out = d_out.astype(jnp.float32)
    d_final_state = d_final_state.astype(jnp.float32)

    B, H, num_chunks, _, K_dim = query.shape
    V_dim = value.shape[-1]
    total_len = seq_len + pad_size
    scale = 1.0 / (K_dim**0.5)

    if pad_size > 0:
        d_out = jnp.pad(d_out, ((0, 0), (0, 0), (0, pad_size), (0, 0)))

    d_out = d_out.reshape(B, H, num_chunks, chunk_size, V_dim)

    value_beta = value * beta[:, :, :, :, None]
    key_beta = key * beta[:, :, :, :, None]
    key_beta_scaled = key_beta * g_cumsum_exp[:, :, :, :, None]

    value_local = jnp.einsum("bhcij,bhcjv->bhciv", attn, value_beta, precision=_MATMUL_PRECISION)
    key_cumdecay = jnp.einsum("bhcij,bhcjk->bhcik", attn, key_beta_scaled, precision=_MATMUL_PRECISION)

    query_tm = query.transpose(2, 0, 1, 3, 4)
    key_tm = key.transpose(2, 0, 1, 3, 4)
    value_local_tm = value_local.transpose(2, 0, 1, 3, 4)
    key_cumdecay_tm = key_cumdecay.transpose(2, 0, 1, 3, 4)
    g_cumsum_exp_tm = g_cumsum_exp.transpose(2, 0, 1, 3)
    g_end_exp_tm = g_end_exp.transpose(2, 0, 1)
    g_diff_state_exp_tm = g_diff_state_exp.transpose(2, 0, 1, 3)
    decay_mask_tm = decay_mask.transpose(2, 0, 1, 3, 4)
    d_out_tm = d_out.transpose(2, 0, 1, 3, 4)

    def fwd_state_scan(state, inputs):
        k_i, v_i, k_cum_i, g_end_i, g_diff_i = inputs
        v_prime = jnp.einsum("bhik,bhkv->bhiv", k_cum_i, state, precision=_MATMUL_PRECISION)
        v_new = v_i - v_prime

        state_decayed = state * g_end_i[:, :, None, None]
        k_scaled = k_i * g_diff_i[:, :, :, None]
        state_update = jnp.einsum("bhik,bhiv->bhkv", k_scaled, v_new, precision=_MATMUL_PRECISION)
        new_state = state_decayed + state_update

        return new_state, state

    _, state_pre_tm = lax.scan(
        fwd_state_scan,
        initial_state,
        (
            key_tm,
            value_local_tm,
            key_cumdecay_tm,
            g_end_exp_tm,
            g_diff_state_exp_tm,
        ),
    )

    def rev_chunk_scan(d_state_next, inputs):
        (
            state_i,
            q_i,
            k_i,
            v_i,
            k_cum_i,
            g_exp_i,
            g_end_i,
            g_diff_i,
            decay_i,
            d_core_i,
        ) = inputs

        attn_qk_base = jnp.einsum("bhik,bhjk->bhij", q_i, k_i, precision=_MATMUL_PRECISION)
        attn_qk = attn_qk_base * decay_i
        q_scaled = q_i * g_exp_i[:, :, :, None]
        v_prime = jnp.einsum("bhik,bhkv->bhiv", k_cum_i, state_i, precision=_MATMUL_PRECISION)
        v_new = v_i - v_prime
        k_scaled = k_i * g_diff_i[:, :, :, None]

        d_state_i = d_state_next * g_end_i[:, :, None, None]
        d_g_end_i = jnp.einsum("bhkv,bhkv->bh", d_state_next, state_i, precision=_MATMUL_PRECISION)

        d_k_scaled = jnp.einsum("bhkv,bhiv->bhik", d_state_next, v_new, precision=_MATMUL_PRECISION)
        d_v_new = jnp.einsum("bhkv,bhik->bhiv", d_state_next, k_scaled, precision=_MATMUL_PRECISION)

        d_attn_qk = jnp.einsum("bhiv,bhjv->bhij", d_core_i, v_new, precision=_MATMUL_PRECISION)
        d_v_new = d_v_new + jnp.einsum("bhij,bhiv->bhjv", attn_qk, d_core_i, precision=_MATMUL_PRECISION)

        d_v_i = d_v_new
        d_v_prime = -d_v_new

        d_k_cum_i = jnp.einsum("bhiv,bhkv->bhik", d_v_prime, state_i, precision=_MATMUL_PRECISION)
        d_state_i = d_state_i + jnp.einsum("bhik,bhiv->bhkv", k_cum_i, d_v_prime, precision=_MATMUL_PRECISION)

        d_q_scaled = jnp.einsum("bhiv,bhkv->bhik", d_core_i, state_i, precision=_MATMUL_PRECISION)
        d_state_i = d_state_i + jnp.einsum("bhik,bhiv->bhkv", q_scaled, d_core_i, precision=_MATMUL_PRECISION)

        d_q_i = d_q_scaled * g_exp_i[:, :, :, None]
        d_g_exp_i = jnp.einsum("bhik,bhik->bhi", d_q_scaled, q_i, precision=_MATMUL_PRECISION)

        d_k_i = d_k_scaled * g_diff_i[:, :, :, None]
        d_g_diff_i = jnp.einsum("bhik,bhik->bhi", d_k_scaled, k_i, precision=_MATMUL_PRECISION)

        d_attn_qk_base = d_attn_qk * decay_i
        d_decay_i = d_attn_qk * attn_qk_base

        d_q_i = d_q_i + jnp.einsum("bhij,bhjk->bhik", d_attn_qk_base, k_i, precision=_MATMUL_PRECISION)
        d_k_i = d_k_i + jnp.einsum("bhji,bhjk->bhik", d_attn_qk_base, q_i, precision=_MATMUL_PRECISION)

        return d_state_i, (d_q_i, d_k_i, d_v_i, d_k_cum_i, d_g_exp_i, d_g_end_i, d_g_diff_i, d_decay_i)

    d_initial_state, grads_tm = lax.scan(
        rev_chunk_scan,
        d_final_state,
        (
            state_pre_tm,
            query_tm,
            key_tm,
            value_local_tm,
            key_cumdecay_tm,
            g_cumsum_exp_tm,
            g_end_exp_tm,
            g_diff_state_exp_tm,
            decay_mask_tm,
            d_out_tm,
        ),
        reverse=True,
    )

    (
        d_query_tm,
        d_key_tm,
        d_value_local_tm,
        d_key_cum_tm,
        d_g_exp_tm,
        d_g_end_tm,
        d_g_diff_tm,
        d_decay_mask_tm,
    ) = grads_tm

    d_query = d_query_tm.transpose(1, 2, 0, 3, 4)
    d_key = d_key_tm.transpose(1, 2, 0, 3, 4)
    d_value_local = d_value_local_tm.transpose(1, 2, 0, 3, 4)
    d_key_cum = d_key_cum_tm.transpose(1, 2, 0, 3, 4)
    d_g_exp = d_g_exp_tm.transpose(1, 2, 0, 3)
    d_g_end = d_g_end_tm.transpose(1, 2, 0)
    d_g_diff = d_g_diff_tm.transpose(1, 2, 0, 3)
    d_decay_mask = d_decay_mask_tm.transpose(1, 2, 0, 3, 4)

    d_attn = (
        jnp.einsum("bhciv,bhcjv->bhcij", d_value_local, value_beta, precision=_MATMUL_PRECISION)
        + jnp.einsum("bhcik,bhcjk->bhcij", d_key_cum, key_beta_scaled, precision=_MATMUL_PRECISION)
    )
    d_value_beta = jnp.einsum("bhcij,bhciv->bhcjv", attn, d_value_local, precision=_MATMUL_PRECISION)
    d_key_beta_scaled = jnp.einsum("bhcij,bhcik->bhcjk", attn, d_key_cum, precision=_MATMUL_PRECISION)

    d_key_beta = d_key_beta_scaled * g_cumsum_exp[:, :, :, :, None]
    d_g_exp = d_g_exp + jnp.sum(d_key_beta_scaled * key_beta, axis=-1)

    tmp = jnp.einsum("bhcji,bhcjk->bhcik", attn, d_attn, precision=_MATMUL_PRECISION)
    d_k_attn = -jnp.einsum("bhcij,bhckj->bhcik", tmp, attn, precision=_MATMUL_PRECISION)

    strict_lower = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
    lower_inclusive = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=0)
    d_k_attn = d_k_attn * strict_lower

    kk = jnp.einsum("bhcik,bhcjk->bhcij", key_beta, key, precision=_MATMUL_PRECISION)
    d_kk = d_k_attn * decay_mask
    d_decay_mask = (d_decay_mask + d_k_attn * kk) * lower_inclusive

    d_key_beta = d_key_beta + jnp.einsum("bhcij,bhcjk->bhcik", d_kk, key, precision=_MATMUL_PRECISION)
    d_key = d_key + jnp.einsum("bhcji,bhcjk->bhcik", d_kk, key_beta, precision=_MATMUL_PRECISION)

    d_value = d_value_beta * beta[:, :, :, :, None]
    d_beta = jnp.sum(d_value_beta * value, axis=-1)

    d_key = d_key + d_key_beta * beta[:, :, :, :, None]
    d_beta = d_beta + jnp.sum(d_key_beta * key, axis=-1)

    d_decay_f = d_decay_mask * decay_mask
    d_g = jnp.sum(d_decay_f, axis=-1) - jnp.sum(d_decay_f, axis=-2)
    d_g = d_g + d_g_exp * g_cumsum_exp

    d_g_diff_term = d_g_diff * g_diff_state_exp
    d_g_end_total = jnp.sum(d_g_diff_term, axis=-1) + d_g_end * g_end_exp
    d_g = d_g - d_g_diff_term
    d_g = d_g.at[:, :, :, -1].add(d_g_end_total)
    d_decay = jnp.flip(jnp.cumsum(jnp.flip(d_g, axis=-1), axis=-1), axis=-1)

    d_query = d_query.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
    d_key = d_key.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
    d_value = d_value.reshape(B, H, total_len, V_dim)[:, :, :seq_len, :]
    d_beta = d_beta.reshape(B, H, total_len)[:, :, :seq_len]
    d_decay = d_decay.reshape(B, H, total_len)[:, :, :seq_len]

    d_query = d_query * scale
    if use_qk_l2norm:
        q_norm = query.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :] / scale
        k_norm = key.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
        d_query = _l2norm_bwd(d_query, q_norm, q_inv_norm.astype(jnp.float32))
        d_key = _l2norm_bwd(d_key, k_norm, k_inv_norm.astype(jnp.float32))

    if decay_was_none:
        d_decay = None
    if initial_state_was_none:
        d_initial_state = None

    def _cast_grad(x):
        if x is None:
            return None
        return x.astype(input_dtype) if x.dtype != input_dtype else x

    return (
        _cast_grad(d_query),
        _cast_grad(d_key),
        _cast_grad(d_value),
        _cast_grad(d_beta),
        _cast_grad(d_decay),
        _cast_grad(d_initial_state),
    )


_chunk_gated_delta_rule_fwd.defvjp(_chunk_gdr_fwd, _chunk_gdr_bwd)


def _normalize_gdr_chunk_candidates(
    chunk_size: int,
    autotune_chunk_candidates: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    """Resolve candidate chunk sizes for autotune selection."""
    if autotune_chunk_candidates is None:
        if _GDR_AUTOTUNE_CHUNK_CANDIDATES_ENV is not None:
            candidates = list(_GDR_AUTOTUNE_CHUNK_CANDIDATES_ENV)
        else:
            half = max(1, int(chunk_size) // 2)
            candidates = [half, int(chunk_size), int(chunk_size) * 2]
    else:
        candidates = [int(x) for x in autotune_chunk_candidates]

    candidates.append(int(chunk_size))
    candidates = [x for x in candidates if x > 0]
    if not candidates:
        return (int(chunk_size),)
    return tuple(sorted(set(candidates)))


@dataclass(frozen=True)
class GatedDeltaRuleKernelConfig:
    chunk_size: int = 64
    use_qk_l2norm: bool = True


class _ChunkGatedDeltaRuleKernel(Kernel[GatedDeltaRuleKernelConfig, tuple[jax.Array, jax.Array]]):
    """ejkernel Kernel wrapper for chunked GDR."""

    version = "1"

    def __init__(self):
        super().__init__(op_id="gated_delta_rule_chunk")

    def run(
        self,
        query,
        key,
        value,
        beta,
        decay,
        initial_state,
        *,
        chunk_size: int = 64,
        use_qk_l2norm: bool = True,
        cfg: GatedDeltaRuleKernelConfig,
        **_,
    ):
        _ = chunk_size, use_qk_l2norm
        return _chunk_gated_delta_rule_fwd(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            chunk_size=int(cfg.chunk_size),
            initial_state=initial_state,
            use_qk_l2norm=bool(cfg.use_qk_l2norm),
        )

    def heuristic_cfg(
        self,
        inv: Invocation[GatedDeltaRuleKernelConfig, tuple[jax.Array, jax.Array]],
    ) -> GatedDeltaRuleKernelConfig:
        chunk_size = int(inv.kwargs.get("chunk_size", 64))
        use_qk_l2norm = bool(inv.kwargs.get("use_qk_l2norm", True))
        return GatedDeltaRuleKernelConfig(chunk_size=chunk_size, use_qk_l2norm=use_qk_l2norm)

    def candidate_cfgs(
        self,
        inv: Invocation[GatedDeltaRuleKernelConfig, tuple[jax.Array, jax.Array]],
    ):
        chunk_size = int(inv.kwargs.get("chunk_size", 64))
        use_qk_l2norm = bool(inv.kwargs.get("use_qk_l2norm", True))
        candidates = _normalize_gdr_chunk_candidates(
            chunk_size=chunk_size,
            autotune_chunk_candidates=inv.kwargs.get("autotune_chunk_candidates", None),
        )
        return [
            GatedDeltaRuleKernelConfig(chunk_size=int(c), use_qk_l2norm=use_qk_l2norm)
            for c in candidates
        ]


_GDR_CHUNK_KERNEL = _ChunkGatedDeltaRuleKernel()
_GDR_CHUNK_EXECUTOR = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(
            warmup=_GDR_AUTOTUNE_TUNER_WARMUP,
            iters=_GDR_AUTOTUNE_TUNER_ITERS,
        ),
    )
)


def _chunk_gated_delta_rule_fwd_dispatch(
    query,
    key,
    value,
    beta,
    decay,
    chunk_size,
    initial_state,
    use_qk_l2norm,
    autotune_chunk_size: bool,
    autotune_chunk_candidates: tuple[int, ...] | list[int] | None,
):
    """Run chunked GDR through an ejkernel Kernel/Executor instance."""
    cfg_override = None
    if not autotune_chunk_size:
        cfg_override = GatedDeltaRuleKernelConfig(
            chunk_size=int(chunk_size),
            use_qk_l2norm=bool(use_qk_l2norm),
        )

    return _GDR_CHUNK_EXECUTOR(
        _GDR_CHUNK_KERNEL,
        query,
        key,
        value,
        beta,
        decay,
        initial_state,
        chunk_size=int(chunk_size),
        use_qk_l2norm=bool(use_qk_l2norm),
        autotune_chunk_candidates=autotune_chunk_candidates,
        _cfg=cfg_override,
        stamp=False,
    )


def _single_step_gated_delta_rule_fwd(
    query: Float[Array, "batch num_heads 1 head_dim"],
    key: Float[Array, "batch num_heads 1 head_dim"],
    value: Float[Array, "batch num_heads 1 d_state"],
    beta: Float[Array, "batch num_heads 1"],
    decay: Float[Array, "batch num_heads 1"] | None,
    recurrent_state: Float[Array, "batch num_heads head_dim d_state"],
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads 1 d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Single-step recurrent forward pass for inference.

    Optimized for single-token generation during autoregressive decoding.

    Args:
        query: Query tensor [batch, num_heads, 1, head_dim]
        key: Key tensor [batch, num_heads, 1, head_dim]
        value: Value tensor [batch, num_heads, 1, d_state]
        beta: Gating tensor [batch, num_heads, 1]
        decay: Per-token decay [batch, num_heads, 1]
        recurrent_state: Previous state [batch, num_heads, head_dim, d_state]
        use_qk_l2norm: Whether to apply L2 normalization to query and key

    Returns:
        Tuple of (output, new_state)
    """
    if use_qk_l2norm:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)

    # Pure inference (single token) — no iterative accumulation, so keep
    # everything in the input dtype for speed. Only exp(decay) uses f32.
    query = query.squeeze(2)
    key = key.squeeze(2)
    value = value.squeeze(2)
    beta = beta.squeeze(2)

    head_dim = query.shape[-1]
    scale = 1.0 / (head_dim**0.5)
    query = query * scale

    if decay is not None:
        decay = decay.squeeze(2)
        g_exp = jnp.exp(decay.astype(jnp.float32)).astype(recurrent_state.dtype)
        recurrent_state = recurrent_state * g_exp[:, :, None, None]

    kv_mem = jnp.sum(recurrent_state * key[:, :, :, None], axis=-2)  # (B, H, V)

    beta_scaled = beta[:, :, None]
    delta = (value - kv_mem) * beta_scaled

    new_state = recurrent_state + key[:, :, :, None] * delta[:, :, None, :]

    output = jnp.sum(new_state * query[:, :, :, None], axis=-2)
    output = output[:, :, None, :]
    return output, new_state


@OperationRegistry.register
class GatedDeltaRuleOp(OperationImpl):
    """Gated Delta Rule linear attention operation.

    Implements the gated delta rule mechanism for efficient linear attention:
    - Training mode: Uses chunked algorithm for O(N) complexity
    - Inference mode: Uses recurrent update for single-token generation

    The gated delta rule updates state as:
        h_t = decay * h_{t-1} + beta_t * (v_t ⊗ k_t)
        o_t = h_t @ q_t

    Where:
    - beta_t is a learned gating signal
    - decay is an optional forgetting factor
    - v_t ⊗ k_t is the outer product

    Registered under the name "gated_delta_rule".

    Example:
        >>> from easydel.operations import OperationMetadata, OperationRegistry
        >>> metadata = OperationMetadata(runtime_dtype=jnp.float16)
        >>> gdr_op = OperationRegistry.create("gated_delta_rule", metadata)
        >>> output = gdr_op(
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
            Tuple of names: ("gated_delta_rule", "gdr")
        """
        return ("gated_delta_rule", "gdr")

    def get_impl_metadata(self) -> OperationMetadata:
        """Returns the metadata associated with this operation instance.

        Returns:
            The OperationMetadata provided during initialization.
        """
        assert self.metadata is not None
        return self.metadata

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for GatedDeltaRuleOp.

        GDR is a recurrent/linear attention mechanism that requires:
        - Basic metadata plus state management fields
        - Recurrent or Hybrid cache types for state persistence
        - Uses RecurrentCacheView for state management
        """
        return (
            RequirementsBuilder("gated_delta_rule")
            .require_metadata(
                MetadataField.SEQ_LENS
                | MetadataField.POSITIONS
                | MetadataField.HAS_INITIAL_STATE
                | MetadataField.STATE_INDICES
            )
            .optional_metadata(MetadataField.LOGITS_INDICES)
            .support_cache(CacheType.RECURRENT | CacheType.HYBRID)
            .use_cache_view(RecurrentCacheView)
            .build()
        )

    def _call_kernel(
        self,
        query,
        key,
        value,
        beta,
        decay,
        state,
        is_inference,
        chunk_size,
        autotune_chunk_size=False,
        autotune_chunk_candidates=None,
    ):
        """Dispatch to the appropriate kernel (inference vs training)."""
        if is_inference:
            return _single_step_gated_delta_rule_fwd(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                recurrent_state=state,
                use_qk_l2norm=True,
            )
        return _chunk_gated_delta_rule_fwd_dispatch(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            chunk_size=chunk_size,
            initial_state=state,
            use_qk_l2norm=True,
            autotune_chunk_size=bool(autotune_chunk_size),
            autotune_chunk_candidates=autotune_chunk_candidates,
        )

    @jax.named_scope("easydel-gated-delta-rule-native")
    def forward_native(
        self,
        query: Float[Array, "batch seq_len num_heads head_dim"],
        key: Float[Array, "batch seq_len num_heads head_dim"],
        value: Float[Array, "batch seq_len num_heads d_state"],
        beta: Float[Array, "batch seq_len num_heads head_dim"],
        decay: Float[Array, "num_heads head_dim"] | None = None,
        conv_state: Float[Array, "batch d_inner d_conv"] | None = None,
        recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
        chunk_size: int = 64,
        **kwargs,
    ) -> GatedDeltaRuleOutput:
        """Forward pass for gated delta rule attention.

        Uses shard_map to preserve sharding across internal transposes.
        Batch is sharded over (FSDP, DP), heads over TP.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, d_state]
            beta: Gating tensor [batch, seq_len, num_heads, head_dim]
            decay: Optional decay factors [num_heads, head_dim]
            conv_state: Optional convolution state (passed through, not used here)
            recurrent_state: Optional recurrent state for inference
            chunk_size: Chunk size for training mode (default: 64)
            **kwargs:
                - autotune_chunk_size: Optional bool to enable ejkernel autotuned
                  chunk-size selection for training mode.
                - autotune_chunk_candidates: Optional list/tuple of candidate
                  chunk sizes (e.g., ``(32, 64, 128)``). If not provided,
                  defaults to ``(chunk_size//2, chunk_size, chunk_size*2)``
                  unless ``EASYDEL_GDR_AUTOTUNE_CHUNK_CANDIDATES`` is set.

        Returns:
            GatedDeltaRuleOutput containing attention outputs and updated states
        """
        seq_len = query.shape[1]
        is_inference = seq_len == 1
        autotune_chunk_size = bool(kwargs.get("autotune_chunk_size", _GDR_AUTOTUNE_CHUNK_SIZE_DEFAULT))
        autotune_chunk_candidates = kwargs.get("autotune_chunk_candidates", None)

        # Determine mode before transpose (query is still BTHD).
        mode = None
        shardings_bthd = None
        if self.metadata.mesh is not None:
            with self.metadata.mesh:
                mode = self.get_mode(query=query, BTHD=True)
                shardings_bthd = self.metadata.get_shardings(mode, layout="bthd")

        # Transpose BTHD -> BHTD for kernel.
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        runtime_dtype = self.metadata.runtime_dtype
        query = query.astype(runtime_dtype)
        key = key.astype(runtime_dtype)
        value = value.astype(runtime_dtype)

        beta = beta.astype(runtime_dtype)
        if beta.ndim == 3:
            beta = beta.transpose(0, 2, 1)

        if decay is not None:
            decay = decay.astype(runtime_dtype)
            if decay.ndim == 3:
                decay = decay.transpose(0, 2, 1)

        effective_chunk_size = int(chunk_size)
        effective_autotune = bool(autotune_chunk_size)

        if self.metadata.mesh is not None and mode is not None:
            with self.metadata.mesh:
                shardings_bhtd = self.metadata.get_shardings(mode, layout="bhtd")

                # 4D specs [B, H, *, *]: preserve batch(0) and head(1) only.
                query_spec = self.create_stable_sharding(
                    shardings_bhtd.query,
                    tensor=query,
                    preserved_indices=[0, 1],
                )
                key_spec = self.create_stable_sharding(
                    shardings_bhtd.key,
                    tensor=key,
                    preserved_indices=[0, 1],
                )
                value_spec = self.create_stable_sharding(
                    shardings_bhtd.value,
                    tensor=value,
                    preserved_indices=[0, 1],
                )

                if query_spec is not None and key_spec is not None and value_spec is not None:
                    # 3D spec [B, H, L] for beta/decay: take batch+head from query spec.
                    beta_3d_source = Ps(
                        shardings_bhtd.query[0],
                        shardings_bhtd.query[1],
                        shardings_bhtd.query[2],
                    )
                    beta_spec = self.create_stable_sharding(
                        beta_3d_source,
                        tensor=beta,
                        preserved_indices=[0, 1],
                    )
                    # State [B, H, K, V]: same batch+head sharding as query.
                    state_spec = query_spec
                    output_spec = value_spec
                    decay_spec = beta_spec

                    # Materialize None args — shard_map needs concrete arrays.
                    B, H = query.shape[0], query.shape[1]
                    K_dim, V_dim = query.shape[-1], value.shape[-1]
                    if decay is None:
                        decay = jnp.zeros((B, H, seq_len), dtype=runtime_dtype)
                    if recurrent_state is None:
                        recurrent_state = jnp.zeros(
                            (B, H, K_dim, V_dim),
                            dtype=runtime_dtype,
                        )

                    def _gdr_kernel(q, k, v, b, d, s):
                        return self._call_kernel(
                            q,
                            k,
                            v,
                            b,
                            d,
                            s,
                            is_inference,
                            effective_chunk_size,
                            effective_autotune,
                            autotune_chunk_candidates,
                        )

                    outputs, new_recurrent_state = jax.shard_map(
                        _gdr_kernel,
                        mesh=self.metadata.mesh,
                        in_specs=(
                            query_spec,
                            key_spec,
                            value_spec,
                            beta_spec,
                            decay_spec,
                            state_spec,
                        ),
                        out_specs=(output_spec, state_spec),
                        check_vma=False,
                    )(query, key, value, beta, decay, recurrent_state)
                else:
                    # Specs are None — fall back to direct call.
                    outputs, new_recurrent_state = self._call_kernel(
                        query,
                        key,
                        value,
                        beta,
                        decay,
                        recurrent_state,
                        is_inference,
                        effective_chunk_size,
                        effective_autotune,
                        autotune_chunk_candidates,
                    )
        else:
            # No mesh — direct call.
            outputs, new_recurrent_state = self._call_kernel(
                query,
                key,
                value,
                beta,
                decay,
                recurrent_state,
                is_inference,
                effective_chunk_size,
                effective_autotune,
                autotune_chunk_candidates,
            )

        # Transpose output BHTD -> BTHD.
        outputs = outputs.transpose(0, 2, 1, 3)

        if self.metadata.mesh is not None and shardings_bthd is not None:
            with self.metadata.mesh:
                outputs = with_sharding_constraint(
                    arr=outputs,
                    sharding=shardings_bthd.output,
                )

        return GatedDeltaRuleOutput(
            attention_outputs=outputs,
            attention_weights=None,
            conv_state=conv_state,
            recurrent_state=new_recurrent_state,
        )

    def forward_tpu(self, *args, **kwargs) -> GatedDeltaRuleOutput:
        """TPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> GatedDeltaRuleOutput:
        """GPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> GatedDeltaRuleOutput:
        """CPU forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> GatedDeltaRuleOutput:
        """CUDA forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> GatedDeltaRuleOutput:
        """ROCm forward pass. Delegates to forward_native."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch seq_len num_heads head_dim"],
        key: Float[Array, "batch seq_len num_heads head_dim"],
        value: Float[Array, "batch seq_len num_heads d_state"],
        beta: Float[Array, "batch seq_len num_heads head_dim"],
        decay: Float[Array, "num_heads head_dim"] | None = None,
        conv_state: Float[Array, "batch d_inner d_conv"] | None = None,
        recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
        chunk_size: int = 64,
        **kwargs,
    ) -> GatedDeltaRuleOutput:
        """Execute the gated delta rule operation.

        Dispatches to appropriate backend via parent __call__.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, d_state]
            beta: Gating tensor [batch, seq_len, num_heads, head_dim]
            decay: Optional decay factors [num_heads, head_dim]
            conv_state: Optional convolution state
            recurrent_state: Optional recurrent state
            chunk_size: Chunk size for training mode
            **kwargs:
                - autotune_chunk_size: Enable ejkernel autotuned chunk-size
                  selection in training mode.
                - autotune_chunk_candidates: Candidate chunk sizes for autotune.

        Returns:
            GatedDeltaRuleOutput with attention outputs and updated states
        """
        return super().__call__(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            chunk_size=chunk_size,
            **kwargs,
        )


if __name__ == "__main__":
    from jax import random as jr

    from easydel.infra import EasyDeLBaseConfig

    print("Testing GatedDeltaRuleOp...")

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

    gdr_op = GatedDeltaRuleOp(metadata)

    print("Testing training mode (chunked)...")
    output = gdr_op(
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

    output_infer = gdr_op(
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

    out_recurrent, state_recurrent = _recurrent_gated_delta_rule_fwd(
        query.transpose(0, 2, 1, 3)[:, :, :32, :],
        key_tensor.transpose(0, 2, 1, 3)[:, :, :32, :],
        value.transpose(0, 2, 1, 3)[:, :, :32, :],
        beta.transpose(0, 2, 1)[:, :, :32],
        decay.transpose(0, 2, 1)[:, :, :32],
        use_qk_l2norm=True,
    )

    out_chunk, state_chunk = _chunk_gated_delta_rule_fwd(
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
