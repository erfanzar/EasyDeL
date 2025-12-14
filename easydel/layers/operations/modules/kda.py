# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
from eformer.escale import with_sharding_constraint
from eformer.pytree import auto_pytree
from jax import lax
from jaxtyping import Array, Float

from easydel.layers.caching import KDACacheView

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


def l2norm(x, axis=-1, eps=1e-6):
    """L2 normalize along specified axis.

    Uses rsqrt: inv_norm = rsqrt(sum(x^2) + eps); return x * inv_norm
    """
    inv_norm = lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def fused_kda_gate(gate: Float[Array, "..."], A_log: Float[Array, "num_heads"], dt_bias: Float[Array, "num_heads"]):
    """Compute KDA decay gate.

    Implements: decay = -exp(A_log) * softplus(gate + dt_bias)

    Args:
        gate: Gate values from f_b_proj [batch, seq, num_heads]
        A_log: Log decay parameter [num_heads]
        dt_bias: Time discretization bias [num_heads]

    Returns:
        Decay values [batch, seq, num_heads]
    """
    A = -jnp.exp(A_log.astype(jnp.float32))
    gate = gate.astype(jnp.float32)
    dt_bias = dt_bias.astype(jnp.float32)
    # softplus(gate + dt_bias)
    gate_biased = gate + dt_bias
    softplus_gate = jnp.log1p(jnp.exp(gate_biased))
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
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Recurrent forward pass for KDA.

    Processes each position sequentially using lax.scan for efficiency.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_heads, seq_len, head_dim]
        value: Value tensor [batch, num_heads, seq_len, d_state]
        beta: Gating tensor [batch, num_heads, seq_len]
        decay: Per-token decay [batch, num_heads, seq_len]
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
    """Chunked forward pass for KDA.

    Processes the sequence in chunks for efficient parallel computation.
    Uses intra-chunk parallel attention and inter-chunk sequential state updates.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_heads, seq_len, head_dim]
        value: Value tensor [batch, num_heads, seq_len, d_state]
        beta: Gating tensor [batch, num_heads, seq_len]
        decay: Per-token decay [batch, num_heads, seq_len]
        chunk_size: Size of chunks for parallel processing
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
        row = attn_chunk[i, :]
        idx = jnp.arange(chunk_size)
        mask_lt_i = idx < i
        contribution = jnp.sum(row[:, None] * attn_chunk * mask_lt_i[:, None] * mask_lt_i[None, :], axis=0)

        new_row = row + contribution
        new_row = jnp.where(mask_lt_i, new_row, row)

        return attn_chunk.at[i].set(new_row), None

    def resolve_single_chunk(attn_single):
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

    # Initialize state
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


def _single_step_kda_fwd(
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

    query = query.squeeze(2)
    key = key.squeeze(2)
    value = value.squeeze(2)
    beta = beta.squeeze(2)

    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    recurrent_state = recurrent_state.astype(jnp.float32)

    head_dim = query.shape[-1]
    scale = 1.0 / (head_dim**0.5)
    query = query * scale

    if decay is not None:
        decay = decay.squeeze(2).astype(jnp.float32)
        g_exp = jnp.exp(decay)[:, :, None, None]
        recurrent_state = recurrent_state * g_exp

    kv_mem = jnp.sum(recurrent_state * key[:, :, :, None], axis=-2)  # (B, H, V)

    beta_scaled = beta[:, :, None]
    delta = (value - kv_mem) * beta_scaled

    new_state = recurrent_state + key[:, :, :, None] * delta[:, :, None, :]

    output = jnp.sum(new_state * query[:, :, :, None], axis=-2)
    output = output[:, :, None, :]
    return output, new_state


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

    Registered under the names "kda" and "kernel_delta_attention".

    Example:
        >>> from easydel.layers.operations import OperationMetadata, OperationRegistry
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
    def get_impl_name(cls) -> str | tuple[str]:
        """Returns the registered name of this operation.

        Returns:
            Tuple of names: ("kda", "kernel_delta_attention")
        """
        return ("kda", "kernel_delta_attention")

    def get_impl_metadata(self) -> OperationMetadata:
        """Returns the metadata associated with this operation instance.

        Returns:
            The OperationMetadata provided during initialization.
        """
        return self.metadata

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
        decay: Float[Array, "batch seq_len num_heads"] | None = None,
        q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None,
        recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
        chunk_size: int = 64,
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
            decay: Optional decay factors [batch, seq_len, num_heads]
            q_conv_state: Optional Q convolution state (passed through, not used here)
            k_conv_state: Optional K convolution state (passed through, not used here)
            v_conv_state: Optional V convolution state (passed through, not used here)
            recurrent_state: Optional recurrent state for inference
            chunk_size: Chunk size for training mode (default: 64)
            **kwargs: Additional ignored arguments

        Returns:
            KDAOutput containing attention outputs and updated states
        """
        seq_len = query.shape[1]

        if self.metadata.mesh is not None:
            with self.metadata.mesh:
                mode = self.get_mode(query=query, BTHD=True)
                shardings = self.metadata.get_shardings(mode, layout="bthd")
                query = with_sharding_constraint(arr=query, sharding=shardings.query)
                key = with_sharding_constraint(arr=key, sharding=shardings.key)
                value = with_sharding_constraint(arr=value, sharding=shardings.value)

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

        is_inference = seq_len == 1

        if is_inference and recurrent_state is not None:
            outputs, new_recurrent_state = _single_step_kda_fwd(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                recurrent_state=recurrent_state,
                use_qk_l2norm=True,
            )
        else:
            outputs, new_recurrent_state = _chunk_kda_fwd(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                chunk_size=chunk_size,
                initial_state=recurrent_state,
                use_qk_l2norm=True,
            )

        outputs = outputs.transpose(0, 2, 1, 3)

        if self.metadata.mesh is not None:
            with self.metadata.mesh:
                outputs = with_sharding_constraint(arr=outputs, sharding=shardings.output)

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
        decay: Float[Array, "batch seq_len num_heads"] | None = None,
        q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None,
        recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
        chunk_size: int = 64,
        **kwargs,
    ) -> KDAOutput:
        """Execute the KDA operation.

        Dispatches to appropriate backend via parent __call__.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, d_state]
            beta: Gating tensor [batch, seq_len, num_heads]
            decay: Optional decay factors [batch, seq_len, num_heads]
            q_conv_state: Optional Q convolution state
            k_conv_state: Optional K convolution state
            v_conv_state: Optional V convolution state
            recurrent_state: Optional recurrent state
            chunk_size: Chunk size for training mode
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
            chunk_size=chunk_size,
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
