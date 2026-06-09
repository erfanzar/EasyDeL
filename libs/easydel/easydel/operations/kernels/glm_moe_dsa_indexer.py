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

"""Dynamic Sparse Attention (DSA) top-k indexer for GLM-MoE-DSA models.

This module provides :class:`GlmMoeDsaIndexerOp`, an :class:`OperationImpl`
that selects per-query top-k key indices used by the DSA branch of the
GLM-MoE-DSA architecture. Optional RoPE is applied (interleaved or split),
keys may be cached across calls, and an attention mask may be supplied to
suppress invalid positions before the top-k reduction.

Exports:
    GlmMoeDsaIndexerOp: The registered operation, named ``"glm_moe_dsa_indexer"``
        / ``"dsa_indexer"``.
    GlmMoeDsaIndexerOutput: Pytree container holding the selected indices
        and the (optionally updated) key cache.
"""

from __future__ import annotations

import jax
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from .._operation_impl import OperationImpl, OperationOutput, OperationRegistry
from ..requirements import CacheType, ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder


@auto_pytree
class GlmMoeDsaIndexerOutput(OperationOutput):
    """Output container for the GLM-MoE-DSA indexer operation.

    Attributes:
        topk_indices: Selected per-query token indices of shape
            ``(batch, seq, topk)``, or ``None`` before the operation runs.
        cached_keys: Optional cached key tensor of shape
            ``(batch, total_seq, head_dim)``. Populated only when
            ``use_cache=True`` was passed to the operator; ``None`` otherwise.
    """

    topk_indices: Int[Array, "batch seq topk"] | None = None
    cached_keys: Float[Array, "batch total_seq head_dim"] | None = None


@OperationRegistry.register
class GlmMoeDsaIndexerOp(OperationImpl):
    """Compute DSA top-k token indices with an optional indexer-local key cache.

    The indexer multiplies queries against (optionally cached) keys, weights
    the per-head scores by ``head_weights``, applies an optional attention
    mask, and returns the indices of the highest-scoring keys. It is used by
    the GLM-MoE-DSA architecture to drive the dynamic-sparse-attention
    routing decision.

    Registered names: ``"glm_moe_dsa_indexer"`` and ``"dsa_indexer"``.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the registered names of this operation.

        Returns:
            tuple[str, ...]: ``("glm_moe_dsa_indexer", "dsa_indexer")``.
        """
        return ("glm_moe_dsa_indexer", "dsa_indexer")

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Declare metadata and cache requirements for the indexer.

        The indexer needs ``POSITIONS`` for RoPE and is compatible with the
        transformer or hybrid cache types, but does not itself require a KV
        cache (``requires_cache(False)``).

        Args:
            mode: Execution mode (unused for this operation).

        Returns:
            OperationRequirements: The declared requirements.
        """
        del mode
        return (
            RequirementsBuilder("glm_moe_dsa_indexer")
            .require_metadata(MetadataField.POSITIONS)
            .support_cache(CacheType.TRANSFORMER | CacheType.HYBRID)
            .requires_cache(False)
            .build()
        )

    @staticmethod
    def _apply_rope_interleaved(
        x: Float[Array, "batch seq ... rope_dim"],
        cos: Float[Array, "batch seq rope_dim_half"],
        sin: Float[Array, "batch seq rope_dim_half"],
    ) -> Float[Array, "batch seq ... rope_dim"]:
        """Apply RoPE in interleaved layout (even/odd lane pairs).

        Args:
            x: Tensor whose last dimension is RoPE-rotated, of shape
                ``(batch, seq, ..., rope_dim)``.
            cos: Cosine table of shape ``(batch, seq, rope_dim // 2)``.
            sin: Sine table with the same shape as ``cos``.

        Returns:
            Float[Array]: ``x`` after RoPE rotation, with the same shape.
        """
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos[..., None, :] - x2 * sin[..., None, :]
        o2 = x2 * cos[..., None, :] + x1 * sin[..., None, :]
        return jnp.stack((o1, o2), axis=-1).reshape(x.shape)

    @staticmethod
    def _apply_rope_split(
        x: Float[Array, "batch seq ... rope_dim"],
        cos: Float[Array, "batch seq rope_dim_half"],
        sin: Float[Array, "batch seq rope_dim_half"],
    ) -> Float[Array, "batch seq ... rope_dim"]:
        """Apply RoPE in split (first-half / second-half) layout.

        Args:
            x: Tensor whose last dimension is RoPE-rotated, of shape
                ``(batch, seq, ..., rope_dim)``.
            cos: Cosine table of shape ``(batch, seq, rope_dim // 2)``.
            sin: Sine table with the same shape as ``cos``.

        Returns:
            Float[Array]: ``x`` after RoPE rotation, with the same shape.
        """
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate(
            [x1 * cos[..., None, :] - x2 * sin[..., None, :], x2 * cos[..., None, :] + x1 * sin[..., None, :]],
            axis=-1,
        )

    @jax.named_scope("easydel-glm-moe-dsa-indexer-native")
    def forward_native(
        self,
        query_states: Float[Array, "batch seq n_heads head_dim"],
        key_states: Float[Array, "batch seq head_dim"],
        head_weights: Float[Array, "batch seq n_heads"],
        position_ids: Int[Array, "batch seq"],
        qk_rope_head_dim: int,
        index_topk: int,
        softmax_scale: float,
        frequencies: Float[Array, "max_seq rope_dim_x2"] | None = None,
        attention_mask: Bool[Array, "batch seq kv"] | Float[Array, "batch seq kv"] | None = None,
        cached_keys: Float[Array, "batch cached_seq head_dim"] | None = None,
        use_cache: bool = False,
        reset_cache: bool = False,
        indexer_rope_interleave: bool = False,
        **ignore,
    ) -> GlmMoeDsaIndexerOutput:
        """Compute DSA top-k token indices for dynamic sparse attention.

        Applies optional RoPE to query and key states, optionally
        concatenates with cached keys, computes per-head attention scores
        weighted by ``head_weights``, applies masking, and returns the
        top-k token indices per query position.

        Args:
            query_states: Query tensor [batch, seq, n_heads, head_dim].
            key_states: Key tensor [batch, seq, head_dim] (single-head).
            head_weights: Per-head weighting [batch, seq, n_heads].
            position_ids: Position indices [batch, seq] for RoPE.
            qk_rope_head_dim: Number of head dimensions to apply RoPE to.
            index_topk: Number of top-k indices to return per query token.
            softmax_scale: Scaling factor for attention scores.
            frequencies: Precomputed RoPE frequencies. None skips RoPE.
            attention_mask: Optional boolean or float mask [batch, seq, kv].
            cached_keys: Previously cached keys for incremental indexing.
            use_cache: Whether to update and return cached keys.
            reset_cache: Whether to discard existing cached keys.
            indexer_rope_interleave: Use interleaved RoPE layout if True.
            **ignore: Additional ignored keyword arguments.

        Returns:
            GlmMoeDsaIndexerOutput with ``topk_indices`` and optionally
            updated ``cached_keys``.
        """
        del ignore

        query_states_f32 = query_states.astype(jnp.float32)
        key_states_f32 = key_states.astype(jnp.float32)

        if frequencies is not None:
            freq_array = frequencies.value if hasattr(frequencies, "value") else frequencies
            freqs = freq_array[position_ids]
            cos, sin = jnp.split(freqs, 2, axis=-1)

            rope_dim = min(
                int(qk_rope_head_dim),
                int(query_states_f32.shape[-1]),
                int(key_states_f32.shape[-1]),
                int(cos.shape[-1] * 2),
            )
            rope_dim = rope_dim - (rope_dim % 2)
            if rope_dim > 0:
                cos = cos[..., : rope_dim // 2]
                sin = sin[..., : rope_dim // 2]

                q_pe = query_states_f32[..., :rope_dim]
                q_nope = query_states_f32[..., rope_dim:]
                k_pe = key_states_f32[..., :rope_dim]
                k_nope = key_states_f32[..., rope_dim:]

                if indexer_rope_interleave:
                    q_pe = self._apply_rope_interleaved(q_pe, cos, sin)
                    k_pe = self._apply_rope_interleaved(k_pe[:, :, None, :], cos, sin).squeeze(2)
                else:
                    q_pe = self._apply_rope_split(q_pe, cos, sin)
                    k_pe = self._apply_rope_split(k_pe[:, :, None, :], cos, sin).squeeze(2)

                query_states_f32 = jnp.concatenate([q_pe, q_nope], axis=-1)
                key_states_f32 = jnp.concatenate([k_pe, k_nope], axis=-1)

        cached_keys_local = None if reset_cache else cached_keys

        if use_cache:
            if cached_keys_local is not None:
                all_keys = jnp.concatenate([cached_keys_local.astype(jnp.float32), key_states_f32], axis=1)
            else:
                all_keys = key_states_f32
            new_cached_keys = all_keys
        else:
            all_keys = key_states_f32
            new_cached_keys = cached_keys_local

        scores = jnp.einsum("bshd,btd->bsht", query_states_f32, all_keys) * softmax_scale
        index_scores = jnp.einsum("bsht,bsh->bst", scores, head_weights.astype(jnp.float32))

        total_len = int(index_scores.shape[-1])

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, :]

            q_len = int(index_scores.shape[-2])
            if attention_mask.shape[-2] > q_len:
                attention_mask = attention_mask[:, :q_len, :]
            elif attention_mask.shape[-2] < q_len:
                q_pad_len = q_len - int(attention_mask.shape[-2])
                if attention_mask.dtype == jnp.bool_:
                    attention_mask = jnp.pad(attention_mask, ((0, 0), (0, q_pad_len), (0, 0)), constant_values=True)
                else:
                    attention_mask = jnp.pad(attention_mask, ((0, 0), (0, q_pad_len), (0, 0)), constant_values=0.0)

            if attention_mask.shape[-1] > total_len:
                attention_mask = attention_mask[..., :total_len]
            elif attention_mask.shape[-1] < total_len:
                pad_len = total_len - int(attention_mask.shape[-1])
                if attention_mask.dtype == jnp.bool_:
                    attention_mask = jnp.pad(attention_mask, ((0, 0), (0, 0), (0, pad_len)), constant_values=True)
                else:
                    attention_mask = jnp.pad(attention_mask, ((0, 0), (0, 0), (0, pad_len)), constant_values=0.0)

            if attention_mask.dtype == jnp.bool_:
                index_scores = jnp.where(attention_mask, index_scores, -jnp.inf)
            else:
                index_scores = index_scores + attention_mask.astype(index_scores.dtype)

        topk = min(max(int(index_topk), 1), total_len)
        topk_indices = jax.lax.top_k(index_scores, k=topk)[1]

        return GlmMoeDsaIndexerOutput(
            topk_indices=topk_indices,
            cached_keys=None if new_cached_keys is None else new_cached_keys.astype(key_states.dtype),
        )

    def forward_tpu(self, *args, **kwargs) -> GlmMoeDsaIndexerOutput:
        """TPU forward pass. Delegates to ``forward_native``."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> GlmMoeDsaIndexerOutput:
        """GPU forward pass. Delegates to ``forward_native``."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> GlmMoeDsaIndexerOutput:
        """CPU forward pass. Delegates to ``forward_native``."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> GlmMoeDsaIndexerOutput:
        """CUDA forward pass. Delegates to ``forward_native``."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> GlmMoeDsaIndexerOutput:
        """ROCm forward pass. Delegates to ``forward_native``."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query_states: Float[Array, "batch seq n_heads head_dim"],
        key_states: Float[Array, "batch seq head_dim"],
        head_weights: Float[Array, "batch seq n_heads"],
        position_ids: Int[Array, "batch seq"],
        qk_rope_head_dim: int,
        index_topk: int,
        softmax_scale: float,
        frequencies: Float[Array, "max_seq rope_dim_x2"] | None = None,
        attention_mask: Bool[Array, "batch seq kv"] | Float[Array, "batch seq kv"] | None = None,
        cached_keys: Float[Array, "batch cached_seq head_dim"] | None = None,
        use_cache: bool = False,
        reset_cache: bool = False,
        indexer_rope_interleave: bool = False,
        **kwargs,
    ) -> GlmMoeDsaIndexerOutput:
        """Execute the DSA indexer by dispatching to the appropriate backend.

        Args:
            query_states: Query tensor [batch, seq, n_heads, head_dim].
            key_states: Key tensor [batch, seq, head_dim].
            head_weights: Per-head weighting [batch, seq, n_heads].
            position_ids: Position indices [batch, seq].
            qk_rope_head_dim: RoPE dimension count.
            index_topk: Number of top-k indices to select.
            softmax_scale: Attention score scaling factor.
            frequencies: Optional RoPE frequencies.
            attention_mask: Optional attention mask.
            cached_keys: Optional previously cached keys.
            use_cache: Whether to maintain a key cache.
            reset_cache: Whether to discard existing cache.
            indexer_rope_interleave: Use interleaved RoPE layout.
            **kwargs: Additional keyword arguments passed to the backend.

        Returns:
            GlmMoeDsaIndexerOutput with top-k indices and cached keys.
        """
        return super().__call__(
            query_states=query_states,
            key_states=key_states,
            head_weights=head_weights,
            position_ids=position_ids,
            qk_rope_head_dim=qk_rope_head_dim,
            index_topk=index_topk,
            softmax_scale=softmax_scale,
            frequencies=frequencies,
            attention_mask=attention_mask,
            cached_keys=cached_keys,
            use_cache=use_cache,
            reset_cache=reset_cache,
            indexer_rope_interleave=indexer_rope_interleave,
            **kwargs,
        )


__all__ = ("GlmMoeDsaIndexerOp", "GlmMoeDsaIndexerOutput")
