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

import typing as tp

import jax
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int
from spectrax import sanitize_partition_spec_for_mesh_and_shape

from easydel.layers.rotary import apply_rope_interleaved, apply_rope_split_half

from .._operation_impl import OperationImpl, OperationOutput, OperationRegistry
from ..requirements import CacheType, ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder
from ._mask_info import topk_selection_mask


@auto_pytree
class GlmMoeDsaIndexerOutput(OperationOutput):
    """Output container for the GLM-MoE-DSA indexer operation.

    Attributes:
        topk_indices: Selected per-query token indices of shape
            ``(batch, seq, topk)``, or ``None`` before the operation runs.
            With explicit indexed/visibility metadata, unfilled slots are -1;
            legacy calls preserve raw top_k indices (including masked filler slots).
        topk_mask: Authoritative selection over the scored keys,
            ``(batch, seq, total_seq)``; excludes hidden/masked filler lanes even
            when legacy raw ``topk_indices`` retain them.
        cached_keys: Optional cached key tensor of shape
            ``(batch, total_seq, head_dim)``. Populated only when
            ``use_cache=True`` was passed to the operator; ``None`` otherwise.
    """

    topk_indices: Int[Array, "batch seq topk"] | None = None
    cached_keys: Float[Array, "batch total_seq head_dim"] | None = None
    topk_mask: Bool[Array, "batch seq total_seq"] | None = None


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
        *,
        precision: jax.lax.PrecisionLike = None,
        cache_positions: Array | None = None,
        key_valid_mask: Array | None = None,
        query_positions: Array | None = None,
        key_positions: Array | None = None,
        segment_ids: Array | None = None,
        key_segment_ids: Array | None = None,
        select_topk: tp.Callable[[Array, int, Array], tuple[Array, Array]] | None = None,
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
            precision: Matmul precision for query-key scoring and head weighting.
                None inherits JAX's ambient policy; fp32 operands alone do not
                imply full-fp32 multiplication on TPU.
            cache_positions: Optional physical write slots [batch, seq] in fixed-capacity
                cached_keys. Negative/out-of-range slots are ignored; requires use_cache
                and post-update key_valid_mask. Reset zeroes storage without changing shape.
            key_valid_mask: Optional post-update historical validity [batch, kv].
            query_positions: Logical causal positions [batch, seq], defaulting to
                cache_positions in indexed mode or the appended query offsets otherwise.
            key_positions: Logical key positions [batch, kv], defaulting to storage slots.
                Supply these with query_positions for sliding/reordered physical storage.
            segment_ids: Optional query document identities [batch, seq], negative for padding.
            key_segment_ids: Post-update historical document identities [batch, kv];
                required with segment_ids when caching. The owning cache retains metadata.
            select_topk: Backend row selector ``(scores, k, query) -> (values, indices)``
                with ``jax.lax.top_k`` semantics; ``None`` uses ``jax.lax.top_k``.
            **ignore: Additional ignored keyword arguments.

        Returns:
            GlmMoeDsaIndexerOutput with ``topk_indices`` and optionally
            updated ``cached_keys``.
        """
        del ignore
        explicit_visibility = any(
            value is not None
            for value in (cache_positions, key_valid_mask, query_positions, key_positions, segment_ids, key_segment_ids)
        )

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
                    q_pe = apply_rope_interleaved(q_pe, cos[..., None, :], sin[..., None, :])
                    k_pe = apply_rope_interleaved(k_pe, cos, sin)
                else:
                    cos_full = jnp.concatenate((cos, cos), axis=-1)
                    sin_full = jnp.concatenate((sin, sin), axis=-1)
                    q_pe = apply_rope_split_half(q_pe, cos_full, sin_full, unsqueeze_dim=-2)
                    k_pe = apply_rope_split_half(k_pe, cos_full, sin_full, unsqueeze_dim=None)

                query_states_f32 = jnp.concatenate([q_pe, q_nope], axis=-1)
                key_states_f32 = jnp.concatenate([k_pe, k_nope], axis=-1)

        cached_keys_local = None if reset_cache else cached_keys

        if cache_positions is not None:
            if not use_cache or cached_keys is None or key_valid_mask is None:
                raise ValueError("cache_positions requires use_cache, preallocated cached_keys and key_valid_mask")
            batch, q_len, dim = key_states_f32.shape
            if cached_keys.shape[0] != batch or cached_keys.shape[-1] != dim:
                raise ValueError("cached_keys batch/head dimensions must match key_states")
            if cache_positions.shape != (batch, q_len):
                raise ValueError("cache_positions must have shape [batch, seq]")
            if key_valid_mask.shape != cached_keys.shape[:2]:
                raise ValueError("key_valid_mask must describe the full post-update cache capacity")
            capacity = cached_keys.shape[1]
            slots = jnp.asarray(cache_positions, dtype=jnp.int32)
            # Negative slots are padding, not Python-style indexes from the end.
            slots = jnp.where((slots >= 0) & (slots < capacity), slots, capacity)
            storage = jnp.zeros_like(cached_keys) if reset_cache else cached_keys
            all_keys = jax.vmap(lambda old, pos, new: old.at[pos].set(new, mode="drop"))(
                storage.astype(jnp.float32), slots, key_states_f32
            )
            new_cached_keys = all_keys
            if query_positions is None:
                query_positions = cache_positions
        elif use_cache:
            if cached_keys_local is not None:
                all_keys = jnp.concatenate([cached_keys_local.astype(jnp.float32), key_states_f32], axis=1)
            else:
                all_keys = key_states_f32
            new_cached_keys = all_keys
        else:
            all_keys = key_states_f32
            new_cached_keys = cached_keys_local

        scores = jnp.einsum("bshd,btd->bsht", query_states_f32, all_keys, precision=precision) * softmax_scale
        # HF GlmMoeDsaIndexer / DeepseekV32Indexer clamp each head's score before weighting.
        scores = jax.nn.relu(scores)
        index_scores = jnp.einsum("bsht,bsh->bst", scores, head_weights.astype(jnp.float32), precision=precision)

        total_len = int(index_scores.shape[-1])
        caller_visible = None

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, :]

            q_len = int(index_scores.shape[-2])
            if attention_mask.shape[-2] == 1:
                attention_mask = jnp.broadcast_to(
                    attention_mask, (attention_mask.shape[0], q_len, attention_mask.shape[-1])
                )
            elif attention_mask.shape[-2] > q_len:
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
                caller_visible = attention_mask
                index_scores = jnp.where(attention_mask, index_scores, -jnp.inf)
            else:
                # A finite-min additive sentinel is just as ineligible as -inf.
                # Inspect the original mask dtype before converting to fp32 (a
                # bf16/fp16 sentinel is not fp32's minimum after conversion).
                caller_visible = jnp.isfinite(attention_mask) & (attention_mask > jnp.finfo(attention_mask.dtype).min)
                index_scores = index_scores + attention_mask.astype(index_scores.dtype)

        # Causality (HF masks future keys before top-k): the current queries are the last
        # ``q_len`` keys (no-cache prefill or cache append), so query ``i`` sees keys
        # ``<= total_len - q_len + i``. A non-causal caller mask must not let future keys
        # take top-k slots that the attention kernel would then hide.
        batch, q_len = index_scores.shape[:2]
        if query_positions is None:
            query_positions = jnp.broadcast_to(total_len - q_len + jnp.arange(q_len), (batch, q_len))
        if key_positions is None:
            key_positions = jnp.broadcast_to(jnp.arange(total_len), (batch, total_len))
        if query_positions.shape != (batch, q_len) or key_positions.shape != (batch, total_len):
            raise ValueError("query_positions/key_positions must match [batch, seq]/[batch, kv]")
        visible = (key_positions[:, None, :] <= query_positions[:, :, None]) & (query_positions[:, :, None] >= 0)
        if key_valid_mask is not None:
            if key_valid_mask.shape != (batch, total_len):
                raise ValueError("key_valid_mask must match [batch, kv]")
            visible = visible & key_valid_mask[:, None, :].astype(jnp.bool_)
        if segment_ids is not None:
            if key_segment_ids is None and not use_cache:
                key_segment_ids = segment_ids
            if key_segment_ids is None:
                raise ValueError("cached segment masking requires post-update key_segment_ids")
            if segment_ids.shape != (batch, q_len) or key_segment_ids.shape != (batch, total_len):
                raise ValueError("segment_ids/key_segment_ids must match [batch, seq]/[batch, kv]")
            visible = visible & (segment_ids[:, :, None] >= 0)
            visible = visible & (segment_ids[:, :, None] == key_segment_ids[:, None, :])
        elif key_segment_ids is not None:
            raise ValueError("key_segment_ids requires query segment_ids")
        eligible = visible & (index_scores != -jnp.inf)
        if caller_visible is not None:
            eligible = eligible & caller_visible
        index_scores = jnp.where(eligible, index_scores, -jnp.inf)

        topk = min(max(int(index_topk), 1), total_len)
        if select_topk is None:
            topk_values, topk_indices = jax.lax.top_k(index_scores, k=topk)
        else:
            topk_values, topk_indices = select_topk(index_scores, topk, query_states)
        # Even legacy calls need an authoritative mask: top_k fills all k slots
        # when fewer than k candidates are eligible, including fully masked rows.
        topk_mask = topk_selection_mask(index_scores, topk_values, topk_indices) & eligible
        if explicit_visibility:
            # The indexed/segment contract marks unfilled top-k slots explicitly, so
            # callers consuming indices as well as masks cannot resurrect hidden keys.
            # Keep the historical raw top_k behavior for legacy calls without metadata.
            selected_valid = jnp.take_along_axis(eligible, topk_indices, axis=-1)
            topk_indices = jnp.where(selected_valid, topk_indices, -1)

        return GlmMoeDsaIndexerOutput(
            topk_indices=checkpoint_name(topk_indices, "indexer_topk"),
            topk_mask=checkpoint_name(topk_mask, "indexer_topk"),
            cached_keys=(
                None
                if new_cached_keys is None
                else new_cached_keys.astype(cached_keys.dtype if cache_positions is not None else key_states.dtype)
            ),
        )

    def _select_topk_tpu(self, index_scores: Array, topk: int, query_states: Array) -> tuple[Array, Array]:
        """Row-parallel exact top-k on TPU, identical to ``jax.lax.top_k``.

        ``jax.lax.top_k`` lowers to a full sort of every score row on TPU; the
        registered ejkernel operation bisects for the ``k``-th key instead. A
        Pallas call is not SPMD-partitionable, so rows are split explicitly
        with the query's batch/sequence layout and the candidate axis kept
        whole; each shard selects over complete rows, as XLA's own top-k does.
        """
        from ejkernel.modules import topk as ejkernel_topk  # pyright: ignore[reportMissingTypeStubs]

        def select(scores: Array) -> tuple[Array, Array]:
            return ejkernel_topk(scores, topk)

        mesh = self.metadata.mesh
        if mesh is None:
            return select(index_scores)
        with mesh:
            query_spec = self.metadata.get_shardings(self.get_mode(query=query_states, BTHD=True), layout="bthd").query
        jax_mesh = getattr(mesh, "jax_mesh", mesh)
        # Policy specs may name axes absent from a reduced/stage-local mesh.
        # Filter each compound placement before checking row divisibility, keeping
        # surviving members (e.g. dp in (fsdp, dp)) and the candidate axis whole.
        row_spec = sanitize_partition_spec_for_mesh_and_shape(
            PartitionSpec(*query_spec[:2], None), mesh=mesh, shape=index_scores.shape
        )
        return jax.shard_map(
            select,
            mesh=jax_mesh,
            in_specs=(row_spec,),
            out_specs=(row_spec, row_spec),
            check_vma=False,
        )(index_scores)

    def forward_tpu(self, *args, **kwargs) -> GlmMoeDsaIndexerOutput:
        """TPU forward pass: ``forward_native`` with the row-parallel top-k."""
        return self.forward_native(*args, select_topk=self._select_topk_tpu, **kwargs)

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
        *,
        precision: jax.lax.PrecisionLike = None,
        cache_positions: Array | None = None,
        key_valid_mask: Array | None = None,
        query_positions: Array | None = None,
        key_positions: Array | None = None,
        segment_ids: Array | None = None,
        key_segment_ids: Array | None = None,
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
            precision: Matmul precision for both score contractions. None
                preserves the ambient JAX policy. Capture this argument in a
                JIT closure or mark it static when compiling the operation.
            cache_positions: Optional physical write slots [batch, seq] in preallocated
                cached_keys; negative/out-of-range slots are ignored. Requires use_cache
                and key_valid_mask. Omitted preserves legacy append/reset semantics.
            key_valid_mask: Post-update historical key validity [batch, kv].
            query_positions: Logical query positions [batch, seq] for causal selection.
            key_positions: Logical key positions [batch, kv], independent of physical slots.
            segment_ids: Query document identities [batch, seq], negative for padding.
            key_segment_ids: Post-update key identities [batch, kv]; required for
                cached segment masking. Metadata remains owned by the cache.
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
            precision=precision,
            cache_positions=cache_positions,
            key_valid_mask=key_valid_mask,
            query_positions=query_positions,
            key_positions=key_positions,
            segment_ids=segment_ids,
            key_segment_ids=key_segment_ids,
            **kwargs,
        )


__all__ = ("GlmMoeDsaIndexerOp", "GlmMoeDsaIndexerOutput")
