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

"""Ring Attention implementation for distributed and ultra-long sequence processing.

This module implements Ring Attention, a technique for computing attention over
extremely long sequences that exceed single-device memory capacity. Ring Attention
partitions sequences across devices and uses a ring communication pattern to
compute exact attention without approximation.

Key concepts:
- **Ring Topology**: Devices arranged in a ring, each holding a chunk of the sequence
- **Blockwise Processing**: Attention computed in blocks to minimize memory usage
- **Communication Pattern**: Each device passes its KV chunks around the ring
- **Exact Computation**: Produces the same result as standard attention

The implementation provides:
1. Native JAX scan-based version for all backends
2. TPU-optimized Pallas kernel for maximum performance
3. Support for sequences > 100K tokens
4. Memory usage O(N/P) where N=sequence length, P=number of devices

Ring Attention is ideal for:
- Training on very long documents or books
- Processing entire codebases as context
- Multi-document reasoning tasks
- Any scenario requiring exact attention over long sequences

Example:
    >>> from easydel.layers.attention_operator import OperationMetadata
    >>> from easydel.layers.attention_operator.modules import RingAttn
    >>>
    >>> # Configure for distributed execution
    >>> metadata = OperationMetadata(
    ...     runtime_dtype=jnp.bfloat16,
    ...     blocksize_q=512,  # Query block size
    ...     blocksize_k=1024,  # Key block size
    ...     sequence_axis_name="sp",  # Sequence parallel axis
    ...     scan_ring_attention=True
    ... )
    >>> ring_attn = RingAttn(metadata)
    >>>
    >>> # Process ultra-long sequences
    >>> output = ring_attn(query, key, value, causal=True)

References:
    - Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
    - https://arxiv.org/abs/2310.01889
"""

import jax
from ejkernel.modules import ring_attention  # pyright: ignore[reportMissingTypeStubs]
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float
from spectrax import common_types

from easydel.caching import RaggedPagesCacheView, TransformerCacheView, UnifiedAttentionCacheView

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from .._operation_meta import normalize_head_sinks
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
)
from ._mask_info import align_mask_info_to_qkv_specs
from .vanilla_attention import VanillaAttn


@OperationRegistry.register
class RingAttn(OperationImpl):
    """
    Ring attention implementation for distributed and memory-efficient processing.

    Ring attention processes attention in a ring topology, where each device/chunk
    communicates with neighbors to compute attention over very long sequences.
    This is particularly useful for sequences that don't fit in memory.

    Features:
        - Memory-efficient chunked processing
        - Distributed computation across devices
        - Support for sequences > 100K tokens
        - Native JAX scan-based implementation
        - TPU-optimized Pallas kernels

    Registered name: "ring"

    Attributes:
        metadata: OperationMetadata configuration
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """Get the registered name for this attention implementation.

        Returns:
            str: The name "ring" used for registry lookup.
        """
        return "ring"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for RingAttn.

        Dense calls retain their transformer-cache contract. Paged serving
        additionally requires request boundaries, context lengths and page tables;
        the registered paged adapter updates KV and invokes the actual ring kernel.
        """
        return OperationRequirements.create(
            name="ring",
            required_metadata=MetadataField.basic(),
            optional_metadata=(
                MetadataField.CONTEXT_LENS
                | MetadataField.QUERY_START_LOC
                | MetadataField.PAGES_TABLES
                | MetadataField.SLOT_MAPPING
                | MetadataField.REQUEST_DISTRIBUTION
            ),
            supported_cache=CacheType.TRANSFORMER | CacheType.HYBRID | CacheType.RAGGED_PAGES,
            cache_view_class=TransformerCacheView,
        )

    @jax.named_scope("easydel-ringimpl")
    def forward_native(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        mask_info: MaskInfo | None = None,
        logits_soft_cap: float | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        causal: bool = True,
        fused_backward: bool = False,
        **ignore,
    ) -> AttentionOutput:
        """Compute attention using ejkernel's ``ring_attention`` kernel.

        Handles MLA-style fallback to :class:`VanillaAttn` when QKV head
        dims disagree, and otherwise dispatches to the scan-based ring
        kernel under the configured sequence-axis name.

        Args:
            query: Query tensor of shape ``(B, seq_len_q, num_heads,
                head_dim)``.
            key: Key tensor of shape ``(B, seq_len_k, num_kv_heads,
                head_dim)``.
            value: Value tensor of shape ``(B, seq_len_k, num_kv_heads,
                head_dim)``.
            softmax_aux: Optional sink-token logits, shape
                ``(num_kv_heads, num_sinks)``, ``(num_q_heads, num_sinks)``
                or ``(num_sinks,)``. A 1-D vector of length ``num_q_heads`` is
                per-query-head; other 1-D lengths are shared sinks. Per-head
                sinks are passed to the kernel as 2-D ``(num_q_heads, S)``.
            mask_info: Optional :class:`MaskInfo` describing the attention
                pattern; axis names are populated from the resolved
                query/key sharding before the kernel is invoked.
            logits_soft_cap: Optional soft-cap for logits.
            softmax_scale: Logits scaling factor; defaults to
                ``1 / sqrt(head_dim)``.
            sliding_window: ``int`` or ``(left, right)`` tuple for
                sliding-window attention.
            causal: Whether to apply causal masking. Forced ``False`` in
                decode mode.
            fused_backward: Use the fused backward kernel during training.
            **ignore: Forward-compatibility kwargs (ignored).

        Returns:
            AttentionOutput: ``attention_outputs`` of shape ``(B, seq_len_q,
            num_heads, head_dim)``. ``attention_weights`` is ``None``.
        """
        cache_view = ignore.pop("cache_view", None)
        cache_metadata = ignore.pop("cache_metadata", None)
        if isinstance(cache_view, (RaggedPagesCacheView, UnifiedAttentionCacheView)):
            paged = OperationRegistry.create("paged_attention_reference", self.metadata)
            return paged(
                query=query,
                key=key,
                value=value,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
                dense_impl="ring",
                softmax_aux=softmax_aux,
                mask_info=mask_info,
                logits_soft_cap=logits_soft_cap,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                causal=causal,
                **ignore,
            )
        query_dim: int = query.shape[-1]
        key_dim: int = key.shape[-1]
        value_dim: int = value.shape[-1]
        dims_incompatible: bool = not (query_dim == key_dim == value_dim)

        if dims_incompatible:
            vanilla_attn: VanillaAttn = OperationRegistry.create("vanilla", self.metadata)
            fallback_output: AttentionOutput = vanilla_attn(
                query=query,
                key=key,
                value=value,
                softmax_aux=softmax_aux,
                mask_info=mask_info,
                logits_soft_cap=logits_soft_cap,
                sliding_window=sliding_window,
                softmax_scale=softmax_scale,
                fused_backward=fused_backward,
                causal=causal,
                **ignore,
            )
            return fallback_output

        pairwise_mask = None if mask_info is None else getattr(mask_info, "_attention_mask", None)
        # Canonical sink layouts: shared sinks stay 1-D ``(S,)`` (every ring kernel reads
        # 1-D as shared); per-query-head sinks become 2-D ``(Hq, S)`` so they are never
        # mistaken for shared ones. Never ``reshape(-1)``: that flattens multi-sink rows.
        softmax_aux, per_head_sinks = normalize_head_sinks(
            softmax_aux,
            num_q_heads=query.shape[2],
            num_kv_heads=key.shape[2],
            per_head_2d=True,
            expand_kv_rows=True,
            disambiguate_shared=False,
        )

        model_mode = self.get_mode(query=query, BTHD=True)
        is_decode_mode = model_mode == common_types.MODE_DECODE
        causal_computed: bool = causal if not is_decode_mode else False
        head_dim: int = query.shape[-1]
        softmax_scale_computed: float = softmax_scale if softmax_scale is not None else head_dim**-0.5
        dtype_runtime: jnp.dtype = self.metadata.runtime_dtype
        model_mode = self.get_mode(query=query, BTHD=True)

        shardings = self.metadata.get_shardings(
            mode=model_mode,
            layout="bthd",
            qkv_mni_sharding=True,
            softmax_aux=softmax_aux,
            num_q_heads=query.shape[2],
            num_kv_heads=key.shape[2],
        )

        query: Float[Array, "batch seq_len_q num_heads head_dim"] = query.astype(dtype_runtime)
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"] = key.astype(dtype_runtime)
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"] = value.astype(dtype_runtime)

        query_sharding = self.create_stable_sharding(
            shardings.query,
            dep=query,
            tensor=query,
            preserved_indices=[0, 1, 2],
        )

        # Preserve the global GQA grouping within each BTHD head shard.
        key, value = self.expand_kv_heads_for_shard(
            key, value, head_axis=2, query_sharding=query_sharding, query_head_axis=2
        )

        key_sharding = self.create_stable_sharding(
            shardings.key,
            dep=key,
            tensor=key,
            preserved_indices=[0, 1, 2],
        )
        value_sharding = self.create_stable_sharding(
            shardings.value,
            dep=value,
            tensor=value,
            preserved_indices=[0, 1, 2],
        )
        # Per-head sinks are 2-D here; any 1-D result is a SHARED sink set and must be replicated.
        # (``get_shardings`` head-shards every 1-D vector of length Hq, which would hand each head
        # shard a slice of a shared ``(1, Hq)`` row.)
        softmax_aux_spec = shardings.softmax_aux
        if softmax_aux is not None and not per_head_sinks:
            softmax_aux_spec = jax.sharding.PartitionSpec(*([None] * softmax_aux.ndim))
        softmax_aux_sharding = self.create_stable_sharding(softmax_aux_spec, dep=softmax_aux, tensor=softmax_aux)
        output_sharding = self.create_stable_sharding(shardings.output, tensor=query, preserved_indices=[0, 1, 2])
        mask_info = align_mask_info_to_qkv_specs(
            mask_info,
            query_spec=query_sharding,
            key_spec=key_sharding,
            layout="bthd",
        )

        dense_bias = ignore.get("bias")
        if pairwise_mask is not None:
            # ejkernel's ring wrapper only consumes segment IDs/positions from
            # MaskInfo. Pairwise causal, sliding and custom masks are NOT
            # equivalence relations and cannot be converted to segment IDs.
            # Preserve them losslessly through the kernel's additive bias input.
            if pairwise_mask.ndim == 2:
                pairwise_mask = pairwise_mask[:, None, None, :]
            elif pairwise_mask.ndim == 3:
                pairwise_mask = pairwise_mask[:, None, :, :]
            pairwise_mask = pairwise_mask[..., : query.shape[1], : key.shape[1]].astype(jnp.bool_)
            # Hard exclusions must remain -inf, not the smallest finite logit.
            # Otherwise causal/segment/window masking can remove every allowed
            # key and the online softmax assigns mass to a pairwise-masked key.
            # The XLA ring handles empty rows with zero output and zero QKV VJP.
            mask_bias = jnp.where(pairwise_mask, 0.0, -jnp.inf)
            dense_bias = mask_bias if dense_bias is None else jnp.where(pairwise_mask, dense_bias, mask_bias)
            # Strip only the dense representation; never derive segment IDs
            # from its row/column patterns. Explicit segments remain independent.
            q_segments = getattr(mask_info, "_q_segment_ids", None)
            kv_segments = getattr(mask_info, "_kv_segment_ids", None)
            mask_info = mask_info.replace(
                _attention_mask=None,
                q_segment_ids=q_segments if q_segments is not None else jnp.zeros(query.shape[:2], dtype=jnp.int32),
                kv_segment_ids=kv_segments if kv_segments is not None else jnp.zeros(key.shape[:2], dtype=jnp.int32),
            )
        if mask_info is not None and mask_info.causal_mask_baked_in:
            causal_computed = False
        if mask_info is not None and mask_info.sliding_window_baked_in:
            sliding_window = None
        if dense_bias is not None:
            dense_bias = jnp.broadcast_to(dense_bias, (query.shape[0], query.shape[2], query.shape[1], key.shape[1]))
        bias_sharding = self.create_stable_sharding(getattr(shardings, "bias", None), dep=dense_bias, tensor=dense_bias)
        outputs: Float[Array, "batch seq_len_q num_heads head_dim"] = ring_attention(
            query,
            key,
            value,
            softmax_aux,
            dense_bias,
            axis_name=self.metadata.sequence_axis_name,
            mask_info=mask_info,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=softmax_scale_computed,
            sliding_window=sliding_window,
            causal=causal_computed,
            fused_backward=fused_backward,
            precision=ignore.get("precision", jax.lax.Precision.DEFAULT),
            cfg=self.metadata.get_operation_config("ring"),
            # The Pallas splash ring path cannot represent additive pairwise
            # bias or per-head sinks. The registered XLA ring supports both,
            # including ring collectives; do not select a non-ring mechanism.
            platform="xla" if per_head_sinks or dense_bias is not None else None,
            mesh=self.metadata.mesh,
            in_specs=(
                query_sharding,
                key_sharding,
                value_sharding,
                softmax_aux_sharding,
                bias_sharding,
            ),
            out_specs=output_sharding,
        )

        result: AttentionOutput = AttentionOutput(attention_weights=None, attention_outputs=outputs)
        return result

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Currently delegates to `forward_native` (scan-based).

        Future versions may include GPU-specific ring attention kernels.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass. Delegates to `forward_native` (scan-based).

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA GPU forward pass. Currently delegates to `forward_native` (scan-based).

        Future versions may include CUDA-specific ring attention kernels.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Currently delegates to `forward_native` (scan-based).

        Future versions may include ROCm-specific ring attention kernels.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        mask_info: MaskInfo | None = None,
        logits_soft_cap: float | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        causal: bool = True,
        fused_backward: bool = False,
        **ignore,
    ) -> AttentionOutput:
        """Run Ring Attention by directly invoking :meth:`forward_native`.

        Bypasses the backend dispatch in :class:`OperationImpl` because all
        backends currently use the same scan/Pallas-based kernel.

        Args:
            query: Query tensor ``(B, seq_len_q, num_heads, head_dim)``.
            key: Key tensor ``(B, seq_len_k, num_kv_heads, head_dim)``.
            value: Value tensor ``(B, seq_len_k, num_kv_heads, head_dim)``.
            softmax_aux: Optional sink-token logits.
            mask_info: Optional mask info object.
            logits_soft_cap: Optional soft-cap for logits.
            softmax_scale: Optional logits scaling factor.
            sliding_window: Optional sliding-window size or tuple.
            causal: Whether to apply causal masking.
            fused_backward: Use the fused backward kernel.
            **ignore: Forward-compatibility kwargs (ignored).

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(
            query=query,
            key=key,
            value=value,
            softmax_aux=softmax_aux,
            mask_info=mask_info,
            logits_soft_cap=logits_soft_cap,
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            fused_backward=fused_backward,
            causal=causal,
            **ignore,
        )


@OperationRegistry.register
class RingPagedAttn(RingAttn):
    """Ring serving adapter with paged allocation and continuous-batch metadata.

    ``ring`` keeps the transformer allocation default. Serving
    builders select ``ring_paged`` to allocate paged KV and supply request/page
    metadata; both use the same registered ring calculation after request-local
    page gathers. No paged mechanism is silently substituted for ring attention.
    """

    @classmethod
    def get_impl_name(cls):
        return "ring_paged"

    @classmethod
    def get_requirements(cls, mode=ExecutionMode.MIXED):
        return OperationRequirements.create(
            name=cls.get_impl_name(),
            required_metadata=(
                MetadataField.basic()
                | MetadataField.CONTEXT_LENS
                | MetadataField.QUERY_START_LOC
                | MetadataField.PAGES_TABLES
            ),
            optional_metadata=MetadataField.SLOT_MAPPING | MetadataField.REQUEST_DISTRIBUTION,
            supported_cache=CacheType.RAGGED_PAGES,
            preferred_cache=CacheType.RAGGED_PAGES,
            cache_view_class=RaggedPagesCacheView,
        )


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    b, qs, ks, qh, kh, d, vd = 1, 1024, 1024, 32, 32, 128, 128
    q = jr.normal(jr.key(0), (b, qs, qh, d), "f2")
    k = jr.normal(jr.key(1), (b, ks, kh, d), "f2")
    v = jr.normal(jr.key(2), (b, ks, kh, vd), "f2")
    mask_info = MaskInfo.from_random(b, qs, ks)
    ring = RingAttn(
        OperationMetadata(
            runtime_dtype=jnp.bfloat16,
            base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, -1)),
        )
    )
    from ejkernel.modules import attention  # pyright: ignore[reportMissingTypeStubs]

    out = attention(q, k, v, mask_info=mask_info)[0]
    vout = ring(query=q, key=k, value=v, mask_info=mask_info).attention_outputs

    print(out[-1, -1, -1, -5:], out[-1, 0, -1, -5:])
    if vout is None:
        raise RuntimeError("ring attention returned None for attention_outputs")
    print(vout[-1, -1, -1, -5:], vout[-1, 0, -1, -5:])

    print(jnp.allclose(out, vout, atol=0.125))
