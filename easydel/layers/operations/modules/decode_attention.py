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

"""Autoregressive decode attention implementation for efficient token generation.

This module provides specialized attention implementations optimized for the
autoregressive decoding phase of transformer models. During generation, models
process one token at a time while attending to all previously generated tokens
stored in a key-value cache.

Key optimizations:
- Single query token processing (query sequence length = 1)
- Efficient cache access with ragged boundaries
- Backend-specific kernels for TPU and GPU
- Optimized memory access patterns for decode phase
- Support for variable sequence lengths per batch element

The implementation uses:
- Pallas kernels for TPU acceleration
- Triton kernels for GPU acceleration
- Native JAX operations as fallback

This is particularly important for:
- Text generation and completion
- Real-time inference serving
- Streaming model outputs
- Interactive applications

Example:
    >>> from easydel.layers.attention_operator import OperationMetadata
    >>> from easydel.layers.attention_operator.modules import AutoRegressiveDecodeAttn
    >>> from easydel.layers.caching.transformer import TransformerMetadata
    >>>
    >>> # Configure for decoding
    >>> metadata = OperationMetadata(
    ...     runtime_dtype=jnp.float16,
    ...     softmax_scale=1.0 / math.sqrt(head_dim)
    ... )
    >>> decode_attn = AutoRegressiveDecodeAttn(metadata)
    >>>
    >>> # Use with cache during generation
    >>> cache_metadata = TransformerMetadata(
    ...     starts=jnp.array([0, 0, 0, 0]),  # Start indices per batch
    ...     indexs=jnp.array([10, 15, 8, 12])  # Current lengths per batch
    ... )
    >>> output = decode_attn(query, cached_keys, cached_values, cache_metadata)
"""

import jax
from eformer import common_types
from ejkernel.modules import ragged_decode_attention
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Float

from easydel.layers.caching import TransformerCacheView
from easydel.layers.caching.transformer import TransformerMetadata

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
)
from .vanilla_attention import VanillaAttn


@OperationRegistry.register
class AutoRegressiveDecodeAttn(OperationImpl):
    """
    Attention implementation tailored for the autoregressive decoding step.

    This class handles the attention mechanism when generating tokens one by one,
    attending to the previously generated sequence stored in a cache. It utilizes
    `shard_map` for distributed computation and supports different backends,
    including a potential Pallas-optimized version for TPUs. It assumes the
    query sequence length is 1.

    Attributes:
        metadata (OperationMetadata): Configuration metadata for the attention mechanism.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name of this attention implementation.

        Returns:
            The string "autoregressive_decodeattn".
        """
        return "autoregressive_decodeattn"

    def get_impl_metadata(self) -> OperationMetadata:
        """
        Returns the metadata associated with this attention implementation instance.

        Returns:
            The `OperationMetadata` provided during initialization.
        """
        return self.metadata

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for AutoRegressiveDecodeAttn.

        Decode attention operates specifically in decode mode and requires
        basic metadata plus context lens. It works with transformer or hybrid cache.
        Uses TransformerCacheView for KV-cache management.
        """
        return OperationRequirements.create(
            name="autoregressive_decode",
            required_metadata=MetadataField.basic() | MetadataField.CONTEXT_LENS,
            supported_cache=CacheType.TRANSFORMER | CacheType.HYBRID,
            cache_view_class=TransformerCacheView,
        )

    @jax.named_scope("easydel-autoregressive_decodeattn")
    def forward_native(
        self,
        query: Float[Array, "batch 1 num_q_heads head_dim"],
        key: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        cache_metadata: TransformerMetadata,
        softmax_scale: float | None = None,
        sliding_window: tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        **ignores,
    ) -> AttentionOutput:
        """
        Performs the native JAX/XLA forward pass for autoregressive decoding attention.

        This implementation uses `shard_map` to distribute the computation across devices
        and leverages the `ragged_decode_attention` kernel for efficient processing.
        It calculates attention between a single query token and all previous keys/values
        stored in the cache, respecting the valid range defined by cache metadata.

        Args:
            query: Query tensor [batch, 1, num_q_heads, head_dim].
                Single token query for next-token prediction.
            key: Key tensor from cache [batch, kv_seq_len, num_kv_heads, head_dim].
                All previous keys in the sequence.
            value: Value tensor from cache [batch, kv_seq_len, num_kv_heads, head_dim].
                All previous values in the sequence.
            cache_metadata: Cache metadata containing:
                - starts: Start indices for valid cache entries per batch [batch].
                - indexs: Current sequence lengths per batch [batch].
            softmax_scale: Scaling factor for attention logits. Defaults to 1/sqrt(head_dim).
            sliding_window: Window bounds (left, right) for local attention. Optional.
            logits_soft_cap: Soft capping value to prevent extreme attention logits. Optional.
            softmax_aux: Auxiliary tensor for sink token attention. Optional.
            **ignores: Additional ignored keyword arguments.

        Returns:
            AttentionOutput containing:
                - attention_outputs: [batch, 1, num_q_heads, head_dim]
                  Attended representation for the current query token.
                - attention_weights: None (not computed for memory efficiency).
        """
        if jax.default_backend() != "gpu":
            vanilla_attn: VanillaAttn = VanillaAttn(self.metadata)
            fallback_output_1: AttentionOutput = vanilla_attn(
                query=query,
                key=key,
                value=value,
                cache_metadata=cache_metadata,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                softmax_aux=softmax_aux,
                **ignores,
            )
            return fallback_output_1
        head_dim: int = query.shape[-1]
        softmax_scale_computed: float = softmax_scale if softmax_scale is not None else head_dim**-0.5
        model_mode: common_types.RUNTIME_MODE_TYPES = self.get_mode(query=query, BTHD=True)  # type: ignore
        assert model_mode == common_types.MODE_DECODE, "AutoRegressiveDecodeAttn requires decode mode"

        shardings = self.metadata.get_shardings(model_mode, layout="bthd")

        # Create sharding for cache metadata (batch dimension only)
        views_sharding: Ps = Ps(shardings.query[0])

        # Reshape cache metadata for processing
        starts_2d = cache_metadata.starts.reshape(-1, 1)
        indexs_2d = cache_metadata.indexs.reshape(-1, 1)

        # Extract last query token and flatten cache metadata
        query_squeezed: Float[Array, "batch num_q_heads head_dim"] = query[:, -1, :, :]
        starts_flat = starts_2d.reshape(-1)
        indexs_flat = indexs_2d.reshape(-1)

        # Create sharding specs for all inputs
        query_sharding: Ps | None = self.create_stable_sharding(
            shardings.query3d,
            dep=query_squeezed,
            tensor=query_squeezed,
            preserved_indices=[0, 1],
        )
        key_sharding: Ps | None = self.create_stable_sharding(
            shardings.key,
            dep=key,
            tensor=key,
            preserved_indices=[0, 2],
        )
        value_sharding: Ps | None = self.create_stable_sharding(
            shardings.value,
            dep=value,
            tensor=value,
            preserved_indices=[0, 2],
        )
        starts_sharding: Ps | None = self.create_stable_sharding(
            views_sharding,
            dep=starts_flat,
            tensor=starts_flat,
            preserved_indices=[0],
        )
        indexs_sharding: Ps | None = self.create_stable_sharding(
            views_sharding,
            dep=indexs_flat,
            tensor=indexs_flat,
            preserved_indices=[0],
        )
        softmax_aux_sharding: Ps | None = self.create_stable_sharding(
            shardings.softmax_aux,
            dep=softmax_aux,
            tensor=softmax_aux,
        )
        output_sharding: Ps | None = self.create_stable_sharding(
            shardings.query3d,
            tensor=query_squeezed,
            preserved_indices=[0, 1],
        )
        if sliding_window is not None:
            if isinstance(sliding_window, int):
                sliding_window = (sliding_window, sliding_window)
        attn_output: Float[Array, "batch num_q_heads head_dim"] = ragged_decode_attention(
            query_squeezed,
            key,
            value,
            starts_flat,
            indexs_flat,
            softmax_aux,
            softmax_scale=softmax_scale_computed,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            mesh=self.metadata.mesh,
            in_specs=(
                query_sharding,
                key_sharding,
                value_sharding,
                starts_sharding,
                indexs_sharding,
                softmax_aux_sharding,
            ),
            out_specs=output_sharding,
        )

        # Expand to match expected output shape [batch, 1, num_q_heads, head_dim]
        attn_output_expanded: Float[Array, "batch 1 num_q_heads head_dim"] = jax.numpy.expand_dims(attn_output, 1)

        result: AttentionOutput = AttentionOutput(attention_weights=None, attention_outputs=attn_output_expanded)
        return result

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """
        CPU forward pass for autoregressive decoding attention.

        Delegates to the native JAX/XLA implementation (`forward_native`).

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """
        CPU forward pass for autoregressive decoding attention.

        Delegates to the native JAX/XLA implementation (`forward_native`).

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """
        CPU forward pass for autoregressive decoding attention.

        Delegates to the native JAX/XLA implementation (`forward_native`).

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """
        CUDA GPU forward pass for autoregressive decoding attention.

        Delegates to the GPU implementation which uses Triton kernels.
        Future optimizations might add CUDA-specific kernels here.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_gpu(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """
        ROCm GPU forward pass for autoregressive decoding attention.

        Delegates to the GPU implementation. Future optimizations might
        add ROCm-specific kernels here.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_gpu(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch 1 num_q_heads head_dim"],
        key: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        cache_metadata: TransformerMetadata,
        softmax_scale: float | None = None,
        sliding_window: tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        **ignores,
    ) -> AttentionOutput:
        """
        Executes autoregressive decode attention by dispatching to the appropriate backend.

        This method handles token-by-token attention during autoregressive generation,
        where the query is a single token and keys/values come from the cache.

        Args:
            query: Query tensor [batch, 1, num_q_heads, head_dim]. Single token query.
            key: Key tensor from cache [batch, kv_seq_len, num_kv_heads, head_dim].
            value: Value tensor from cache [batch, kv_seq_len, num_kv_heads, head_dim].
            cache_metadata: Metadata containing cache start indices and current lengths.
            softmax_scale: Scaling factor for attention logits. Defaults to 1/sqrt(head_dim).
            sliding_window: Sliding window bounds for local attention.
            logits_soft_cap: Soft capping value for attention logits.
            softmax_aux: Auxiliary softmax tensor for sink tokens.
            **ignores: Additional ignored arguments.

        Returns:
            AttentionOutput: Contains attention outputs [batch, 1, num_q_heads, head_dim].
                Attention weights are not computed for efficiency.
        """
        return super().__call__(
            query=query,
            key=key,
            value=value,
            cache_metadata=cache_metadata,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=softmax_scale,
            **ignores,
        )
