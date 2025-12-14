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
from eformer import common_types
from ejkernel.modules import ring_attention
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float

from easydel.layers.caching import TransformerCacheView

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

    def get_impl_metadata(self) -> OperationMetadata:
        """Get the metadata configuration for this attention instance.

        Returns:
            OperationMetadata: Configuration including dtype, mesh, etc.
        """
        return self.metadata

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for RingAttn.

        Ring attention requires basic metadata and uses TransformerCacheView
        for KV-cache management.
        """
        return OperationRequirements.create(
            name="ring",
            required_metadata=MetadataField.basic(),
            supported_cache=CacheType.TRANSFORMER | CacheType.HYBRID,
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
        """
        Computes attention using the scan-based `blockwise_attn` function.

        Handles optional mask/bias, KV head repetition, and sharding constraints.

        Args:
            q: Query tensor (B, T, H, D).
            k: Key tensor (B, S, H_kv, D).
            v: Value tensor (B, S, H_kv, D).
            mask: Optional boolean attention mask (broadcastable to B, 1, T, S).
            bias: Optional attention bias (broadcastable to B, H, T, S).
            init_bias: Optional callable to initialize bias if mask/bias are None.
            deterministic: If False, enables dropout. Requires `dropout_rng`.
            dropout_rng: JAX PRNG key for dropout if `deterministic` is False.
            causal: Apply causal mask if True.
            **ignore: Ignored keyword arguments.

        Returns:
            AttentionOutput containing the attention result.
        """
        # Check dimension compatibility for MLA-style attention
        query_dim: int = query.shape[-1]
        key_dim: int = key.shape[-1]
        value_dim: int = value.shape[-1]
        dims_incompatible: bool = query_dim != value_dim != key_dim

        if dims_incompatible:
            vanilla_attn: VanillaAttn = VanillaAttn(self.metadata)
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

        if softmax_aux is not None:
            softmax_aux = softmax_aux.reshape(-1)

        model_mode = self.get_mode(query=query, BTHD=False)
        is_decode_mode = model_mode == common_types.MODE_DECODE
        causal_computed: bool = causal if not is_decode_mode else False
        head_dim: int = query.shape[-1]
        softmax_scale_computed: float = softmax_scale if softmax_scale is not None else head_dim**-0.5
        dtype_runtime: jnp.dtype = self.metadata.runtime_dtype
        model_mode = self.get_mode(query=query, BTHD=False)

        shardings = self.metadata.get_shardings(
            mode=model_mode,
            layout="bthd",
            qkv_mni_sharding=True,
            softmax_aux=softmax_aux,
        )

        # Cast tensors to runtime dtype
        query: Float[Array, "batch seq_len_q num_heads head_dim"] = query.astype(dtype_runtime)
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"] = key.astype(dtype_runtime)
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"] = value.astype(dtype_runtime)

        # Create sharding specs
        query_sharding = self.create_stable_sharding(
            shardings.query,
            dep=query,
            tensor=query,
            preserved_indices=[0, 1, 2],
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
        softmax_aux_sharding = self.create_stable_sharding(shardings.softmax_aux, dep=softmax_aux, tensor=softmax_aux)
        output_sharding = self.create_stable_sharding(shardings.output, tensor=query, preserved_indices=[0, 1, 2])

        outputs: Float[Array, "batch seq_len_q num_heads head_dim"] = ring_attention(
            query,
            key,
            value,
            softmax_aux,
            None,
            axis_name=self.metadata.sequence_axis_name,
            mask_info=mask_info,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=softmax_scale_computed,
            sliding_window=sliding_window,
            causal=causal_computed,
            fused_backward=fused_backward,
            cfg=self.metadata.get_operation_config("ring"),
            mesh=self.metadata.mesh,
            in_specs=(
                query_sharding,
                key_sharding,
                value_sharding,
                softmax_aux_sharding,
                None,
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
        """
        Executes the Ring Attention computation.

        Currently bypasses the backend dispatch and directly calls `forward_native`.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            mask: Optional attention mask.
            bias: Optional attention bias.
            init_bias: Optional callable to initialize bias.
            deterministic: If False, enables dropout (requires dropout_rng).
            dropout_rng: JAX PRNG key for dropout if deterministic is False.
            causal: Apply causal mask if True.
            **ignore: Additional ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the results.
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
            base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, -1)),
        )
    )
    from ejkernel.modules import attention

    out = attention(q, k, v, mask_info=mask_info)[0]
    vout = ring(query=q, key=k, value=v, mask_info=mask_info).attention_outputs

    print(out[-1, -1, -1, -5:], out[-1, 0, -1, -5:])
    print(vout[-1, -1, -1, -5:], vout[-1, 0, -1, -5:])

    print(jnp.allclose(out, vout, atol=0.125))
