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

"""Flash Attention V2 implementation for EasyDeL.

This module provides optimized Flash Attention V2 implementations for TPU and GPU
backends using JAX's Pallas operations and Triton kernels respectively. Flash Attention
is a memory-efficient attention mechanism that reduces memory usage from O(N²) to O(N)
by computing attention in blocks and avoiding materialization of the full attention matrix.

Key Features:
- TPU implementation using JAX Pallas operations
- GPU implementation using Triton kernels
- Support for causal masking
- Support for attention bias
- Efficient handling of multi-query and grouped-query attention
- Automatic sharding for distributed computation

The implementation follows the Flash Attention V2 algorithm which:
1. Processes attention in blocks to minimize HBM access
2. Uses online softmax computation to avoid storing the full attention matrix
3. Achieves significant speedup and memory savings compared to standard attention

Note: This implementation does not support CPU execution as Flash Attention
relies on specialized hardware features available only on TPUs and GPUs.

Example:
    >>> from easydel.layers.attention_operator import OperationMetadata
    >>> from easydel.layers.attention_operator.modules import FlashAttn
    >>>
    >>> metadata = OperationMetadata(
    ...     runtime_dtype=jnp.float16,
    ...     softmax_scale=1.0 / math.sqrt(head_dim),
    ...     blocksize_q=512,
    ...     blocksize_k=1024
    ... )
    >>> flash_attn = FlashAttn(metadata)
    >>> output = flash_attn(query, key, value, causal=True)
"""

import jax
from eformer import common_types
from eformer.escale import with_sharding_constraint
from ejkernel.modules import flash_attention
from ejkernel.types import MaskInfo
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float, Int

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
class FlashAttn(OperationImpl):
    """
    An implementation of Flash Attention V2 using specialized JAX primitives.

    This class leverages `jax.experimental.pallas.ops.tpu.flash_attention` for TPUs
    and a Triton kernel (`triton_flash_attention`) for GPUs (CUDA). It is registered
    under the name "flash_attn2". CPU execution is not supported and will raise an error.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name of this attention implementation.

        Returns:
            The string "flash_attn2".
        """
        return "flash_attn2"

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
        """Returns requirements for FlashAttn.

        FlashAttention requires basic metadata and uses TransformerCacheView
        for KV-cache management.
        """
        return OperationRequirements.create(
            name="flash",
            required_metadata=MetadataField.basic(),
            supported_cache=CacheType.TRANSFORMER | CacheType.HYBRID,
            cache_view_class=TransformerCacheView,
        )

    def forward_native(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        mask_info: MaskInfo | None = None,
        bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
        softmax_scale: float | None = None,
        dropout_prob: float = 0.0,
        causal: bool = False,
        dropout_seed: int | None = None,
        cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,  # noqa
        cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,  # noqa
        sliding_window: int | tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        normalize_output: bool = True,
        precision: lax.PrecisionLike = lax.Precision.DEFAULT,
        **ignore,
    ) -> AttentionOutput:
        """
        Performs Flash Attention V2 using optimized kernels (TPU Pallas or GPU Triton).

        Flash Attention V2 is a memory-efficient attention mechanism that reduces memory usage
        from O(N²) to O(N) by computing attention in blocks and avoiding materialization of
        the full attention matrix. This implementation uses specialized kernels for different
        hardware backends.

        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim].
            key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim].
            value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim].
            attention_mask: Optional boolean attention mask [batch, num_heads_or_1, seq_len_q, seq_len_k].
                Used by the kernel if bias is not provided.
            bias: Optional attention bias tensor [batch, num_heads, seq_len_q, seq_len_k].
                Added to attention logits before softmax. Takes precedence over attention_mask.
            softmax_scale: Scaling factor for attention logits. Defaults to 1/sqrt(head_dim).
            dropout_prob: Dropout probability for attention weights. Defaults to 0.0.
            causal: If True, applies causal (autoregressive) masking. Defaults to False.
            dropout_seed: Random seed for dropout. Optional.
            cum_seqlens_q: Cumulative sequence lengths for queries (for variable-length sequences).
            cum_seqlens_k: Cumulative sequence lengths for keys (for variable-length sequences).
            sliding_window: Sliding window size for local attention. Optional.
            logits_soft_cap: Soft capping value for attention logits. Optional.
            softmax_aux: Auxiliary softmax tensor (e.g., for sink tokens). Optional.
            normalize_output: Whether to normalize the output. Defaults to True.
            precision: JAX precision setting for matmul operations.
            q_segment_ids: Segment IDs for queries. Optional.
            kv_segment_ids: Segment IDs for keys/values. Optional.
            **ignore: Additional ignored keyword arguments.

        Returns:
            AttentionOutput: Object containing attention outputs [batch, seq_len_q, num_heads, head_dim].
                Attention weights are not computed for efficiency.
        """
        head_dim: int = query.shape[-1]
        softmax_scale_computed: float = softmax_scale if softmax_scale is not None else head_dim**-0.5

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
                bias=bias,
                softmax_aux=softmax_aux,
                mask_info=mask_info,
                logits_soft_cap=logits_soft_cap,
                softmax_scale=softmax_scale_computed,
                sliding_window=sliding_window,
                causal=causal,
                **ignore,
            )
            return fallback_output

        dtype: jnp.dtype = self.metadata.runtime_dtype
        model_mode: common_types.RUNTIME_MODE_TYPES = self.get_mode(query=query, BTHD=True)  # type: ignore
        shardings = self.metadata.get_shardings(model_mode, layout="bthd")

        is_decode_mode = model_mode == common_types.MODE_DECODE
        causal_computed: bool = causal if not is_decode_mode else False

        # Cast tensors to runtime dtype
        query: Float[Array, "batch seq_len_q num_heads head_dim"] = query.astype(dtype)
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"] = key.astype(dtype)
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"] = value.astype(dtype)
        bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = (
            bias.astype(dtype) if bias is not None else None
        )

        # Create sharding specs
        query_sharding = self.create_stable_sharding(
            shardings.query,
            tensor=query,
            preserved_indices=[0, 2],
        )
        key_sharding = self.create_stable_sharding(
            shardings.key,
            tensor=key,
            preserved_indices=[0, 2],
        )
        value_sharding = self.create_stable_sharding(
            shardings.value,
            tensor=value,
            preserved_indices=[0, 2],
        )
        bias_sharding = self.create_stable_sharding(
            shardings.bias,
            dep=bias,
            tensor=bias,
            preserved_indices=[0, 1],
        )
        cum_seqlens_q_sharding = self.create_stable_sharding(
            PartitionSpec(None),
            dep=cum_seqlens_q,
            tensor=cum_seqlens_q,
        )
        cum_seqlens_k_sharding = self.create_stable_sharding(
            PartitionSpec(None),
            dep=cum_seqlens_k,
            tensor=cum_seqlens_k,
        )
        softmax_aux_sharding = self.create_stable_sharding(
            shardings.softmax_aux,
            dep=softmax_aux,
            tensor=softmax_aux,
        )
        output_sharding = self.create_stable_sharding(
            shardings.output,
            tensor=query,
            preserved_indices=[0, 2],
        )

        attn: Float[Array, "batch seq_len_q num_heads head_dim"] = flash_attention(
            query,
            key,
            value,
            bias,
            cum_seqlens_q,
            cum_seqlens_k,
            softmax_aux,
            mask_info=mask_info,
            dropout_prob=dropout_prob,
            causal=causal_computed,
            dropout_seed=dropout_seed,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            normalize_output=normalize_output,
            precision=precision,
            logits_dtype=jnp.bfloat16,
            cfg=self.metadata.get_operation_config("flash_attn2"),
            mesh=self.metadata.mesh,
            in_specs=(
                query_sharding,
                key_sharding,
                value_sharding,
                bias_sharding,
                cum_seqlens_q_sharding,
                cum_seqlens_k_sharding,
                softmax_aux_sharding,
            ),
            out_specs=output_sharding,
        )

        attn_sharded: Float[Array, "batch seq_len_q num_heads head_dim"] = with_sharding_constraint(
            arr=attn, sharding=shardings.output
        )
        output: AttentionOutput = AttentionOutput(attention_weights=None, attention_outputs=attn_sharded)
        return output

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """
        GPU forward pass. Delegates to the CUDA-specific implementation.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_gpu(*args, **kwargs)

    @jax.named_scope("easydel-flash-attnimpl-tpu")
    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """
        GPU forward pass. Delegates to the CUDA-specific implementation.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """
        CPU forward pass. Delegates to `forward_native`, which raises an error.

        Raises:
            NotImplementedError: Via `forward_native`.
        """
        return self.forward_native(*args, **kwargs)

    @jax.named_scope("easydel-flash-attnimpl-gpu-cuda-rocm")
    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """
        GPU forward pass. Delegates to the CUDA-specific implementation.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """
        ROCm GPU forward pass.

        Currently delegates to the standard GPU implementation. Future versions
        may include ROCm-specific optimizations using hipFlashAttention or similar.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        # ROCm would require a specific hipFlashAttention kernel or similar
        return self.forward_gpu(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        mask_info: MaskInfo | None = None,
        bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
        softmax_scale: float | None = None,
        dropout_prob: float = 0.0,
        causal: bool = False,
        dropout_seed: int | None = None,
        cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,  # noqa
        cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,  # noqa
        sliding_window: int | tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        normalize_output: bool = True,
        precision: lax.PrecisionLike = lax.Precision.DEFAULT,
        **ignore,
    ) -> AttentionOutput:
        """
        Executes Flash Attention V2 by dispatching to the appropriate backend implementation.

        This method automatically selects the optimal backend (TPU, GPU, CPU) based on the
        runtime environment and calls the corresponding forward method.

        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim].
            key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim].
            value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim].
            attention_mask: Optional boolean mask [batch, num_heads_or_1, seq_len_q, seq_len_k].
            bias: Optional attention bias [batch, num_heads, seq_len_q, seq_len_k].
            softmax_scale: Scaling factor for attention logits.
            dropout_prob: Dropout probability.
            causal: Apply causal masking.
            dropout_seed: Random seed for dropout.
            cum_seqlens_q: Cumulative sequence lengths for queries.
            cum_seqlens_k: Cumulative sequence lengths for keys.
            sliding_window: Sliding window size.
            logits_soft_cap: Soft capping value.
            softmax_aux: Auxiliary softmax tensor.
            normalize_output: Normalize output flag.
            precision: JAX precision setting.
            q_segment_ids: Query segment IDs.
            kv_segment_ids: Key/value segment IDs.
            **ignore: Additional ignored arguments.

        Returns:
            AttentionOutput: Contains attention outputs and optionally attention weights.
        """
        return super().__call__(
            query=query,
            key=key,
            value=value,
            bias=bias,
            softmax_aux=softmax_aux,
            cum_seqlens_k=cum_seqlens_k,
            cum_seqlens_q=cum_seqlens_q,
            mask_info=mask_info,
            softmax_scale=softmax_scale,
            dropout_prob=dropout_prob,
            causal=causal,
            dropout_seed=dropout_seed,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            normalize_output=normalize_output,
            precision=precision,
        )


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    # Test cace when qkv might refer to mla
    b, qs, ks, qh, kh, d, vd = 4, 1024, 1024, 32, 32, 128, 128
    query = jr.normal(jr.key(0), (b, qs, qh, d), "f4")
    key = jr.normal(jr.key(1), (b, ks, kh, d), "f4")
    value = jr.normal(jr.key(2), (b, ks, kh, vd), "f4")
    a = jnp.astype(jr.randint(jr.key(3), (b, 1, qs, ks), 0, 4) > 2, "b1")
    metadata = OperationMetadata(
        runtime_dtype=jnp.bfloat16,
        base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, -1, 1)),
    )
    attn = FlashAttn(metadata)
    vanilla = VanillaAttn(metadata)
    fout = attn(query=query, key=key, value=value, attention_mask=a, causal=False).attention_outputs
    vout = vanilla(query=query, key=key, value=value, attention_mask=a).attention_outputs
    print(fout[-1, -1, -1, -5:])
    print(vout[-1, -1, -1, -5:])
