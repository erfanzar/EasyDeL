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


from eformer import common_types as ct
from ejkernel.modules import ragged_page_attention
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, DTypeLike, Float

from easydel.layers.caching import RaggedPagesCacheView, RaggedPagesMetadata

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry

USE_SHARDMAP = True


@OperationRegistry.register
class RaggedPageAttn(OperationImpl):
    """
    Attention implementation using the Paged Attention mechanism with TPU Pallas kernels.

    This class provides an attention mechanism suitable for scenarios where the
    Key-Value cache is managed in non-contiguous pages (Paged KV Cache). It leverages
    specialized kernels
    for efficient execution on TPUs, handling prefill and decode phases separately
    or in a mixed mode.

    Attributes:
        metadata (OperationMetadata): Configuration metadata for the attention mechanism.
            While this class uses `OperationMetadata`, it primarily relies on the
            additional `RaggedPagesMetadata` passed during the forward call for
            paged-specific information.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name for this attention implementation.

        Returns:
            tp.Union[str, tp.Tuple[str]]: The name "ragged_page_attention".
        """
        return "ragged_page_attention"

    def get_impl_metadata(self) -> OperationMetadata:
        """
        Retrieves the metadata associated with this attention implementation instance.

        Returns:
            OperationMetadata: The metadata object provided during initialization.
        """
        return self.metadata

    def forward_native(
        self,
        query: Float[Array, "total_tokens num_q_heads head_dim"],
        cache_view: RaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        compute_dtype: DTypeLike = jnp.bfloat16,
        optimized: bool = False,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        mask_value: float | None = None,
        vmem_limit_bytes: int | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Native forward pass for paged attention using ragged format.

        This implementation handles attention with a paged KV cache stored in non-contiguous
        memory pages. It uses the `ragged_page_attention` kernel which efficiently processes
        variable-length sequences with page table lookups.

        Args:
            query: Query tensor [total_tokens, num_q_heads, head_dim] in ragged format.
                Total_tokens is the sum of all sequence lengths in the batch.
            cache_view: Paged KV cache view containing:
                - kv_pages: Paged key/value tensors [num_pages, page_size, num_kv_heads, head_dim].
            cache_metadata: Metadata for paged cache including:
                - context_lens: Length of each sequence [num_seqs].
                - pages_tables: Page table for cache access [num_seqs, max_pages].
                - query_start_loc: Starting index for each sequence [num_seqs + 1].
                - num_seqs: Number of sequences in the batch.
            softmax_scale: Scaling factor for attention logits. Defaults to 1/sqrt(head_dim).
            logits_soft_cap: Soft capping value for attention logits. Optional.
            compute_dtype: Data type for kernel computation. Defaults to bfloat16.
            optimized: Use optimized kernel variant if available. Defaults to False.
            sliding_window: Sliding window size for local attention. Optional.
            softmax_aux: Auxiliary softmax tensor for sink tokens. Optional.
            mask_value: Value for masked positions. Optional.
            vmem_limit_bytes: VMEM limit for TPU memory management. Optional.
            **ignore: Additional ignored arguments.

        Returns:
            AttentionOutput: Contains attention outputs [total_tokens, num_q_heads, head_dim].
                Attention weights are not computed.
        """
        kv_pages = cache_view.kv_pages
        manager = self.metadata.partition_manager
        resolve = manager.resolve
        num_seqs = cache_metadata.num_seqs.reshape(-1)
        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=query.shape)

        aux_spec = PartitionSpec(None)
        if softmax_aux is not None:
            if softmax_aux.ndim == 2:
                aux_spec = resolve(axes=[ct.KV_HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)
            elif softmax_aux.ndim == 1:
                aux_spec = resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)

        if compute_dtype is None:
            compute_dtype = jnp.bfloat16

        output = ragged_page_attention(
            query,
            kv_pages,
            cache_metadata.context_lens,
            cache_metadata.pages_tables,
            cache_metadata.query_start_loc,
            num_seqs,
            softmax_aux,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            vmem_limit_bytes=vmem_limit_bytes,
            optimized=optimized,
            compute_dtype=compute_dtype,
            mask_value=mask_value,
            sliding_window=sliding_window,
            in_specs=(
                qaxes,
                resolve(axes=[ct.EMPTY, ct.EMPTY, ct.KV_HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=kv_pages.shape),
                Ps(),
                Ps(),
                Ps(),
                Ps(),
                aux_spec,
            ),
            out_specs=qaxes,
            mesh=self.metadata.mesh,
        )
        return AttentionOutput(attention_weights=None, attention_outputs=output)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch 1 num_heads head_dim"],
        cache_view: RaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        compute_dtype: DTypeLike = jnp.bfloat16,
        optimized: bool = False,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        mask_value: float | None = None,
        vmem_limit_bytes: int | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Executes paged attention by dispatching to the appropriate backend implementation.

        This method handles attention with non-contiguous paged KV cache, preprocessing
        the query tensor by reshaping if needed, then restoring the original shape in the output.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim] or [batch, num_heads, head_dim].
            cache_view: Contains the paged KV cache tensors with page table information.
            cache_metadata: Metadata describing batch state including:
                - context_lens: Length of each sequence in the batch.
                - pages_tables: Page table mapping for cache access.
                - query_start_loc: Starting locations for queries in ragged format.
                - num_seqs: Number of sequences in the batch.
            softmax_scale: Scaling factor for attention logits. Defaults to 1/sqrt(head_dim).
            logits_soft_cap: Soft capping value for attention logits. Optional.
            compute_dtype: Data type for computation (e.g., bfloat16, float32).
            optimized: Use optimized kernel variant if available.
            sliding_window: Sliding window size for local attention. Optional.
            softmax_aux: Auxiliary softmax tensor for sink tokens. Optional.
            mask_value: Value to use for masked positions. Optional.
            vmem_limit_bytes: VMEM limit in bytes for TPU memory management. Optional.
            **ignore: Additional ignored keyword arguments.

        Returns:
            AttentionOutput: Contains attention outputs with shape matching input query.
                Attention weights are not computed.
        """
        if query.ndim == 4:
            batch, sequence, head, dim = query.shape
        output = super().__call__(
            query=query.reshape(-1, *query.shape[-2:]),
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            vmem_limit_bytes=vmem_limit_bytes,
            optimized=optimized,
            compute_dtype=compute_dtype,
            softmax_aux=softmax_aux,
            mask_value=mask_value,
            sliding_window=sliding_window,
            **ignore,
        )
        if query.ndim == 4:
            output.attention_outputs = output.attention_outputs.reshape(batch, sequence, head, dim)
        return output
