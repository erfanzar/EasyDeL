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


from functools import partial

import jax
from eformer import common_types as ct
from jax import Array
from jax.sharding import PartitionSpec as Ps

from easydel.kernels.cpu_ops import jax_ragged_paged_attention
from easydel.kernels.gpu_ops import triton_ragged_paged_attention
from easydel.kernels.tpu_ops import pallas_ragged_paged_attention
from easydel.layers.caching import PagesCacheView, PagesMetadata

from .._attention_impl import AttentionImpl, AttentionMetadata, AttentionOutput, AttentionRegistry

USE_SHARDMAP = True


@AttentionRegistry.register
class RaggedPageAttn(AttentionImpl):
    """
    Attention implementation using the Paged Attention mechanism with TPU Pallas kernels.

    This class provides an attention mechanism suitable for scenarios where the
    Key-Value cache is managed in non-contiguous pages (Paged KV Cache). It leverages
    specialized kernels
    for efficient execution on TPUs, handling prefill and decode phases separately
    or in a mixed mode.

    Attributes:
        metadata (AttentionMetadata): Configuration metadata for the attention mechanism.
            While this class uses `AttentionMetadata`, it primarily relies on the
            additional `PagesMetadata` passed during the forward call for
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

    def get_impl_metadata(self) -> AttentionMetadata:
        """
        Retrieves the metadata associated with this attention implementation instance.

        Returns:
            AttentionMetadata: The metadata object provided during initialization.
        """
        return self.metadata

    def forward_native(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagesCacheView,
        cache_metadata: PagesMetadata,
        **ignore,
    ) -> AttentionOutput:
        """
        Native (XLA) forward pass.
        """
        kv_pages = cache_view.kv_pages
        manager = self.metadata.partition_manager
        resolve = manager.resolve
        num_seqs = cache_metadata.num_seqs.reshape(-1)
        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=q.shape)
        fn = partial(
            jax_ragged_paged_attention,
            softmax_scale=self.metadata.softmax_scale,
            soft_cap=self.metadata.soft_cap,
            compute_dtype=self.metadata.runtime_dtype,
        )
        if USE_SHARDMAP:
            fn = jax.shard_map(
                fn,
                in_specs=(
                    qaxes,
                    resolve(axes=[ct.EMPTY, ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=kv_pages.shape),
                    Ps(),
                    Ps(),
                    Ps(),
                    Ps(),
                ),
                out_specs=qaxes,
                mesh=self.metadata.mesh,
                check_vma=False,
            )
        output = fn(
            q,
            kv_pages,
            cache_metadata.context_lens,
            cache_metadata.pages_tables,
            cache_metadata.query_start_loc,
            num_seqs,
        )
        return AttentionOutput(attention_weights=None, attention_outputs=output)

    def forward_gpu(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagesCacheView,
        cache_metadata: PagesMetadata,
        **ignore,
    ) -> AttentionOutput:
        """
        GPU forward pass.
        """
        kv_pages = cache_view.kv_pages
        manager = self.metadata.partition_manager
        resolve = manager.resolve
        num_seqs = cache_metadata.num_seqs.reshape(-1)
        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=q.shape)
        output = jax.shard_map(
            partial(
                triton_ragged_paged_attention,
                softmax_scale=self.metadata.softmax_scale,
                kv_pages_per_block=32,
            ),
            in_specs=(
                qaxes,
                resolve(axes=[ct.EMPTY, ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=kv_pages.shape),
                Ps(),
                Ps(),
                Ps(),
                Ps(),
            ),
            out_specs=qaxes,
            mesh=self.metadata.mesh,
            check_vma=False,
        )(
            q,
            kv_pages,
            cache_metadata.context_lens,
            cache_metadata.pages_tables,
            cache_metadata.query_start_loc,
            num_seqs,
        )
        return AttentionOutput(attention_weights=None, attention_outputs=output)

    @jax.named_scope("easydel-pagedattn-tpu")
    def forward_tpu(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagesCacheView,
        cache_metadata: PagesMetadata,
        **ignore,
    ) -> AttentionOutput:
        """
        TPU forward pass for Paged Attention.
        """
        kv_pages = cache_view.kv_pages
        manager = self.metadata.partition_manager
        resolve = manager.resolve
        num_seqs = cache_metadata.num_seqs.reshape(-1)
        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=q.shape)
        pages_per_seq = cache_metadata.pages_tables.shape[1]
        num_kv_pages_per_block = min(8, pages_per_seq)

        if num_kv_pages_per_block <= 0 or num_kv_pages_per_block > pages_per_seq:
            raise ValueError(f"num_kv_pages_per_block={num_kv_pages_per_block} must be in range (0, {pages_per_seq}].")
        output = jax.shard_map(
            partial(
                pallas_ragged_paged_attention,
                sm_scale=self.metadata.softmax_scale,
                soft_cap=self.metadata.soft_cap,
                num_kv_pages_per_block=num_kv_pages_per_block,
            ),
            in_specs=(
                qaxes,
                resolve(axes=[ct.EMPTY, ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=kv_pages.shape),
                Ps(),
                Ps(),
                Ps(),
                Ps(),
            ),
            out_specs=qaxes,
            mesh=self.metadata.mesh,
            check_vma=False,
        )(
            q,
            kv_pages,
            cache_metadata.context_lens,
            cache_metadata.pages_tables,
            cache_metadata.query_start_loc,
            num_seqs,
        )
        return AttentionOutput(attention_weights=None, attention_outputs=output)

    def forward_cpu(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagesCacheView,
        cache_metadata: PagesMetadata,
        **ignore,
    ) -> AttentionOutput:
        """
        CUDA GPU forward pass.

        Raises:
            NotImplementedError: Paged Attention currently relies on Pallas for TPUs
                and does not have a specific CUDA implementation. (Future work might add this).
        """
        return self.forward_native(q=q, k=k, v=v, cache_view=cache_view, cache_metadata=cache_metadata, **ignore)

    def forward_cuda(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagesCacheView,
        cache_metadata: PagesMetadata,
        **ignore,
    ) -> AttentionOutput:
        """
        GPU forward pass.

        Raises:
            NotImplementedError: Paged Attention currently relies on Pallas for TPUs
                and does not have a specific ROCm implementation.
        """
        return self.forward_gpu(q=q, k=k, v=v, cache_view=cache_view, cache_metadata=cache_metadata, **ignore)

    def forward_rocm(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagesCacheView,
        cache_metadata: PagesMetadata,
        **ignore,
    ) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_gpu(q=q, k=k, v=v, cache_view=cache_view, cache_metadata=cache_metadata, **ignore)

    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagesCacheView,
        cache_metadata: PagesMetadata,
        **ignore,
    ) -> AttentionOutput:
        """
        Makes the RaggedPageAttn instance callable.

        Preprocesses the query tensor by removing the sequence dimension (which is
        implicit or handled differently in paged attention kernels) before dispatching
        to the appropriate backend implementation via the parent class `__call__`.
        It then restores the sequence dimension to the output.

        Args:
            q (Array): Query tensor. Expected shape [batch, seq_len, num_heads, head_dim].
                The `seq_len` dimension is squeezed before passing to the kernel.
            k (Array): Key tensor (ignored by TPU kernels).
            v (Array): Value tensor (ignored by TPU kernels).
            cache_view (PagesCacheView): Contains the paged KV cache tensors.
            cache_metadata (PagesMetadata): Contains metadata describing the
                state and mode of the current batch.
            **ignore: Ignored keyword arguments.

        Returns:
            AttentionOutput: The result of the attention computation with the sequence
                dimension restored.
        """
        if q.ndim == 4:
            batch, sequence, head, dim = q.shape
        output = super().__call__(
            q=q.reshape(-1, *q.shape[-2:]),
            k=k,
            v=v,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            **ignore,
        )
        if q.ndim == 4:
            output.attention_outputs = output.attention_outputs.reshape(batch, sequence, head, dim)
        return output
