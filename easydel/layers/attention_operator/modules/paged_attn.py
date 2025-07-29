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
from jax.experimental.shard_map import shard_map

from easydel.kernels.cpu_ops import jax_ragged_paged_attention
from easydel.kernels.tpu_ops import pallas_ragged_paged_attention
from easydel.layers.caching import PagesCacheView, PagesMetadata

from .._attention_impl import AttentionImpl, AttentionMetadata, AttentionOutput, AttentionRegistry


@AttentionRegistry.register
class PagedAttn(AttentionImpl):
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
            tp.Union[str, tp.Tuple[str]]: The name "paged_attention".
        """
        return "paged_attention"

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
        kpages = cache_view.key_pages
        vpages = cache_view.value_pages
        manager = self.metadata.partition_manager
        resolve = manager.resolve
        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=q.shape)
        output = shard_map(
            f=partial(
                jax_ragged_paged_attention,
                softmax_scale=self.metadata.softmax_scale,
                soft_cap=self.metadata.soft_cap,
            ),
            in_specs=(
                qaxes,
                resolve(axes=[ct.EMPTY, ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=kpages.shape),
                resolve(axes=[ct.EMPTY, ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=vpages.shape),
                resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=cache_metadata.context_lens.shape),
                resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=cache_metadata.pages_tables.shape),
                resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=cache_metadata.query_start_loc.shape),
                resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=cache_metadata.num_seqs.shape),
            ),
            out_specs=qaxes,
            mesh=self.metadata.mesh,
            check_rep=False,
        )(
            q,
            kpages,
            vpages,
            cache_metadata.context_lens,
            cache_metadata.pages_tables,
            cache_metadata.query_start_loc,
            cache_metadata.num_seqs,
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
        Generic GPU forward pass.

        Raises:
            NotImplementedError: Paged Attention relies on specific kernels (currently
                Pallas for TPU) and does not have a generic GPU implementation.
        """
        return self.forward_native(q, k, v, cache_view, cache_metadata, **ignore)

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

        Raises:
            NotImplementedError: Paged Attention currently relies on Pallas for TPUs
                and does not have a specific implementation.
        """
        kv_pages = cache_view.kv_pages
        manager = self.metadata.partition_manager
        resolve = manager.resolve
        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=q.shape)
        output = shard_map(
            f=partial(
                pallas_ragged_paged_attention,
                sm_scale=self.metadata.softmax_scale,
                soft_cap=self.metadata.soft_cap,
                num_kv_pages_per_block=None,
                num_queries_per_block=None,
                vmem_limit_bytes=None,
            ),
            in_specs=(
                qaxes,
                resolve(axes=[ct.EMPTY, ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=kv_pages.shape),
                resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=cache_metadata.context_lens.shape),
                resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=cache_metadata.pages_tables.shape),
                resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=cache_metadata.query_start_loc.shape),
                resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=cache_metadata.num_seqs.shape),
            ),
            out_specs=qaxes,
            mesh=self.metadata.mesh,
            check_rep=False,
        )(
            q,
            kv_pages,
            cache_metadata.context_lens,
            cache_metadata.pages_tables,
            cache_metadata.query_start_loc,
            cache_metadata.num_seqs,
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
        Makes the PagedAttn instance callable.

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
        batch, sequence, head, dim = q.shape
        output = super().__call__(
            q=q.reshape(-1, head, dim),
            k=k,
            v=v,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            **ignore,
        )
        output.attention_outputs = output.attention_outputs.reshape(batch, sequence, head, dim)
        return output
