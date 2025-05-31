# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from eformer import common_types
from eformer import escale as es
from jax import Array
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as Ps

from easydel.kernels.tpu_ops.paged_attention_pallas import pallas_chunked_prefill_attention, pallas_paged_attention
from easydel.layers.caching import PagedAttentionCacheView
from easydel.layers.caching.paged_attention.paged_attention_cache import PagedAttentionMetadata

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
            additional `PagedAttentionMetadata` passed during the forward call for
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

    def forward_native(self, *args, **kwargs) -> AttentionOutput:
        """
        Native (CPU) forward pass.

        Raises:
            NotImplementedError: Paged Attention requires specialized kernels and
                does not have a native CPU implementation.
        """
        raise NotImplementedError("Paged Attention does not have a native CPU implementation.")

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """
        Generic GPU forward pass.

        Raises:
            NotImplementedError: Paged Attention relies on specific kernels (currently
                Pallas for TPU) and does not have a generic GPU implementation.
        """
        raise NotImplementedError("Paged Attention does not have a generic GPU implementation.")

    def _prefill_tpu(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagedAttentionCacheView,
        cache_metadata: PagedAttentionMetadata,
    ):
        """
        Internal TPU implementation for the prefill phase using Pallas.

        Args:
            q (Array): Query tensor for the prefill tokens. Shape typically
                (num_prefill_tokens, num_attn_heads_per_device, head_dim).
            k (Array): Key tensor (ignored, uses `cache_view.key_pages`).
            v (Array): Value tensor (ignored, uses `cache_view.value_pages`).
            cache_view (PagedAttentionCacheView): Contains the paged KV cache tensors
                (`key_pages`, `value_pages`).
            cache_metadata (PagedAttentionMetadata): Contains metadata specific to the
                paged attention state, including `prefill_length` and `prefill_page_table`.

        Returns:
            Array: The attention output for the prefill tokens.
        """
        kv_pages_sharding = getattr(cache_view.key_pages, "sharding", None)
        kv_pages_sharding = getattr(kv_pages_sharding, "spec", kv_pages_sharding)
        if kv_pages_sharding is None:
            kv_pages_sharding = self.metadata.partition_manager.resolve(
                [
                    common_types.HEAD,
                    common_types.EMPTY,
                    common_types.EMPTY,
                    common_types.EMPTY,
                ],
                mode=common_types.MODE_PREFILL,
                shape=cache_view.value_pages.shape,
            )
        return shard_map(
            partial(
                pallas_chunked_prefill_attention,
                sm_scale=self.metadata.softmax_scale,
            ),
            mesh=self.metadata.mesh,
            in_specs=(
                Ps(None, kv_pages_sharding[0], None),
                kv_pages_sharding,
                kv_pages_sharding,
                Ps(),
                Ps(),
            ),
            out_specs=Ps(None, kv_pages_sharding[0], None),
            check_rep=False,
        )(
            q,
            cache_view.key_pages,
            cache_view.value_pages,
            cache_metadata.prefill_length,
            cache_metadata.prefill_page_table,
        ).reshape(q.shape)

    def _decode_tpu(
        self,
        q: Array,
        k: Array,
        v: Array,  # lol
        cache_view: PagedAttentionCacheView,
        cache_metadata: PagedAttentionMetadata,
    ):
        """
        Internal TPU implementation for the decode phase using Pallas.

        Args:
            q (Array): Query tensor for the decode tokens (typically one token per sequence).
                Shape typically (num_decode_sequences, num_attn_heads_per_device, head_dim).
            k (Array): Key tensor (ignored, uses `cache_view.key_pages`).
            v (Array): Value tensor (ignored, uses `cache_view.value_pages`).
            cache_view (PagedAttentionCacheView): Contains the paged KV cache tensors.
            cache_metadata (PagedAttentionMetadata): Contains metadata specific to the
                paged attention state, including `decodes_position` and `decodes_page_table`.

        Returns:
            Array: The attention output for the decode tokens.
        """
        kv_pages_sharding = getattr(cache_view.key_pages, "sharding", None)
        kv_pages_sharding = getattr(kv_pages_sharding, "spec", kv_pages_sharding)
        if kv_pages_sharding is None:
            kv_pages_sharding = self.metadata.partition_manager.resolve(
                [
                    common_types.HEAD,
                    common_types.EMPTY,
                    common_types.EMPTY,
                    common_types.EMPTY,
                ],
                mode=common_types.MODE_PREFILL,
                shape=cache_view.value_pages.shape,
            )

        return shard_map(
            partial(
                pallas_paged_attention,
                pages_per_compute_block=8,
                sm_scale=self.metadata.softmax_scale,
            ),
            mesh=es.get_incontext_mesh(),
            in_specs=(
                Ps(None, kv_pages_sharding[0], None),
                kv_pages_sharding,
                kv_pages_sharding,
                Ps(None),
                Ps(None),
            ),
            out_specs=Ps(None, kv_pages_sharding[0], None),
            check_rep=False,
        )(
            q,
            cache_view.key_pages,
            cache_view.value_pages,
            cache_metadata.decodes_position + 1,
            cache_metadata.decodes_page_table,
        ).reshape(q.shape)

    def _mixed_tpu(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagedAttentionCacheView,
        cache_metadata: PagedAttentionMetadata,
    ):
        """
        Internal TPU implementation for mixed prefill and decode batches.

        This method handles batches containing both prefill and decode operations
        by slicing the input query tensor, calling the respective Pallas kernels
        (`_prefill_tpu`, `_decode_tpu`), and then combining the results.

        Args:
            q (Array): Combined query tensor for both prefill and decode tokens.
                Shape (total_tokens, num_attn_heads_per_device, head_dim).
            k (Array): Key tensor (ignored).
            v (Array): Value tensor (ignored).
            cache_view (PagedAttentionCacheView): Contains the paged KV cache tensors.
            cache_metadata (PagedAttentionMetadata): Contains combined metadata for
                prefill and decode parts of the batch.

        Returns:
            Array: The combined attention output for the entire batch.
        """
        total_len, num_attn_heads_per_device, head_dim = q.shape
        output = jnp.zeros(shape=(total_len, num_attn_heads_per_device, head_dim), dtype=q.dtype)
        padded_prompt_length = cache_metadata.prefill_position.shape[0]

        cache_view = cache_view.write_prefill_to_cache(
            k[:padded_prompt_length, :, :],
            v[:padded_prompt_length, :, :],
            cache_metadata,
        )
        prefill_output = self._prefill_tpu(
            q=q[:padded_prompt_length, :, :],
            k=k[:padded_prompt_length, :, :],
            v=v[:padded_prompt_length, :, :],
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        cache_view = cache_view.write_decodes_to_cache(
            k[padded_prompt_length:, :, :],
            v[padded_prompt_length:, :, :],
            cache_metadata,
        )
        decodes_output = self._decode_tpu(
            q=q[padded_prompt_length:, :, :],
            k=k[padded_prompt_length:, :, :],
            v=v[padded_prompt_length:, :, :],
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        output = jax.lax.dynamic_update_slice_in_dim(
            output,
            prefill_output,
            start_index=0,
            axis=0,
        )

        output = jax.lax.dynamic_update_slice_in_dim(
            output,
            decodes_output,
            start_index=padded_prompt_length,
            axis=0,
        )
        return output

    @jax.named_scope("easydel-pagedattn-tpu")
    def forward_tpu(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagedAttentionCacheView,
        cache_metadata: PagedAttentionMetadata,
        **ignore,
    ) -> AttentionOutput:
        """
        TPU forward pass for Paged Attention.

        Determines the execution mode (prefill, decode, or mixed) based on the
        provided `cache_metadata` and dispatches the computation to the corresponding
        internal TPU method (`_prefill_tpu`, `_decode_tpu`, `_mixed_tpu`).

        Args:
            q (Array): Query tensor. Shape depends on mode (prefill/decode/mixed).
            k (Array): Key tensor (ignored).
            v (Array): Value tensor (ignored).
            cache_view (PagedAttentionCacheView): Contains the paged KV cache tensors.
            cache_metadata (PagedAttentionMetadata): Contains metadata describing the
                state and mode (prefill/decode/mixed) of the current batch.
            **ignore: Ignored keyword arguments.

        Returns:
            AttentionOutput: An object containing the computed attention outputs.
                Attention weights are typically not computed or returned in paged attention.
        """
        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5

        if cache_metadata.is_prefill_mode():
            out = self._prefill_tpu(
                q=q,
                k=k,
                v=v,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
        elif cache_metadata.is_decode_mode():
            out = self._decode_tpu(
                q=q,
                k=k,
                v=v,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
        else:
            out = self._mixed_tpu(
                q=q,
                k=k,
                v=v,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
        return AttentionOutput(
            attention_weights=None,
            attention_outputs=out,
        )

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """
        CUDA GPU forward pass.

        Raises:
            NotImplementedError: Paged Attention currently relies on Pallas for TPUs
                and does not have a specific CUDA implementation. (Future work might add this).
        """
        raise NotImplementedError("Paged Attention does not have a CPU implementation.")

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """
        ROCm GPU forward pass.

        Raises:
            NotImplementedError: Paged Attention currently relies on Pallas for TPUs
                and does not have a specific ROCm implementation.
        """
        raise NotImplementedError("Paged Attention does not have a CUDA implementation.")

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        raise NotImplementedError("Paged Attention does not have a ROCm implementation.")

    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
        cache_view: PagedAttentionCacheView,
        cache_metadata: PagedAttentionMetadata,
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
            cache_view (PagedAttentionCacheView): Contains the paged KV cache tensors.
            cache_metadata (PagedAttentionMetadata): Contains metadata describing the
                state and mode of the current batch.
            **ignore: Ignored keyword arguments.

        Returns:
            AttentionOutput: The result of the attention computation with the sequence
                dimension restored.
        """
        if cache_metadata.is_prefill_mode():
            sq_axis = 0
        else:
            sq_axis = 1

        q = q.squeeze(sq_axis)
        k = k.squeeze(sq_axis)
        v = v.squeeze(sq_axis)

        output = super().__call__(  # let use autoswitch ill add gpu kernels later.
            q=q,
            k=k,
            v=v,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            **ignore,
        )
        output.attention_outputs = jnp.expand_dims(output.attention_outputs, sq_axis)
        return output
