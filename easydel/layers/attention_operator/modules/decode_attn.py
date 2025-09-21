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
    >>> from easydel.layers.attention_operator import AttentionMetadata
    >>> from easydel.layers.attention_operator.modules import AutoRegressiveDecodeAttn
    >>> from easydel.layers.caching.transformer import TransformerMetadata
    >>>
    >>> # Configure for decoding
    >>> metadata = AttentionMetadata(
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

from functools import partial

import jax
from eformer import common_types
from eformer import escale as es
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.experimental import shard_map
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Float, Int

from easydel.kernels.gpu_ops import pallas_ragged_decode as gpu_pallas_ragged_decode
from easydel.kernels.tpu_ops import pallas_ragged_decode as tpu_pallas_ragged_decode
from easydel.layers.caching.transformer import TransformerMetadata

from .._attention_impl import AttentionImpl, AttentionMetadata, AttentionOutput, AttentionRegistry

shard_map = shard_map.shard_map


@AttentionRegistry.register
class AutoRegressiveDecodeAttn(AttentionImpl):
    """
    Attention implementation tailored for the autoregressive decoding step.

    This class handles the attention mechanism when generating tokens one by one,
    attending to the previously generated sequence stored in a cache. It utilizes
    `shard_map` for distributed computation and supports different backends,
    including a potential Pallas-optimized version for TPUs. It assumes the
    query sequence length is 1.

    Attributes:
        metadata (AttentionMetadata): Configuration metadata for the attention mechanism.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name of this attention implementation.

        Returns:
            The string "autoregressive_decodeattn".
        """
        return "autoregressive_decodeattn"

    def get_impl_metadata(self) -> AttentionMetadata:
        """
        Returns the metadata associated with this attention implementation instance.

        Returns:
            The `AttentionMetadata` provided during initialization.
        """
        return self.metadata

    @jax.named_scope("easydel-autoregressive_decodeattn-native-xla")
    def forward_native(
        self,
        q: Float[Array, "batch 1 num_q_heads head_dim"],
        k: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        v: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        cache_metadata: TransformerMetadata,
        **ignores,
    ) -> AttentionOutput:
        """
        Performs the native JAX/XLA forward pass for autoregressive decoding attention.

        This implementation uses `shard_map` to distribute the computation and relies
        on standard JAX operations (`einsum`, `softmax`). It calculates attention weights
        between the single query token and the keys in the cache, applies masking
        based on the valid range defined in `cache_metadata`, computes the softmax,
        and finally computes the weighted sum of values.

        Args:
            q: Query tensor with shape [batch, 1, num_q_heads, head_dim].
                Single query token for autoregressive generation.
            k: Key tensor from cache with shape [batch, kv_seq_len, num_kv_heads, head_dim].
                Contains all previous keys in the sequence.
            v: Value tensor from cache with shape [batch, kv_seq_len, num_kv_heads, head_dim].
                Contains all previous values in the sequence.
            cache_metadata: Metadata about the cache containing:
                - starts: Start indices for valid keys/values
                - indexs: Current indices or sequence lengths
            **ignores: Additional keyword arguments that are ignored.

        Returns:
            AttentionOutput containing:
                - attention_outputs: Float[Array, "batch 1 num_q_heads head_dim"]
                  The attended representation for the current query token.
                - attention_weights: None (not computed for efficiency)
        """
        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5

        model_mode = self.get_mode(q=q, BTHD=True)

        assert model_mode == common_types.MODE_DECODE

        q_sharding, k_sharding, v_sharding, *_, a_sharding = self.metadata.get_shardings(model_mode)

        q = es.with_sharding_constraint(q, q_sharding)
        k = es.with_sharding_constraint(k, k_sharding)
        v = es.with_sharding_constraint(v, v_sharding)
        q = q.squeeze(1)

        def _compute(
            q: Float[Array, "batch num_q_heads head_dim"],
            k: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
            v: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
            start: Int[Array, "batch 1"],
            index: Int[Array, "batch 1"],
        ) -> Float[Array, "batch num_q_heads head_dim"]:
            qb, qhead, qdim = q.shape
            _, kvlen, kvhead, _kvdim = k.shape
            repeats = qhead // kvhead
            q = q.reshape(qb, kvhead, repeats, qdim)
            weight = jnp.einsum("bkhd,bmkd->bkhm", q * sm_scale, k)
            ranges = jnp.arange(kvlen).reshape(1, -1)
            mask = (start <= ranges) & (ranges < index)
            weight = jnp.where(mask[:, None, None, :], weight, jnp.finfo(weight.dtype).min)
            weight = jax.nn.softmax(weight, axis=-1)
            return jnp.einsum("bkhm,bmkd->bkhd", weight, v).reshape(qb, qhead, qdim)

        attn_output = _compute(q, k, v, cache_metadata.starts.reshape(-1, 1), cache_metadata.indexs.reshape(-1, 1))
        attn_output = jnp.expand_dims(attn_output, 1)
        return AttentionOutput(
            attention_weights=None,
            attention_outputs=es.with_sharding_constraint(attn_output, a_sharding),
        )

    @jax.named_scope("easydel-autoregressive_decodeattn-gpu-triton")
    def forward_gpu(
        self,
        q: Float[Array, "batch 1 num_q_heads head_dim"],
        k: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        v: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        cache_metadata: TransformerMetadata,
        **ignores,
    ) -> AttentionOutput:
        """
        GPU forward pass for autoregressive decoding attention using Triton kernels.

        Utilizes GPU-optimized Pallas ragged decode kernel for efficient
        attention computation during the decoding phase. The kernel handles
        variable sequence lengths efficiently.

        Args:
            q: Query tensor with shape [batch, 1, num_q_heads, head_dim].
                Single query token for current generation step.
            k: Key tensor from cache with shape [batch, kv_seq_len, num_kv_heads, head_dim].
                Contains all previous keys up to max cache size.
            v: Value tensor from cache with shape [batch, kv_seq_len, num_kv_heads, head_dim].
                Contains all previous values up to max cache size.
            cache_metadata: Metadata about the cache containing:
                - starts: Start indices for valid cache entries per batch element
                - indexs: Current sequence lengths per batch element
            **ignores: Additional keyword arguments that are ignored.

        Returns:
            AttentionOutput containing:
                - attention_outputs: Float[Array, "batch 1 num_q_heads head_dim"]
                  The attended representation for the current token.
                - attention_weights: None (not computed for efficiency)
        """
        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        model_mode = self.get_mode(q=q, BTHD=True)
        assert model_mode == common_types.MODE_DECODE
        q_sharding, k_sharding, v_sharding, *_, a_sharding = self.metadata.get_shardings(model_mode, True)
        views_sharding = Ps(q_sharding[0])

        starts = cache_metadata.starts.reshape(-1, 1)
        indexs = cache_metadata.indexs.reshape(-1, 1)

        @partial(
            shard_map,
            mesh=self.metadata.mesh,
            in_specs=(
                self.create_stable_sharding(q_sharding, dep=q, tensor=q, preserved_indices=[0, 2]),
                self.create_stable_sharding(k_sharding, dep=k, tensor=k, preserved_indices=[0, 2]),
                self.create_stable_sharding(v_sharding, dep=v, tensor=v, preserved_indices=[0, 2]),
                self.create_stable_sharding(views_sharding, dep=starts, tensor=starts, preserved_indices=[0]),
                self.create_stable_sharding(views_sharding, dep=indexs, tensor=indexs, preserved_indices=[0]),
            ),
            out_specs=self.create_stable_sharding(a_sharding, tensor=q, preserved_indices=[0, 2]),
            check_rep=False,
        )
        def _compute(
            q: Float[Array, "batch 1 num_q_heads head_dim"],
            k: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
            v: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
            start: Int[Array, "batch 1"],
            index: Int[Array, "batch 1"],
        ) -> Float[Array, "batch 1 num_q_heads head_dim"]:
            out = gpu_pallas_ragged_decode(
                query_tensor=q.squeeze(1),
                key_tensor=k,
                value_tensor=v,
                sequence_start=start.reshape(-1),
                sequence_end=index.reshape(-1),
                softmax_scale=sm_scale,
            )
            out = jnp.expand_dims(out, 1)
            return out

        attn_output = _compute(q, k, v, starts, indexs)

        return AttentionOutput(attention_weights=None, attention_outputs=attn_output)

    def forward_tpu(
        self,
        q: Float[Array, "batch 1 num_q_heads head_dim"],
        k: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        v: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        cache_metadata: TransformerMetadata,
        **ignores,
    ) -> AttentionOutput:
        """
        TPU-specific forward pass using the Pallas ragged decode kernel.

        This method is intended for TPU execution and leverages the potentially
        optimized `tpu_pallas_ragged_decode` kernel for better performance compared
        to the native `einsum`-based implementation.

        Args:
            q (Array): Query tensor of shape (batch_size, 1, num_query_heads, head_dim).
            k (Array): Key tensor (from cache) of shape (batch_size, kv_sequence_length, num_kv_heads, head_dim).
            v (Array): Value tensor (from cache) of shape (batch_size, kv_sequence_length, num_kv_heads, head_dim).
            cache_metadata (TransformerMetadata): Contains cache metadata (`starts`, `index`).
            **ignores: Ignored keyword arguments.

        Returns:
            AttentionOutput: An object containing the attention outputs (`attention_outputs`)
                calculated by the Pallas kernel. Attention weights are not returned.
        """
        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        model_mode = self.get_mode(q=q, BTHD=True)
        assert model_mode == common_types.MODE_DECODE
        q_sharding, k_sharding, v_sharding, *_, a_sharding = self.metadata.get_shardings(model_mode, True)
        views_sharding = Ps(q_sharding[0])

        starts = cache_metadata.starts.reshape(-1, 1)
        indexs = cache_metadata.indexs.reshape(-1, 1)

        @partial(
            shard_map,
            mesh=self.metadata.mesh,
            in_specs=(
                self.create_stable_sharding(q_sharding, dep=q, tensor=q, preserved_indices=[0, 2]),
                self.create_stable_sharding(k_sharding, dep=k, tensor=k, preserved_indices=[0, 2]),
                self.create_stable_sharding(v_sharding, dep=v, tensor=v, preserved_indices=[0, 2]),
                self.create_stable_sharding(views_sharding, dep=starts, tensor=starts, preserved_indices=[0]),
                self.create_stable_sharding(views_sharding, dep=indexs, tensor=indexs, preserved_indices=[0]),
            ),
            out_specs=self.create_stable_sharding(a_sharding, tensor=q, preserved_indices=[0, 2]),
            check_rep=False,
        )
        def _compute(
            q: Float[Array, "batch 1 num_q_heads head_dim"],
            k: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
            v: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
            start: Int[Array, "batch 1"],
            index: Int[Array, "batch 1"],
        ) -> Float[Array, "batch 1 num_q_heads head_dim"]:
            return tpu_pallas_ragged_decode(
                query_tensor=q,
                key_tensor=k,
                value_tensor=v,
                sequence_start=start.reshape(-1),
                sequence_end=index.reshape(-1),
                softmax_scale=sm_scale,
                block_size=512,
            )

        attn_output = _compute(q, k, v, starts, indexs)

        return AttentionOutput(attention_weights=None, attention_outputs=attn_output)

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
        q: Float[Array, "batch 1 num_q_heads head_dim"],
        k: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        v: Float[Array, "batch kv_seq_len num_kv_heads head_dim"],
        cache_metadata: TransformerMetadata,
        **ignores,
    ) -> AttentionOutput:
        """
        Makes the class instance callable.

        This method routes the call to the appropriate backend-specific forward method
        (e.g., `forward_tpu`, `forward_gpu`) based on the configuration determined
        by the parent class `AttentionImpl`. It passes the necessary arguments:
        query, key, value, and cache view information.

        Args:
            q (Array): Query tensor.
            k (Array): Key tensor (from cache).
            v (Array): Value tensor (from cache).
            cache_metadata (TransformerMetadata): Cache metadata.
            **kwargs: Additional keyword arguments passed to the underlying forward method.

        Returns:
            AttentionOutput: The result of the attention computation.
        """
        return super().__call__(
            q=q,
            k=k,
            v=v,
            cache_metadata=cache_metadata,
            **ignores,
        )


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    # Test cace when qkv might refer to mla
    b, qs, ks, qh, kh, d, vd = 1, 1024, 1024, 32, 8, 128, 128 + 64
    q = jr.normal(jr.key(0), (b, qs, qh, d), "f2")
    k = jr.normal(jr.key(1), (b, ks, kh, d), "f2")
    v = jr.normal(jr.key(2), (b, ks, kh, vd), "f2")
    a = jnp.astype(jr.randint(jr.key(3), (b, 1, qs, ks), 0, 4) > 2, "b1")

    metadata = AttentionMetadata(
        runtime_dtype=jnp.float16,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
        # backend="cpu",
    )

    attn = AutoRegressiveDecodeAttn(metadata)
    out = attn(q=q, k=k, v=v, mask=a)
