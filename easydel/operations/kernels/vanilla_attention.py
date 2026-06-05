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

"""Vanilla (standard) attention implementation for EasyDeL.

This module provides a reference implementation of multi-head attention using
standard JAX operations. It serves as both a baseline for comparison with optimized
implementations and a fallback for platforms where specialized kernels are unavailable.

The vanilla attention implementation:
- Uses standard matrix multiplication and softmax operations
- Supports all standard attention features (masking, bias, dropout)
- Works on all platforms (TPU, GPU, CPU) without specialized kernels
- Provides full attention weights for inspection when needed
- Supports Grouped Query Attention (GQA) and Multi-Query Attention (MQA)

Key characteristics:
- Memory complexity: O(N²) where N is sequence length
- Computation: Uses einsum for efficient batch matrix multiplication
- Flexibility: Supports various mask and bias shapes
- Compatibility: Works with any JAX backend without modification

This implementation is ideal for:
- Debugging and development
- Small sequence lengths where memory is not a constraint
- Platforms without optimized attention kernels
- Cases where attention weights need to be inspected

Example:
    >>> from easydel.layers.attention_operator import OperationMetadata
    >>> from easydel.layers.attention_operator.modules import VanillaAttn
    >>>
    >>> metadata = OperationMetadata(
    ...     runtime_dtype=jnp.float16,
    ...     runtime_softmax_dtype=jnp.float32,  # Higher precision for softmax
    ...     dropout_prob=0.1
    ... )
    >>> vanilla_attn = VanillaAttn(metadata)
    >>> output = vanilla_attn(query, key, value, mask=attention_mask)
    >>> attention_weights = output.attention_weights  # Available for inspection
"""

import math
import typing as tp

import jax
from ejkernel.modules import attention  # pyright: ignore[reportMissingTypeStubs]
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, PRNGKeyArray
from spectrax import common_types, with_sharding_constraint

from easydel.caching import TransformerCacheView

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
)


def _segment_ids_2d(segment_ids: Array | None) -> Array | None:
    if segment_ids is None:
        return None
    if segment_ids.ndim == 3:
        return segment_ids[:, 0, :]
    return segment_ids


def _match_sequence_length(segment_ids: Array, target_length: int) -> Array:
    if segment_ids.shape[-1] == target_length:
        return segment_ids
    segment_ids = segment_ids[..., :target_length]
    if segment_ids.shape[-1] == target_length:
        return segment_ids
    pad = jnp.full(
        (*segment_ids.shape[:-1], target_length - segment_ids.shape[-1]),
        -1,
        dtype=segment_ids.dtype,
    )
    return jnp.concatenate([segment_ids, pad], axis=-1)


def _positions_or_range(positions: Array | None, batch_size: int, seq_len: int) -> Array:
    if positions is not None:
        positions = positions[..., :seq_len]
        if positions.shape[-1] == seq_len:
            return positions
    return jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))


@OperationRegistry.register
class VanillaAttn(OperationImpl):
    """
    A standard, non-optimized implementation of multi-head attention.

    This implementation uses basic JAX operations like `jnp.einsum` and standard
    softmax. It serves as a reference implementation and a fallback for platforms
    where optimized kernels (like Flash Attention) are not available or desired.
    It supports features like attention bias, masking, dropout, and Grouped Query
    Attention (GQA)/Multi-Query Attention (MQA) via reshaping.

    Registered under the name "vanilla".
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name of this attention implementation.

        Returns:
            The string "vanilla".
        """
        return "vanilla"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for VanillaAttn.

        Vanilla attention requires basic metadata and uses TransformerCacheView
        for KV-cache management.
        """
        return OperationRequirements.create(
            name="vanilla",
            required_metadata=MetadataField.basic(),
            supported_cache=CacheType.TRANSFORMER | CacheType.HYBRID,
            cache_view_class=TransformerCacheView,
        )

    @jax.named_scope("easydel-vanillaimpl-native-xla")
    def forward_native(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_len num_kv_heads head_dim"],
        mask_info: MaskInfo | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        init_bias: tp.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
        deterministic: bool = True,
        dropout_rng: PRNGKeyArray | None = None,
        softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        dropout_prob: float = 0.0,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Standard multi-head attention implementation using basic JAX operations.

        Args:
            query: Query tensor [batch, seq_len, num_q_heads, head_dim].
            key: Key tensor [batch, kv_len, num_kv_heads, head_dim].
            value: Value tensor [batch, kv_len, num_kv_heads, head_dim].
            mask_info: Optional mask information for attention.
            bias: Optional attention bias [batch, num_heads, seq_len, kv_len].
            init_bias: Optional callable to initialize bias if mask_info and bias are None.
            deterministic: If True, disables dropout.
            dropout_rng: JAX PRNG key for dropout.
            softmax_aux: Auxiliary softmax tensor (e.g., for sink tokens).
            softmax_scale: Scaling factor for attention logits.
            logits_soft_cap: Soft capping value for attention logits.
            dropout_prob: Dropout probability.
            causal: Apply causal masking.
            sliding_window: Sliding window size for local attention.
            **ignore: Additional ignored arguments.

        Returns:
            AttentionOutput containing attention outputs and weights.
        """
        mesh = self.metadata.mesh
        if mesh is None:
            raise ValueError("VanillaAttn requires a mesh to be set on metadata")
        with mesh:
            model_mode = self.get_mode(query=query, BTHD=True)
            shardings = self.metadata.get_shardings(model_mode, layout="bthd")

            # Initialize bias if needed
            needs_bias_init: bool = mask_info is None and bias is None and init_bias is not None
            bias_computed: Float[Array, "batch num_heads seq_len kv_len"] | None
            if needs_bias_init:
                bias_computed = init_bias()
            else:
                bias_computed = bias

            # Apply sharding constraints to inputs
            query = with_sharding_constraint(arr=query, sharding=shardings.query, mesh=mesh)
            key = with_sharding_constraint(arr=key, sharding=shardings.key, mesh=mesh)
            value = with_sharding_constraint(arr=value, sharding=shardings.value, mesh=mesh)

            bias: Float[Array, "batch num_heads seq_len kv_len"] | None
            if bias_computed is not None:
                bias = with_sharding_constraint(arr=bias_computed, sharding=shardings.bias, mesh=mesh)
            else:
                bias = None

            runtime_dtype: jnp.dtype = self.metadata.runtime_dtype
            softmax_dtype: jnp.dtype | None = self.metadata.runtime_softmax_dtype
            is_decode_mode: bool = model_mode == common_types.MODE_DECODE
            causal_computed: bool = causal if not is_decode_mode else False
            if self._can_use_segmented_attention(mask_info):
                outputs, weights = self._forward_segmented(
                    query=query,
                    key=key,
                    value=value,
                    mask_info=tp.cast("MaskInfo", mask_info),
                    bias=bias,
                    init_bias=init_bias,
                    deterministic=deterministic,
                    dropout_rng=dropout_rng,
                    softmax_aux=softmax_aux,
                    softmax_scale=softmax_scale,
                    logits_soft_cap=logits_soft_cap,
                    dropout_prob=dropout_prob,
                    causal=causal_computed,
                    sliding_window=sliding_window,
                    runtime_dtype=runtime_dtype,
                    softmax_dtype=softmax_dtype,
                )
                outputs_sharded = with_sharding_constraint(arr=outputs, sharding=shardings.output, mesh=mesh)
                return AttentionOutput(attention_weights=weights, attention_outputs=outputs_sharded)

            if mask_info is not None:
                attention_mask = mask_info.attention_mask
                if attention_mask is not None and attention_mask.ndim == 4:
                    q_len = query.shape[1]
                    kv_len = key.shape[1]
                    if attention_mask.shape[-2] != q_len or attention_mask.shape[-1] != kv_len:
                        attention_mask = attention_mask[..., :q_len, :kv_len]
                        mask_info = MaskInfo(_attention_mask=attention_mask)
            attn_result = attention(
                query,
                key,
                value,
                bias,
                dropout_rng,
                softmax_aux,
                mask_info=mask_info,
                deterministic=deterministic,
                dropout_prob=dropout_prob,
                dtype=runtime_dtype,
                sliding_window=sliding_window,
                softmax_dtype=softmax_dtype,
                softmax_scale=softmax_scale,
                init_bias=None,
                causal=causal_computed,
                logits_soft_cap=logits_soft_cap,
            )
            if isinstance(attn_result, tuple):
                outputs, weights = attn_result
            else:
                outputs, weights = attn_result, None

            # Apply output sharding
            outputs_sharded = with_sharding_constraint(arr=outputs, sharding=shardings.output, mesh=mesh)
            return AttentionOutput(attention_weights=weights, attention_outputs=outputs_sharded)

    @staticmethod
    def _can_use_segmented_attention(mask_info: MaskInfo | None) -> bool:
        if mask_info is None:
            return False
        if getattr(mask_info, "_attention_mask", None) is not None:
            return False
        return getattr(mask_info, "_q_segment_ids", None) is not None or getattr(mask_info, "_kv_segment_ids", None) is not None

    def _forward_segmented(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_len num_kv_heads vhead_dim"],
        mask_info: MaskInfo,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None,
        init_bias: tp.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None,
        deterministic: bool,
        dropout_rng: PRNGKeyArray | None,
        softmax_aux: Array | None,
        softmax_scale: float | None,
        logits_soft_cap: float | None,
        dropout_prob: float,
        causal: bool,
        sliding_window: int | tuple[int, int] | None,
        runtime_dtype: jnp.dtype,
        softmax_dtype: jnp.dtype | None,
    ) -> tuple[Array, Array]:
        batch_size, q_len, num_q_heads, head_dim = query.shape
        kv_len = key.shape[1]
        num_kv_heads = key.shape[2]
        if num_kv_heads != num_q_heads:
            if num_q_heads % num_kv_heads != 0:
                raise ValueError(
                    f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for vanilla GQA."
                )
            key, value = self.repeat_kv_heads(key, value, num_q_heads // num_kv_heads)

        compute_dtype = softmax_dtype or runtime_dtype or query.dtype
        scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        logits = jnp.einsum(
            "bqhd,bkhd->bhqk",
            query.astype(compute_dtype),
            key.astype(compute_dtype),
            preferred_element_type=compute_dtype,
        )
        logits = logits * jnp.asarray(scale, dtype=compute_dtype)
        if logits_soft_cap is not None:
            cap = jnp.asarray(logits_soft_cap, dtype=compute_dtype)
            logits = cap * jnp.tanh(logits / cap)

        # In the standard attention stack this closure materializes
        # ``mask_info.create_bias()``, which expands packed segment ids to a
        # dense mask. Segment/causal/sliding masks are applied below directly.
        if bias is None and init_bias is not None and not self._can_use_segmented_attention(mask_info):
            bias = init_bias()
        if bias is not None:
            logits = logits + bias.astype(logits.dtype)

        q_segment_ids = _segment_ids_2d(getattr(mask_info, "_q_segment_ids", None))
        kv_segment_ids = _segment_ids_2d(getattr(mask_info, "_kv_segment_ids", None))
        if q_segment_ids is None and kv_segment_ids is None:
            raise ValueError("Segmented vanilla attention requires q_segment_ids or kv_segment_ids.")
        if q_segment_ids is None:
            q_segment_ids = _match_sequence_length(tp.cast("Array", kv_segment_ids), q_len)
        if kv_segment_ids is None:
            kv_segment_ids = q_segment_ids

        q_segment_ids = _match_sequence_length(q_segment_ids.astype(jnp.int32), q_len)
        kv_segment_ids = _match_sequence_length(kv_segment_ids.astype(jnp.int32), kv_len)
        valid = (
            (q_segment_ids[:, :, None] >= 0)
            & (kv_segment_ids[:, None, :] >= 0)
            & (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :])
        )

        q_positions = _positions_or_range(mask_info.q_positions, batch_size, q_len)
        kv_positions = _positions_or_range(mask_info.kv_positions, batch_size, kv_len)
        if causal and not mask_info.causal_mask_baked_in:
            valid = valid & (kv_positions[:, None, :] <= q_positions[:, :, None])
        if sliding_window is not None and not mask_info.sliding_window_baked_in:
            if isinstance(sliding_window, tuple):
                left_window, right_window = sliding_window
            else:
                left_window = right_window = sliding_window
            valid = valid & (kv_positions[:, None, :] >= q_positions[:, :, None] - int(left_window))
            valid = valid & (kv_positions[:, None, :] <= q_positions[:, :, None] + int(right_window))

        mask_value = jnp.finfo(logits.dtype).min
        logits = jnp.where(valid[:, None, :, :], logits, mask_value)

        if softmax_aux is not None:
            aux = jnp.asarray(softmax_aux, dtype=logits.dtype)
            if aux.ndim == 1:
                aux = jnp.broadcast_to(aux[None, :], (num_q_heads, aux.shape[0]))
            aux = jnp.broadcast_to(aux[None, :, None, :], (batch_size, num_q_heads, q_len, aux.shape[-1]))
            weights = jax.nn.softmax(jnp.concatenate([logits, aux], axis=-1), axis=-1)[..., :kv_len]
        else:
            weights = jax.nn.softmax(logits, axis=-1)

        if dropout_prob > 0.0 and not deterministic:
            if dropout_rng is None:
                raise ValueError("dropout_rng must be provided when deterministic=False and dropout_prob > 0.")
            keep_prob = 1.0 - dropout_prob
            keep = jr.bernoulli(dropout_rng, keep_prob, weights.shape)
            weights = jnp.where(keep, weights / keep_prob, 0)

        outputs = jnp.einsum(
            "bhqk,bkhd->bqhd",
            weights.astype(runtime_dtype),
            value.astype(runtime_dtype),
            preferred_element_type=runtime_dtype,
        )
        return outputs.astype(runtime_dtype), weights

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Delegates to `forward_native`.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU forward pass. Delegates to `forward_native`.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass. Delegates to `forward_native`.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA GPU forward pass. Delegates to `forward_native`.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Delegates to `forward_native`.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_len num_kv_heads head_dim"],
        mask_info: MaskInfo | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        init_bias: tp.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
        deterministic: bool = True,
        dropout_rng: PRNGKeyArray | None = None,
        softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        dropout_prob: float = 0.0,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Executes the vanilla attention computation.

        Calls the appropriate backend-specific forward method via `super().__call__`.
        Since all backend methods delegate to `forward_native`, this effectively
        always runs the native JAX implementation.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Optional attention mask.
            bias: Optional attention bias.
            init_bias: Optional callable to initialize bias.
            deterministic: If True, disables dropout.
            dropout_rng: JAX PRNG key for dropout if deterministic is False.
            **ignore: Additional ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return super().__call__(
            query=query,
            key=key,
            value=value,
            mask_info=mask_info,
            bias=bias,
            deterministic=deterministic,
            dropout_prob=dropout_prob,
            dropout_rng=dropout_rng,
            sliding_window=sliding_window,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            init_bias=init_bias,
            logits_soft_cap=logits_soft_cap,
            causal=causal,
            **ignore,
        )


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    # Test cace when qkv might refer to mla
    b, qs, ks, qh, kh, d, vd = 1, 1024, 1024, 32, 8, 128, 128 + 64
    query = jr.normal(jr.key(0), (b, qs, qh, d), "f2")
    key = jr.normal(jr.key(1), (b, ks, kh, d), "f2")
    value = jr.normal(jr.key(2), (b, ks, kh, vd), "f2")
    mask_info = MaskInfo.from_random(b, qs, ks)

    metadata = OperationMetadata(
        runtime_dtype=jnp.float16,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )

    out = VanillaAttn(metadata)(query=query, key=key, value=value, mask_info=mask_info)
    print(out.attention_outputs)
