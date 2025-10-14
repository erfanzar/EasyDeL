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

"""Splash Attention implementation for TPU acceleration.

This module provides the Splash Attention implementation, a TPU-optimized
attention mechanism that leverages the Pallas framework for maximum performance.
Splash Attention is specifically designed to take advantage of TPU's matrix
multiplication units and memory hierarchy.

Key features:
- TPU-specific optimization using Pallas kernels
- Support for Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
- Efficient handling of causal masks
- Automatic fallback to vanilla attention for unsupported configurations
- Optimized for sequences with lengths divisible by 128

Implementation details:
- Uses `make_splash_mqa_single_device` primitive from JAX experimental
- Requires specific block sizes for optimal TPU utilization
- Falls back to vanilla attention for:
  * Single token generation (seq_len = 1)
  * Non-causal attention patterns
  * Sequences not divisible by 128

Example:
    >>> from easydel.layers.attention_operator import OperationMetadata
    >>> from easydel.layers.attention_operator.modules import SplashAttn
    >>>
    >>> # Configure for TPU execution
    >>> metadata = OperationMetadata(
    ...     runtime_dtype=jnp.bfloat16,
    ...     softmax_scale=1.0 / math.sqrt(head_dim),
    ...     blocksize_q=256,
    ...     blocksize_k=512
    ... )
    >>> splash_attn = SplashAttn(metadata)
    >>>
    >>> # Use with sequences divisible by 128
    >>> output = splash_attn(query, key, value, causal=True)

Note:
    Splash Attention is only available on TPU devices and will raise
    NotImplementedError on CPU or GPU backends.

References:
    - JAX Pallas documentation for TPU kernels
    - Google Research papers on TPU-optimized attention
"""

from __future__ import annotations

import typing as tp

from ejkernel.modules import blocksparse_attention
from jax import numpy as jnp
from jax import random as jr
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int

from easydel.layers.caching.transformer import TransformerMetadata

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from .vanilla_attention import VanillaAttn

if tp.TYPE_CHECKING:
    pass


@OperationRegistry.register
class SplashAttn(OperationImpl):
    """
    An attention implementation using the Pallas Splash Attention kernel for TPUs.

    Splash Attention is an optimized attention mechanism designed for TPUs.
    This implementation provides a wrapper around the `make_splash_mqa_single_device`
    primitive.

    Note:
        - This implementation is primarily intended for TPUs.
        - It falls back to `VanillaAttn` under certain conditions:
            - Query sequence length is 1 (generation mode).
            - `causal` is False.
            - Query sequence length is not divisible by 128 (kernel constraint).
        - Non-TPU forward methods (`forward_native`, `forward_gpu`, etc.) are not
          implemented and will raise `NotImplementedError`.

    Registered under the name "splash".
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name of this attention implementation.

        Returns:
            The string "splash".
        """
        return "splash"

    def get_impl_metadata(self) -> OperationMetadata:
        """
        Returns the metadata associated with this attention implementation instance.

        Returns:
            The `OperationMetadata` provided during initialization.
        """
        return self.metadata

    def forward_native(
        self,
        query: Float[Array, "batch num_heads seq_len head_dim"],
        key: Float[Array, "batch kv_num_heads kv_len head_dim"],
        value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
        q_segment_ids: Int[Array, "batch seq_len"] | None = None,
        kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        attention_mask: Bool[Array, "batch num_heads seq_len kv_len"]
        | Bool[Array, "batch 1 seq_len kv_len"]
        | Int[Array, "batch num_heads seq_len kv_len"]
        | Int[Array, "batch 1 seq_len kv_len"]
        | None = None,
        logits_soft_cap: float | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        causal: bool = True,
        fused_backward: bool = False,
        cache_metadata: TransformerMetadata | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Performs Splash Attention on TPU/GPU using the Pallas/Triton kernel.

        Handles fallback logic, attention_mask processing, block size configuration, and
        sharding via `shard_map`. Expects inputs potentially in BTHD format and
        transposes them to BHTD for the kernel.

        Args:
            query: Query tensor (B, T, Hq, D).
            key: Key tensor (B, S, Hkv, D).
            value: Value tensor (B, S, Hkv, Dv).
            attention_mask: Optional boolean attention attention_mask (broadcastable to B, 1, T, S).
                Used to generate segment IDs if provided.
            causal: If True, applies causal masking via the kernel's attention_mask configuration.
                If False, falls back to VanillaAttn.
            **ignore: Ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention outputs. Attention weights
            are not computed or returned by Splash Attention.
        """
        query_lenght = query.shape[1]
        if not causal or ((query.shape[-1] % 128) != 0) or ((value.shape[-1] % 128) != 0) or query_lenght == 1:
            return VanillaAttn(self.metadata)(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                causal=causal,
                cache_metadata=cache_metadata,
                softmax_scale=softmax_scale,
                softmax_aux=softmax_aux,
                **ignore,
            )
        softmax_scale = softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5
        dtype = self.metadata.runtime_dtype
        model_mode = self.get_mode(query=query, BTHD=False)

        shardings = self.metadata.get_shardings(
            mode=model_mode,
            layout="bhtd",
            qkv_mni_sharding=True,
            softmax_aux=softmax_aux,
        )
        mask = attention_mask

        outputs = blocksparse_attention(
            query.transpose(0, 2, 1, 3).astype(dtype),
            key.transpose(0, 2, 1, 3).astype(dtype),
            value.transpose(0, 2, 1, 3).astype(dtype),
            q_segment_ids,
            kv_segment_ids,
            None,
            None,
            softmax_aux,
            None,
            mask,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            causal=causal,
            fused_backward=fused_backward,
            mesh=self.metadata.mesh,
            out_specs=self.create_stable_sharding(shardings.output, tensor=query, preserved_indices=[0, 1]),
            in_specs=(
                self.create_stable_sharding(shardings.query, dep=query, tensor=query, preserved_indices=[0, 1]),
                self.create_stable_sharding(shardings.key, dep=key, tensor=key, preserved_indices=[0, 1]),
                self.create_stable_sharding(shardings.value, dep=value, tensor=value, preserved_indices=[0, 1]),
                self.create_stable_sharding(
                    shardings.q_segment_ids,
                    dep=q_segment_ids,
                    tensor=q_segment_ids,
                    preserved_indices=[0],
                ),
                self.create_stable_sharding(
                    shardings.kv_segment_ids,
                    dep=kv_segment_ids,
                    tensor=kv_segment_ids,
                    preserved_indices=[0],
                ),
                PartitionSpec(None),
                PartitionSpec(None),
                self.create_stable_sharding(shardings.softmax_aux, dep=softmax_aux, tensor=softmax_aux),
                PartitionSpec(None),
                self.create_stable_sharding(shardings.mask, dep=mask, tensor=mask, preserved_indices=[0]),
            ),
        ).transpose(0, 2, 1, 3)

        return AttentionOutput(attention_weights=None, attention_outputs=outputs)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention.

        Splash Attention is TPU-specific and has no GPU implementation.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            NotImplementedError: Always raised as GPU execution is not supported.
        """
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention.

        Splash Attention is TPU-specific and has no GPU implementation.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            NotImplementedError: Always raised as GPU execution is not supported.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention.

        Splash Attention is TPU-specific and has no GPU implementation.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            NotImplementedError: Always raised as GPU execution is not supported.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention.

        Splash Attention is TPU-specific and has no GPU implementation.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            NotImplementedError: Always raised as GPU execution is not supported.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention.

        Splash Attention is TPU-specific and has no GPU implementation.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            NotImplementedError: Always raised as GPU execution is not supported.
        """
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch num_heads seq_len head_dim"],
        key: Float[Array, "batch kv_num_heads kv_len head_dim"],
        value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
        q_segment_ids: Int[Array, "batch seq_len"] | None = None,
        kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        attention_mask: Bool[Array, "batch num_heads seq_len kv_len"]
        | Bool[Array, "batch 1 seq_len kv_len"]
        | Int[Array, "batch num_heads seq_len kv_len"]
        | Int[Array, "batch 1 seq_len kv_len"]
        | None = None,
        logits_soft_cap: float | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        causal: bool = True,
        fused_backward: bool = False,
        cache_metadata: TransformerMetadata | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Executes the Splash Attention computation or falls back to Vanilla Attention.

        Calls the appropriate backend-specific forward method (`forward_tpu`) via
        `super().__call__`. If the backend is not TPU or fallback conditions are met,
        it relies on the fallback mechanism within `forward_tpu`.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attention_mask: Optional attention attention_mask.
            causal: If True, applies causal masking. Affects fallback logic and
                kernel configuration.
                        cache_metadata: cache view for current layer.
            **ignore: Additional ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return super().__call__(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            causal=causal,
            cache_metadata=cache_metadata,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            softmax_aux=softmax_aux,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=softmax_scale,
            fused_backward=fused_backward,
            **ignore,
        )


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    test_cases = [
        # (batch_size, q_seq_len, k_seq_len, q_heads, k_heads)
        (1, 2048, 2048, 32, 4),
        # (2, 2**13, 2**13, 32, 8),
        # (4, 2**14, 2**14, 16, 8),
        # (4, 2**13, 2**14, 16, 4),
    ]

    metadata = OperationMetadata(
        runtime_dtype=jnp.bfloat16,
        base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, -1)),
    )

    splash_attn = SplashAttn(metadata)
    vanilla_attn = VanillaAttn(metadata)

    for idx, (b, qs, ks, qh, kh) in enumerate(test_cases):
        d, vd = 128, 128
        print(
            f"Running test case {idx + 1}/{len(test_cases)}: b={b}, qs={qs}, ks={ks}, qh={qh}, kh={kh}, d={d}, vd={vd}"
        )
        key_q, key_k, key_v = jr.split(jr.PRNGKey(0), 3)

        query = jr.normal(key_q, (b, qs, qh, d), dtype=jnp.bfloat16)
        key = jr.normal(key_k, (b, ks, kh, d), dtype=jnp.bfloat16)
        value = jr.normal(key_v, (b, ks, kh, vd), dtype=jnp.bfloat16)

        attention_mask = SplashAttn._create_causal_mask(max(qs, ks))[-qs:, :ks]
        attention_mask = jnp.broadcast_to(attention_mask, (b, 1, qs, ks))
        splash_out = splash_attn(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            causal=True,
            sliding_window=None,
        ).attention_outputs
        vanilla_out = vanilla_attn(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            causal=True,
            sliding_window=None,
        ).attention_outputs
        is_close = jnp.allclose(splash_out, vanilla_out, atol=0.125)
        max_diff = jnp.max(jnp.abs(splash_out - vanilla_out))

        print(f"Test case {idx + 1} result: {'PASS' if is_close else 'FAIL'}")
        print(f"Maximum absolute difference: {max_diff}\n")
