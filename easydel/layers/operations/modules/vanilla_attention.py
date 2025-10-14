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

import typing as tp

import jax
from eformer.escale import with_sharding_constraint
from ejkernel.modules import attention
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry


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

    def get_impl_metadata(self) -> OperationMetadata:
        """
        Returns the metadata associated with this attention implementation instance.

        Returns:
            The `OperationMetadata` provided during initialization.
        """
        return self.metadata

    @jax.named_scope("easydel-vanillaimpl-native-xla")
    def forward_native(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_len num_kv_heads head_dim"],
        attention_mask: Bool[Array, "batch 1 seq_len kv_len"] | None = None,
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
        with self.metadata.mesh:
            model_mode = self.get_mode(query=query, BTHD=True)
            shardings = self.metadata.get_shardings(model_mode, layout="bthd")
            if attention_mask is None and bias is None and init_bias is not None:
                bias = init_bias()
            if bias is None and attention_mask is None and init_bias is not None:
                bias = init_bias()
            query = with_sharding_constraint(arr=query, sharding=shardings.query)
            key = with_sharding_constraint(arr=key, sharding=shardings.key)
            value = with_sharding_constraint(arr=value, sharding=shardings.value)

            bias = with_sharding_constraint(arr=bias, sharding=shardings.bias) if bias is not None else bias
            attention_mask = (
                with_sharding_constraint(arr=attention_mask, sharding=shardings.mask)
                if attention_mask is not None
                else attention_mask
            )
            outputs, weights = attention(
                query,
                key,
                value,
                attention_mask,
                bias,
                dropout_rng,
                softmax_aux,
                deterministic=deterministic,
                dropout_prob=dropout_prob,
                dtype=self.metadata.runtime_dtype,
                sliding_window=sliding_window,
                softmax_dtype=self.metadata.runtime_softmax_dtype,
                softmax_scale=softmax_scale,
                init_bias=None,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
            )
            return AttentionOutput(
                attention_weights=weights,
                attention_outputs=with_sharding_constraint(arr=outputs, sharding=shardings.output),
            )

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
        attention_mask: Bool[Array, "batch 1 seq_len kv_len"] | None = None,
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
            attention_mask=attention_mask,
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
    a = jnp.astype(jr.randint(jr.key(3), (b, 1, qs, ks), 0, 4) > 2, "b1")

    metadata = OperationMetadata(
        runtime_dtype=jnp.float16,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )

    out = VanillaAttn(metadata)(query=query, key=key, value=value, mask=a)
    print(out.attention_outputs)
