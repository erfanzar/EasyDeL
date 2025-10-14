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

"""Scaled Dot-Product Attention implementation using JAX's optimized primitives.

This module provides an attention implementation that leverages JAX's
`jax.nn.dot_product_attention` API, which automatically dispatches to
the most efficient implementation available on the current hardware.

Key features:
- Automatic backend selection (XLA, cuDNN, Flash Attention)
- Support for multiple hardware backends (TPU, GPU, CPU)
- Efficient handling through JAX's SDPA primitive
- Automatic optimization based on hardware capabilities
- Compatible with various attention patterns (causal, masked, biased)

Implementation details:
- On CUDA GPUs: Uses cuDNN's optimized attention kernels
- On TPUs/CPUs: Uses XLA's optimized implementations
- Automatically selects Flash Attention when available
- Handles sharding for distributed computation

The implementation is registered under multiple names:
- "sdpa": Scaled Dot-Product Attention (generic name)
- "cudnn": Specifically for CUDA/cuDNN backend
- "cuda_flash_attn2": For Flash Attention v2 on CUDA

Example:
    >>> from easydel.layers.attention_operator import OperationMetadata
    >>> from easydel.layers.attention_operator.modules import ScaledDotProductAttn
    >>>
    >>> # Configure for efficient SDPA
    >>> metadata = OperationMetadata(
    ...     runtime_dtype=jnp.float16,
    ...     softmax_scale=1.0 / math.sqrt(head_dim)
    ... )
    >>> sdpa_attn = ScaledDotProductAttn(metadata)
    >>>
    >>> # Automatically uses best available implementation
    >>> output = sdpa_attn(query, key, value, attention_mask=attention_mask, causal=True)

Note:
    JAX will automatically select the best implementation based on:
    - Hardware availability (GPU with cuDNN, TPU, CPU)
    - Tensor shapes and dtypes
    - JAX version and installed libraries
    - Specific operation parameters (causal, attention_mask type)

References:
    - JAX documentation on dot_product_attention
    - NVIDIA cuDNN documentation
    - Flash Attention papers and implementations
"""

import typing as tp

import jax
from eformer import common_types
from eformer.escale import with_sharding_constraint
from ejkernel.modules import scaled_dot_product_attention
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.sharding import PartitionSpec
from jaxtyping import Bool, Float, Int

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry


@OperationRegistry.register
class ScaledDotProductAttn(OperationImpl):
    """
    An attention implementation that leverages `jax.nn.dot_product_attention`.

    This class utilizes JAX's optimized SDPA primitive, which can dispatch to
    different backend implementations (like XLA, cuDNN, or potentially Flash Attention
    emulation on CUDA depending on JAX version and hardware).

    It handles sharding using `shard_map` and manages backend-specific dispatch
    (primarily distinguishing between CUDA/GPU and other backends like TPU/CPU).

    Registered under the names "sdpa", "cudnn", and "cuda_flash_attn2".
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name(s) for this implementation.

        Returns:
            A tuple of strings: ("sdpa", "cudnn", "cuda_flash_attn2").
        """
        return "sdpa", "cudnn", "cuda_flash_attn2"

    def get_impl_metadata(self) -> OperationMetadata:
        """
        Returns the metadata associated with this attention implementation instance.

        Returns:
            The `OperationMetadata` provided during initialization.
        """
        return self.metadata

    @jax.named_scope("easydel-sdpa-impl-ejkernel")
    def forward_native(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_len num_kv_heads head_dim"],
        attention_mask: Bool[Array, "batch 1 seq_len kv_len"] | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        init_bias: tp.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        cum_seqlens_q: Int[Array, "batch"] | None = None,  # noqa
        cum_seqlens_k: Int[Array, "batch"] | None = None,  # noqa
        **ignore,
    ) -> AttentionOutput:
        """
        Computes attention using `jax.nn.dot_product_attention` with the "xla" implementation.

        This is typically used for CPU and TPU backends. It applies sharding via `shard_map`.

        Args:
            query: Query tensor (B, T, H, D).
            key: Key tensor (B, S, H_kv, D).
            value: Value tensor (B, S, H_kv, D_v).
            attention_mask: Optional boolean attention attention_mask (broadcastable to B, 1, T, S).
                Passed directly to the primitive.
            bias: Optional attention bias tensor (broadcastable to B, H, T, S).
                Passed directly to the primitive. If bias is provided, `causal` is forced to False.
            init_bias: Optional callable to initialize bias if attention_mask/bias are None.
            causal: If True and `bias` is None, applies causal masking within the primitive.
            **ignore: Ignored keyword arguments.

        Returns:
            An `AttentionOutput` object. Note that `jax.nn.dot_product_attention`
            typically does not return attention weights.
        """
        softmax_scale = softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5
        dtype = self.metadata.runtime_dtype
        model_mode = self.get_mode(query=query, BTHD=True)

        causal = causal if model_mode != common_types.MODE_DECODE else False
        shardings = self.metadata.get_shardings(model_mode, layout="bthd")

        if attention_mask is None and bias is None and init_bias is not None:
            bias = init_bias()

        attention_output = scaled_dot_product_attention(
            query.astype(dtype),
            key.astype(dtype),
            value.astype(dtype),
            attention_mask.astype("b1") if attention_mask is not None else None,
            bias.astype(dtype) if bias is not None else None,
            cum_seqlens_q,
            cum_seqlens_k,
            softmax_scale=softmax_scale,
            causal=causal,
            sliding_window=sliding_window,
            mesh=self.metadata.mesh,
            in_specs=(
                self.create_stable_sharding(shardings.query, [0, 2], dep=query),
                self.create_stable_sharding(shardings.key, [0, 2], dep=key),
                self.create_stable_sharding(shardings.value, [0, 2], dep=value),
                self.create_stable_sharding(shardings.mask, dep=attention_mask),
                self.create_stable_sharding(shardings.bias, dep=bias),
                self.create_stable_sharding(PartitionSpec(shardings.query[0]), dep=cum_seqlens_q),
                self.create_stable_sharding(PartitionSpec(shardings.query[0]), dep=cum_seqlens_k),
            ),
            out_specs=self.create_stable_sharding(shardings.output, [0, 2]),
        )
        return AttentionOutput(
            attention_weights=None,
            attention_outputs=with_sharding_constraint(arr=attention_output, sharding=shardings.output),
        )

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Delegates to the CUDA-specific implementation.

        Args:
            *args: Positional arguments for attention calculation.
            **kwargs: Keyword arguments for attention calculation.

        Returns:
            AttentionOutput: Result from CUDA-optimized implementation.
        """
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU forward pass. Delegates to `forward_native` (XLA implementation).

        Args:
            *args: Positional arguments for attention calculation.
            **kwargs: Keyword arguments for attention calculation.

        Returns:
            AttentionOutput: Result from XLA-optimized implementation.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass. Delegates to `forward_native` (XLA implementation).

        Args:
            *args: Positional arguments for attention calculation.
            **kwargs: Keyword arguments for attention calculation.

        Returns:
            AttentionOutput: Result from XLA-optimized implementation.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass. Delegates to `forward_native` (AUTO-DETECT implementation).

        Args:
            *args: Positional arguments for attention calculation.
            **kwargs: Keyword arguments for attention calculation.

        Returns:
            AttentionOutput: Result from XLA-optimized implementation.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Currently delegates to `forward_native`.

        Future versions may include ROCm-specific optimizations.

        Args:
            *args: Positional arguments for attention calculation.
            **kwargs: Keyword arguments for attention calculation.

        Returns:
            AttentionOutput: Result from XLA implementation.
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
        softmax_scale: float | None = None,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        cum_seqlens_q: Int[Array, "batch"] | None = None,  # noqa
        cum_seqlens_k: Int[Array, "batch"] | None = None,  # noqa
        **ignore,
    ) -> AttentionOutput:
        """
        Executes the Scaled Dot Product Attention computation using the appropriate backend.

        Calls the relevant backend-specific forward method (`forward_cuda`, `forward_native`)
        via the `super().__call__` dispatch mechanism based on the metadata's backend setting.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attention_mask: Optional attention attention_mask.
            bias: Optional attention bias.
            init_bias: Optional callable to initialize bias.
            causal: Boolean indicating if causal masking should be applied.
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
            init_bias=init_bias,
            causal=causal,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            cum_seqlens_q=cum_seqlens_q,
            cum_seqlens_k=cum_seqlens_k,
            **ignore,
        )


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    # Test cace when qkv might refer to mla
    b, qs, ks, qh, kh, d, vd = 1, 1024, 1024, 32, 8, 128, 128
    query = jr.normal(jr.key(0), (b, qs, qh, d), "f2")
    key = jr.normal(jr.key(1), (b, ks, kh, d), "f2")
    value = jr.normal(jr.key(2), (b, ks, kh, vd), "f2")
    a = jnp.astype(jr.randint(jr.key(3), (b, 1, qs, ks), 0, 4) > 2, "b1")

    gpu_attn = ScaledDotProductAttn(
        OperationMetadata(runtime_dtype=jnp.float16, base_config=EasyDeLBaseConfig(), backend="gpu")
    )
    cpu_attn = ScaledDotProductAttn(
        OperationMetadata(runtime_dtype=jnp.float16, base_config=EasyDeLBaseConfig(), backend="cpu")
    )
    tpu_attn = ScaledDotProductAttn(
        OperationMetadata(runtime_dtype=jnp.float16, base_config=EasyDeLBaseConfig(), backend="tpu")
    )

    cout = cpu_attn(query=query, key=key, value=value, attention_mask=a).attention_outputs
    gout = gpu_attn(query=query, key=key, value=value, attention_mask=a).attention_outputs
    tout = tpu_attn(query=query, key=key, value=value, attention_mask=a).attention_outputs

    print(jnp.allclose(cout, gout, atol=1e-3), jnp.allclose(tout, gout, atol=1e-3))
