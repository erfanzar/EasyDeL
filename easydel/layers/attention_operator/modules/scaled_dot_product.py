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

import functools
import typing as tp

import jax
from eformer import common_types
from eformer.escale import with_sharding_constraint
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.experimental.shard_map import shard_map

from .._attention_impl import AttentionImpl, AttentionMetadata, AttentionOutput, AttentionRegistry


@AttentionRegistry.register
class ScaledDotProductAttn(AttentionImpl):
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

    def get_impl_metadata(self) -> AttentionMetadata:
        """
        Returns the metadata associated with this attention implementation instance.

        Returns:
            The `AttentionMetadata` provided during initialization.
        """
        return self.metadata

    @jax.named_scope("easydel-sdpaimpl-native-xla")
    def forward_native(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        bias: Array | None = None,
        init_bias: tp.Callable[[], Array] | None = None,
        causal: bool = False,
        **ignore,
    ) -> AttentionOutput:
        """
        Computes attention using `jax.nn.dot_product_attention` with the "xla" implementation.

        This is typically used for CPU and TPU backends. It applies sharding via `shard_map`.

        Args:
            q: Query tensor (B, T, H, D).
            k: Key tensor (B, S, H_kv, D).
            v: Value tensor (B, S, H_kv, D_v).
            mask: Optional boolean attention mask (broadcastable to B, 1, T, S).
                Passed directly to the primitive.
            bias: Optional attention bias tensor (broadcastable to B, H, T, S).
                Passed directly to the primitive. If bias is provided, `causal` is forced to False.
            init_bias: Optional callable to initialize bias if mask/bias are None.
            causal: If True and `bias` is None, applies causal masking within the primitive.
            **ignore: Ignored keyword arguments.

        Returns:
            An `AttentionOutput` object. Note that `jax.nn.dot_product_attention`
            typically does not return attention weights.
        """
        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        dtype = self.metadata.runtime_dtype
        model_mode = self.get_mode(q=q, BTHD=True)
        func = functools.partial(
            jax.nn.dot_product_attention,
            implementation="xla",
            scale=sm_scale,
            is_causal=(causal if model_mode != common_types.MODE_DECODE else False),
        )

        q_sharding, k_sharding, v_sharding, b_sharding, m_sharding, a_sharding = self.metadata.get_shardings(model_mode)
        if mask is None and bias is None and init_bias is not None:
            bias = init_bias()
        with self.metadata.mesh:
            attention_output = shard_map(
                func,
                mesh=self.metadata.mesh,
                in_specs=(
                    self.create_stable_sharding(q_sharding, dep=q, tensor=q),
                    self.create_stable_sharding(k_sharding, dep=k, tensor=k),
                    self.create_stable_sharding(v_sharding, dep=v, tensor=v),
                    self.create_stable_sharding(b_sharding, dep=bias, tensor=bias),
                    self.create_stable_sharding(m_sharding, dep=mask, tensor=mask),
                ),
                out_specs=self.create_stable_sharding(a_sharding, tensor=q),
                check_rep=False,
            )(
                q.astype(dtype),
                k.astype(dtype),
                v.astype(dtype),
                bias.astype(dtype) if bias is not None else None,
                mask.astype("b1") if mask is not None else None,
            )
            return AttentionOutput(
                attention_weights=None,
                attention_outputs=with_sharding_constraint(arr=attention_output, sharding=a_sharding),
            )

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Delegates to the CUDA-specific implementation."""
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU forward pass. Delegates to `forward_native` (XLA implementation)."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass. Delegates to `forward_native` (XLA implementation)."""
        return self.forward_native(*args, **kwargs)

    @jax.named_scope("easydel-sdpaimpl-gpu-cuda")
    def forward_cuda(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        bias: Array | None = None,
        init_bias: tp.Callable[[], Array] | None = None,
        causal: bool = False,
        **ignore,
    ) -> AttentionOutput:
        """
        Computes attention using `jax.nn.dot_product_attention` with the "cudnn" implementation.

        This is optimized for NVIDIA GPUs using cuDNN. It applies sharding via `shard_map`.
        Note: The cuDNN implementation might have specific requirements (e.g., dtype).
        Causal masking is disabled during generation mode (q_len=1) as it's unnecessary.

        Args:
            q: Query tensor (B, T, H, D).
            k: Key tensor (B, S, H_kv, D).
            v: Value tensor (B, S, H_kv, D_v).
            mask: Optional boolean attention mask (broadcastable to B, 1, T, S).
            bias: Optional attention bias tensor (broadcastable to B, H, T, S).
            init_bias: Optional callable to initialize bias if mask/bias are None.
            causal: If True, applies causal masking within the primitive, unless in generation mode.
            **ignore: Ignored keyword arguments.

        Returns:
            An `AttentionOutput` object. Weights are not returned.
        """
        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        dtype = jnp.float16
        model_mode = self.get_mode(q=q, BTHD=True)
        func = functools.partial(
            jax.nn.dot_product_attention,
            implementation="cudnn",
            scale=sm_scale,
            is_causal=(causal if model_mode != common_types.MODE_DECODE else False),
        )

        q_sharding, k_sharding, v_sharding, b_sharding, m_sharding, a_sharding = self.metadata.get_shardings(model_mode)
        if mask is None and bias is None and init_bias is not None:
            bias = init_bias()
        with self.metadata.mesh:
            attention_output = shard_map(
                func,
                mesh=self.metadata.mesh,
                in_specs=(
                    self.create_stable_sharding(q_sharding, [0, 2], dep=q),
                    self.create_stable_sharding(k_sharding, [0, 2], dep=k),
                    self.create_stable_sharding(v_sharding, [0, 2], dep=v),
                    self.create_stable_sharding(b_sharding, dep=bias),
                    self.create_stable_sharding(m_sharding, dep=mask),
                ),
                out_specs=self.create_stable_sharding(a_sharding, [0, 2]),
                check_rep=False,
            )(
                q.astype(dtype),
                k.astype(dtype),
                v.astype(dtype),
                bias.astype(dtype) if bias is not None else None,
                mask.astype("b1") if mask is not None else None,
            )
            return AttentionOutput(
                attention_weights=None,
                attention_outputs=with_sharding_constraint(arr=attention_output, sharding=a_sharding),
            )

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Currently delegates to `forward_native`."""
        # ROCm might require a specific implementation ("hipblaslt"?) if supported by JAX SDPA
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        bias: Array | None = None,
        init_bias: tp.Callable[[], Array] | None = None,
        causal: bool = False,
        **ignore,
    ) -> AttentionOutput:
        """
        Executes the Scaled Dot Product Attention computation using the appropriate backend.

        Calls the relevant backend-specific forward method (`forward_cuda`, `forward_native`)
        via the `super().__call__` dispatch mechanism based on the metadata's backend setting.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            mask: Optional attention mask.
            bias: Optional attention bias.
            init_bias: Optional callable to initialize bias.
            causal: Boolean indicating if causal masking should be applied.
            **ignore: Additional ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return super().__call__(
            q=q,
            k=k,
            v=v,
            mask=mask,
            bias=bias,
            init_bias=init_bias,
            causal=causal,
            **ignore,
        )


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    # Test cace when qkv might refer to mla
    b, qs, ks, qh, kh, d, vd = 1, 1024, 1024, 32, 8, 128, 128
    q = jr.normal(jr.key(0), (b, qs, qh, d), "f2")
    k = jr.normal(jr.key(1), (b, ks, kh, d), "f2")
    v = jr.normal(jr.key(2), (b, ks, kh, vd), "f2")
    a = jnp.astype(jr.randint(jr.key(3), (b, 1, qs, ks), 0, 4) > 2, "b1")

    gpu_attn = ScaledDotProductAttn(
        AttentionMetadata(runtime_dtype=jnp.float16, base_config=EasyDeLBaseConfig(), backend="gpu")
    )
    cpu_attn = ScaledDotProductAttn(
        AttentionMetadata(runtime_dtype=jnp.float16, base_config=EasyDeLBaseConfig(), backend="cpu")
    )
    tpu_attn = ScaledDotProductAttn(
        AttentionMetadata(runtime_dtype=jnp.float16, base_config=EasyDeLBaseConfig(), backend="tpu")
    )

    cout = cpu_attn(q=q, k=k, v=v, mask=a).attention_outputs
    gout = gpu_attn(q=q, k=k, v=v, mask=a).attention_outputs
    tout = tpu_attn(q=q, k=k, v=v, mask=a).attention_outputs

    print(jnp.allclose(cout, gout, atol=1e-3), jnp.allclose(tout, gout, atol=1e-3))
