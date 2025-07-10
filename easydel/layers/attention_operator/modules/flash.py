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
from eformer.escale import with_sharding_constraint
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as pallas_flash_attention
from jax.experimental.shard_map import shard_map

from easydel.kernels.gpu_ops import triton_flash_attention

from .._attention_impl import AttentionImpl, AttentionMetadata, AttentionOutput, AttentionRegistry
from .vanilla import VanillaAttn


@AttentionRegistry.register
class FlashAttn(AttentionImpl):
    """
    An implementation of Flash Attention V2 using specialized JAX primitives.

    This class leverages `jax.experimental.pallas.ops.tpu.flash_attention` for TPUs
    and a Triton kernel (`triton_flash_attention`) for GPUs (CUDA). It is registered
    under the name "flash_attn2". CPU execution is not supported and will raise an error.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name of this attention implementation.

        Returns:
            The string "flash_attn2".
        """
        return "flash_attn2"

    def get_impl_metadata(self) -> AttentionMetadata:
        """
        Returns the metadata associated with this attention implementation instance.

        Returns:
            The `AttentionMetadata` provided during initialization.
        """
        return self.metadata

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
        Native (CPU) forward pass for Flash Attention. Not implemented.

        Raises:
            NotImplementedError: Flash Attention is not supported on CPU via this implementation.
        """
        raise NotImplementedError("Flash Attention v2 does not have a native CPU implementation.")

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """
        GPU forward pass. Delegates to the CUDA-specific implementation.

        Args:
            *args: Positional arguments for the attention calculation.
            **kwargs: Keyword arguments for the attention calculation.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return self.forward_gpu(*args, **kwargs)

    @jax.named_scope("easydel-flash-attnimpl-tpu")
    def forward_tpu(
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
        Performs Flash Attention V2 on TPU using `jax.experimental.pallas.ops.tpu.flash_attention`.

        Handles optional mask/bias, KV head repetition, and sharding based on metadata.
        Note: The Pallas implementation expects inputs in BHTD format.

        Args:
            q: Query tensor, expected shape (batch, q_seq_len, num_q_heads, head_dim).
            k: Key tensor, expected shape (batch, kv_seq_len, num_kv_heads, head_dim).
            v: Value tensor, expected shape (batch, kv_seq_len, num_kv_heads, head_dim).
            mask: Optional boolean attention mask. Bias will be generated from this if provided.
                Shape typically (batch, 1, q_seq_len, kv_seq_len) or broadcastable.
            bias: Optional attention bias tensor. Added directly to attention scores.
                Shape typically (batch, num_heads, q_seq_len, kv_seq_len) or broadcastable.
                Takes precedence over `mask`.
            init_bias: Optional callable function to initialize bias if `mask` and `bias` are None.
            causal: If True, applies a causal mask. Ignored if `q_seq_len` is 1 (generation).
            **ignore: Ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention outputs. Attention weights
            are typically not computed or returned by Flash Attention implementations.
        """
        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        dtype = self.metadata.runtime_dtype
        model_mode = self.get_mode(q=q, BTHD=False)

        (
            q_sharding,
            k_sharding,
            v_sharding,
            b_sharding,
            m_sharding,
            a_sharding,
        ) = self.metadata.get_shardings(model_mode, BTHD=False)

        if mask is None and bias is None and init_bias is not None:
            bias = init_bias()

        if bias is None and mask is not None:
            bias = jnp.where(mask, 0, jnp.finfo(q.dtype).min)
        k, v = self.repeat_kv_heads(k, v, q.shape[2] // k.shape[2])
        query_lenght = q.shape[1]
        value_lenght = v.shape[1]
        if bias is not None:
            if bias.shape[1] != v.shape[2]:
                bias = jnp.repeat(bias, v.shape[2] // bias.shape[1], 1)

        block_sizes = BlockSizes(
            block_q=min(self.metadata.blocksize_q, query_lenght),
            block_k_major=min(self.metadata.blocksize_k, value_lenght),
            block_k=min(self.metadata.blocksize_k, value_lenght),
            block_b=1,
            block_q_major_dkv=min(self.metadata.blocksize_q, query_lenght),
            block_k_major_dkv=min(self.metadata.blocksize_k, value_lenght),
            block_k_dkv=min(self.metadata.blocksize_k, value_lenght),
            block_q_dkv=min(self.metadata.blocksize_q, query_lenght),
            block_k_major_dq=min(self.metadata.blocksize_k, value_lenght),
            block_k_dq=min(self.metadata.blocksize_k, value_lenght),
            block_q_dq=min(self.metadata.blocksize_q, query_lenght),
        )

        @functools.partial(
            shard_map,
            mesh=self.metadata.mesh,
            in_specs=(
                self.create_stable_sharding(q_sharding, dep=q, tensor=q),
                self.create_stable_sharding(k_sharding, dep=k, tensor=k),
                self.create_stable_sharding(v_sharding, dep=v, tensor=v),
                self.create_stable_sharding(b_sharding, dep=bias, tensor=bias),
            ),
            out_specs=self.create_stable_sharding(a_sharding, tensor=q),
            check_rep=False,
        )
        def _wraped_flash_attn(q, k, v, b):
            out = pallas_flash_attention(
                q,
                k,
                v,
                b,
                sm_scale=sm_scale,
                block_sizes=block_sizes,
                causal=False if query_lenght == 1 else causal,
            )
            return out

        attn = _wraped_flash_attn(
            q.transpose(0, 2, 1, 3).astype(dtype),
            k.transpose(0, 2, 1, 3).astype(dtype),
            v.transpose(0, 2, 1, 3).astype(dtype),
            bias.astype(dtype) if bias is not None else bias,
        ).transpose(0, 2, 1, 3)

        return AttentionOutput(
            attention_weights=None,
            attention_outputs=with_sharding_constraint(arr=attn, sharding=a_sharding),
        )

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """
        CPU forward pass. Delegates to `forward_native`, which raises an error.

        Raises:
            NotImplementedError: Via `forward_native`.
        """
        return self.forward_native(*args, **kwargs)

    @jax.named_scope("easydel-flash-attnimpl-gpu-cuda-rocm")
    def forward_gpu(
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
        Performs Flash Attention V2 on GPU (CUDA) using a Triton kernel.

        Handles optional mask/bias, KV head repetition (inside the kernel or before),
        and sharding based on metadata. Assumes `triton_flash_attention` handles
        KV head repetition internally if needed or expects broadcastable KV.
        Assumes BTHD input/output format for the Triton kernel.

        Args:
            q: Query tensor, expected shape (batch, q_seq_len, num_q_heads, head_dim).
            k: Key tensor, expected shape (batch, kv_seq_len, num_kv_heads, head_dim).
            v: Value tensor, expected shape (batch, kv_seq_len, num_kv_heads, head_dim).
            mask: Optional boolean attention mask. Used by the kernel if bias is not provided.
                Shape typically (batch, 1, q_seq_len, kv_seq_len) or broadcastable.
            bias: Optional attention bias tensor. Added by the kernel.
                Shape typically (batch, num_heads, q_seq_len, kv_seq_len) or broadcastable.
                Takes precedence over `mask` within the kernel logic if both are somehow passed.
            init_bias: Optional callable function to initialize bias if `mask` and `bias` are None.
            causal: If True, instructs the kernel to apply a causal mask.
            **ignore: Ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention outputs. Attention weights
            are typically not computed or returned by Flash Attention implementations.
        """
        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        dtype = self.metadata.runtime_dtype
        model_mode = self.get_mode(q=q, BTHD=True)

        (
            q_sharding,
            k_sharding,
            v_sharding,
            b_sharding,
            m_sharding,
            a_sharding,
        ) = self.metadata.get_shardings(model_mode, BTHD=True)

        if bias is None and init_bias is not None:
            bias = init_bias()
            mask = None

        func = functools.partial(
            triton_flash_attention,
            dropout_prob=self.metadata.dropout_prob,
            dropout_seed=None,
            softmax_scale=sm_scale,
            varlen_mode=False,
            causal=causal,
        )
        attn = shard_map(
            func,
            mesh=self.metadata.mesh,
            in_specs=(
                self.create_stable_sharding(q_sharding, tensor=q, preserved_indices=[0, 2]),
                self.create_stable_sharding(k_sharding, tensor=k, preserved_indices=[0, 2]),
                self.create_stable_sharding(v_sharding, tensor=v, preserved_indices=[0, 2]),
                self.create_stable_sharding(m_sharding, dep=mask, tensor=mask, preserved_indices=[0, 1]),
                self.create_stable_sharding(b_sharding, dep=bias, tensor=bias, preserved_indices=[0, 1]),
            ),
            out_specs=self.create_stable_sharding(a_sharding, tensor=q, preserved_indices=[0, 2]),
            check_rep=False,
        )(
            q.astype(dtype),
            k.astype(dtype),
            v.astype(dtype),
            mask,
            bias.astype(dtype) if bias is not None else bias,
        )
        return AttentionOutput(
            attention_weights=None,
            attention_outputs=with_sharding_constraint(arr=attn, sharding=a_sharding),
        )

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """
        ROCm GPU forward pass.
        """
        # ROCm would require a specific hipFlashAttention kernel or similar
        return self.forward_gpu(*args, **kwargs)

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
        Calls the appropriate backend-specific forward method (`forward_tpu`, `forward_cuda`, etc.)
        based on the metadata's backend setting, using the `super().__call__` dispatch mechanism.

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
            An `AttentionOutput` object containing the results from the dispatched method.
        """
        return super().__call__(q=q, k=k, v=v, mask=mask, bias=bias, init_bias=init_bias, causal=causal)


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    # Test cace when qkv might refer to mla
    b, qs, ks, qh, kh, d, vd = 4, 1024, 1024, 32, 32, 128, 128
    q = jr.normal(jr.key(0), (b, qs, qh, d), "f4")
    k = jr.normal(jr.key(1), (b, ks, kh, d), "f4")
    v = jr.normal(jr.key(2), (b, ks, kh, vd), "f4")
    a = jnp.astype(jr.randint(jr.key(3), (b, 1, qs, ks), 0, 4) > 2, "b1")
    metadata = AttentionMetadata(
        runtime_dtype=jnp.bfloat16,
        base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, -1, 1)),
    )
    attn = FlashAttn(metadata)
    vanilla = VanillaAttn(metadata)
    fout = attn(q=q, k=k, v=v, mask=a, causal=False).attention_outputs
    vout = vanilla(q=q, k=k, v=v, mask=a).attention_outputs
    print(fout[-1, -1, -1, -5:])
    print(vout[-1, -1, -1, -5:])
