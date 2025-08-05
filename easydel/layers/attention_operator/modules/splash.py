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

import jax
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.experimental.pallas.ops.tpu.splash_attention import (
    BlockSizes,
    CausalMask,
    MultiHeadMask,
    SegmentIds,
    make_splash_mqa_single_device,
)
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as Ps

from easydel.kernels.tpu_ops import pallas_ragged_decode
from easydel.layers.caching.transformer import TransformerMetadata

from .._attention_impl import AttentionImpl, AttentionMetadata, AttentionOutput, AttentionRegistry
from .vanilla import VanillaAttn


@AttentionRegistry.register
class SplashAttn(AttentionImpl):
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

    def get_impl_metadata(self) -> AttentionMetadata:
        """
        Returns the metadata associated with this attention implementation instance.

        Returns:
            The `AttentionMetadata` provided during initialization.
        """
        return self.metadata

    def forward_native(self, *args, **kwargs) -> AttentionOutput:
        """Native (CPU) forward pass. Not implemented for Splash Attention."""
        raise NotImplementedError("Splash Attention does not have a native CPU implementation.")

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention."""
        raise NotImplementedError("Splash Attention does not have a generic GPU implementation.")

    @jax.named_scope("easydel-splashimpl-tpu")
    def forward_tpu(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        causal: bool = True,
        cache_metadata: TransformerMetadata | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Performs Splash Attention on TPU using the Pallas kernel.

        Handles fallback logic, mask processing, block size configuration, and
        sharding via `shard_map`. Expects inputs potentially in BTHD format and
        transposes them to BHTD for the kernel.

        Args:
            q: Query tensor (B, T, Hq, D).
            k: Key tensor (B, S, Hkv, D).
            v: Value tensor (B, S, Hkv, Dv).
            mask: Optional boolean attention mask (broadcastable to B, 1, T, S).
                Used to generate segment IDs if provided.
            causal: If True, applies causal masking via the kernel's mask configuration.
                If False, falls back to VanillaAttn.
            **ignore: Ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention outputs. Attention weights
            are not computed or returned by Splash Attention.
        """
        query_lenght = q.shape[1]
        value_lenght = v.shape[1]
        if not causal or ((q.shape[-1] % 128) != 0) or ((v.shape[-1] % 128) != 0) or query_lenght == 1:
            return VanillaAttn(self.metadata)(
                q=q,
                k=k,
                v=v,
                mask=mask,
                causal=causal,
                cache_metadata=cache_metadata,
                **ignore,
            )
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
        ) = self.metadata.get_shardings(model_mode, BTHD=False, qkv_mni_sharding=True)
        if mask is not None and mask.shape[0] != q.shape[0]:
            num_reps_mask = q.shape[0] // mask.shape[0]
            mask = jnp.repeat(mask, num_reps_mask, 0)

        block_sizes = BlockSizes(
            block_q=min(self.metadata.blocksize_q, query_lenght),
            block_kv_compute=min(self.metadata.blocksize_k, value_lenght),
            block_kv=min(self.metadata.blocksize_k, value_lenght),
            block_q_dkv=min(self.metadata.blocksize_q, query_lenght),
            block_kv_dkv=min(self.metadata.blocksize_k, value_lenght),
            block_kv_dkv_compute=min(self.metadata.blocksize_k, value_lenght),
            block_q_dq=min(self.metadata.blocksize_q, query_lenght),
            block_kv_dq=min(self.metadata.blocksize_k, value_lenght),
        )
        qkv_mask_sharding = Ps(q_sharding[0], q_sharding[2])
        views_sharding = Ps(q_sharding[0])
        q_mask, kv_mask = [None] * 2
        if mask is not None:
            q_mask, kv_mask = self._split_attention_mask(mask)
            q_mask, kv_mask = (q_mask.astype("i4"), kv_mask.astype("i4"))

        indexs, starts = [None] * 2
        if cache_metadata is not None:
            indexs, starts = cache_metadata.indexs, cache_metadata.starts

        @functools.partial(
            shard_map,
            mesh=self.metadata.mesh,
            in_specs=(
                self.create_stable_sharding(q_sharding, dep=q, tensor=q, preserved_indices=[0, 2]),
                self.create_stable_sharding(k_sharding, dep=k, tensor=k, preserved_indices=[0, 2]),
                self.create_stable_sharding(v_sharding, dep=v, tensor=v, preserved_indices=[0, 2]),
                self.create_stable_sharding(qkv_mask_sharding, dep=q_mask, tensor=q_mask),
                self.create_stable_sharding(qkv_mask_sharding, dep=kv_mask, tensor=kv_mask),
                self.create_stable_sharding(views_sharding, dep=indexs, tensor=indexs),
                self.create_stable_sharding(views_sharding, dep=starts, tensor=starts),
            ),
            out_specs=self.create_stable_sharding(a_sharding, tensor=q, preserved_indices=[0, 2]),
            check_rep=False,
        )
        def _wraped_flash_attn(q, k, v, q_mask, kv_mask, indexs, starts):
            if q.shape[-2] != 1:
                output_shape = (*q.shape[:-1], v.shape[-1])
                num_reps = q.shape[1] // k.shape[1]
                q = q.reshape((*q.shape[:-3], k.shape[-3], num_reps, q.shape[-2], q.shape[-1]))
                fn = jax.vmap(
                    jax.vmap(
                        make_splash_mqa_single_device(
                            mask=MultiHeadMask([CausalMask((q.shape[-2], k.shape[-2])) for _ in range(q.shape[-3])]),
                            block_sizes=block_sizes,
                        ),
                        in_axes=(0, 0, 0, None),
                    ),
                    in_axes=(0, 0, 0, 0),
                )
                m = None
                if kv_mask is not None:
                    m = SegmentIds(q_mask, kv_mask)
                out = fn(q * sm_scale, k, v, m).reshape(output_shape)
            else:
                q = q.transpose(0, 2, 1, 3) * sm_scale
                k = k.transpose(0, 2, 1, 3)
                v = v.transpose(0, 2, 1, 3)
                return pallas_ragged_decode(q, k, v, indexs, starts)[0].transpose(0, 2, 1, 3)
            return out

        attn = _wraped_flash_attn(
            q.transpose(0, 2, 1, 3).astype(dtype),
            k.transpose(0, 2, 1, 3).astype(dtype),
            v.transpose(0, 2, 1, 3).astype(dtype),
            q_mask,
            kv_mask,
            indexs,
            starts,
        ).transpose(0, 2, 1, 3)

        return AttentionOutput(attention_weights=None, attention_outputs=attn)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass. Not implemented for Splash Attention."""
        raise NotImplementedError("Splash Attention does not have a CPU implementation.")

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA GPU forward pass. Not implemented for Splash Attention."""
        raise NotImplementedError("Splash Attention does not have a CUDA implementation.")

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Splash Attention."""
        raise NotImplementedError("Splash Attention does not have a ROCm implementation.")

    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        causal: bool = True,
        cache_metadata: TransformerMetadata | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Executes the Splash Attention computation or falls back to Vanilla Attention.

        Calls the appropriate backend-specific forward method (`forward_tpu`) via
        `super().__call__`. If the backend is not TPU or fallback conditions are met,
        it relies on the fallback mechanism within `forward_tpu`.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            mask: Optional attention mask.
            causal: If True, applies causal masking. Affects fallback logic and
                kernel configuration.
                        cache_metadata: cache view for current layer.
            **ignore: Additional ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return super().__call__(
            q=q,
            k=k,
            v=v,
            mask=mask,
            causal=causal,
            cache_metadata=cache_metadata,
            **ignore,
        )


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    test_cases = [
        # (batch_size, q_seq_len, k_seq_len, q_heads, k_heads)
        (1, 2048, 2048, 32, 4),
        (2, 2**13, 2**13, 32, 8),
        (4, 2**14, 2**14, 16, 8),
        (4, 2**13, 2**14, 16, 4),
    ]

    metadata = AttentionMetadata(
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

        q = jr.normal(key_q, (b, qs, qh, d), dtype=jnp.float32)
        k = jr.normal(key_k, (b, ks, kh, d), dtype=jnp.float32)
        v = jr.normal(key_v, (b, ks, kh, vd), dtype=jnp.float32)

        mask = SplashAttn._create_causal_mask(max(qs, ks))[-qs:, :ks]
        mask = jnp.broadcast_to(mask, (b, 1, qs, ks))
        splash_out = splash_attn(q=q, k=k, v=v, mask=None).attention_outputs
        vanilla_out = vanilla_attn(q=q, k=k, v=v, mask=None).attention_outputs
        is_close = jnp.allclose(splash_out, vanilla_out, atol=0.125)
        max_diff = jnp.max(jnp.abs(splash_out - vanilla_out))

        print(f"Test case {idx + 1} result: {'PASS' if is_close else 'FAIL'}")
        print(f"Maximum absolute difference: {max_diff}\n")
