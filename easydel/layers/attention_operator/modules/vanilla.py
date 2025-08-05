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

import typing as tp

import jax
from eformer.escale import with_sharding_constraint
from flax.nnx.nn.dtypes import promote_dtype
from jax import Array
from jax import numpy as jnp
from jax import random as jr

from .._attention_impl import AttentionImpl, AttentionMetadata, AttentionOutput, AttentionRegistry


@AttentionRegistry.register
class VanillaAttn(AttentionImpl):
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

    def get_impl_metadata(self) -> AttentionMetadata:
        """
        Returns the metadata associated with this attention implementation instance.

        Returns:
            The `AttentionMetadata` provided during initialization.
        """
        return self.metadata

    @jax.named_scope("easydel-vanillaimpl-native-xla")
    def forward_native(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        bias: Array | None = None,
        init_bias: tp.Callable[[], Array] | None = None,
        deterministic: bool = True,  # Default to deterministic (no dropout)
        dropout_rng: jax.random.PRNGKey = None,
        softmax_aux: Array | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Computes multi-head attention using standard JAX operations.

        Supports GQA/MQA by reshaping the query tensor to match the number of
        key/value heads. Applies scaling, optional bias/mask, softmax (potentially
        in float32), and optional dropout.

        Args:
            q: Query tensor (B, T, H_q, D).
            k: Key tensor (B, S, H_kv, D).
            v: Value tensor (B, S, H_kv, D_v).
            mask: Optional boolean attention mask (broadcastable to B, 1, T, S).
                Used if `bias` is not provided.
            bias: Optional attention bias tensor (broadcastable to B, H_q, T, S).
                Takes precedence over `mask`.
            init_bias: Optional callable to initialize bias if mask/bias are None.
            deterministic: If True, disables dropout.
            dropout_rng: JAX PRNG key for dropout. Required if `deterministic` is False
                and `dropout_prob` > 0.
            **ignore: Ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention weights (if computed)
            and the final attention outputs.

        Raises:
            NotImplementedError: If the bias head dimension cannot be reshaped correctly
                to match the query head structure for GQA/MQA.
        """

        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        dtype = self.metadata.runtime_dtype
        softmax_dtype = self.metadata.runtime_softmax_dtype

        if softmax_dtype is None:
            softmax_dtype = jnp.float32

        model_mode = self.get_mode(q=q, BTHD=True)
        q_sharding, k_sharding, v_sharding, b_sharding, m_sharding, a_sharding = self.metadata.get_shardings(model_mode)
        if mask is None and bias is None and init_bias is not None:
            bias = init_bias()
        with self.metadata.mesh:
            if bias is None and mask is None and init_bias is not None:
                bias = init_bias()

            b, qs, qh, d = q.shape
            b, ks, kh, d = k.shape
            *_, vd = v.shape
            num_reps = qh // kh
            q = with_sharding_constraint(arr=q, sharding=q_sharding)
            k = with_sharding_constraint(arr=k, sharding=k_sharding)
            v = with_sharding_constraint(arr=v, sharding=v_sharding)

            bias = with_sharding_constraint(arr=bias, sharding=b_sharding) if bias is not None else bias
            mask = with_sharding_constraint(arr=mask, sharding=m_sharding) if mask is not None else mask

            q = jnp.reshape(q, (b, qs, kh, num_reps, d))
            q, k, v = promote_dtype((q, k, v), dtype=dtype)

            aw = jnp.einsum("bskhd,bmkd->bkhsm", q * sm_scale, k, optimize=True)

        if bias is not None:
            if bias.shape[1] == (kh * num_reps):
                bias = bias.reshape(b, kh, num_reps, qs, ks)
            elif bias.shape[1] == kh:
                bias = bias.reshape(b, kh, 1, qs, ks)
            elif bias.shape[1] == 1:
                bias = bias.reshape(b, 1, 1, qs, ks)
            else:
                raise NotImplementedError("bias heads wont match!")
            aw = jnp.add(aw, bias.astype(aw.dtype))

        elif mask is not None:
            if mask.dtype != jnp.bool_:
                mask = mask.astype(jnp.bool_)

            if mask.ndim == 4:
                if mask.shape[1] == 1:
                    mask = jnp.broadcast_to(mask, (b, kh, qs, ks))
                    mask = jnp.reshape(mask, (b, kh, 1, qs, ks))
                elif mask.shape[1] == kh:
                    mask = jnp.reshape(mask, (b, kh, 1, qs, ks))
                elif mask.shape[1] == (kh * num_reps):
                    mask = jnp.reshape(mask, (b, kh, num_reps, qs, ks))
                else:
                    mask = jnp.broadcast_to(mask[:, :1], (b, 1, qs, ks))
                    mask = jnp.reshape(mask, (b, 1, 1, qs, ks))
            elif mask.ndim == 3:
                mask = jnp.reshape(mask, (b, 1, 1, qs, ks))
            elif mask.ndim == 2:
                mask = jnp.reshape(mask, (b, 1, 1, 1, ks))
                mask = jnp.broadcast_to(mask, (b, 1, 1, qs, ks))
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

            aw = jnp.where(mask, aw, jnp.finfo(aw.dtype).min)
        if softmax_aux is not None:
            if softmax_aux.ndim == 2:
                sinks = softmax_aux.reshape(1, kh, -1, 1, 1)
                sinks = jnp.broadcast_to(sinks, (b, kh, num_reps, qs, 1))
            elif softmax_aux.ndim == 1:
                sinks = softmax_aux.reshape(1, kh, -1, 1, 1)
                sinks = jnp.broadcast_to(sinks, (b, kh, num_reps, qs, 1))
            else:
                raise ValueError(f"Unsupported softmax_aux shape: {softmax_aux.shape}")
            combined_logits = jnp.concatenate([aw, sinks], axis=-1)
            combined_logits = combined_logits - jnp.max(combined_logits, axis=-1, keepdims=True)
            probs = jax.nn.softmax(combined_logits.astype(softmax_dtype), axis=-1).astype(dtype)
            aw = probs[..., :-1]
        else:
            aw = jax.nn.softmax(aw.astype(softmax_dtype), axis=-1).astype(dtype)

        dp = self.metadata.dropout_prob
        if not deterministic and dp > 0.0 and dropout_rng is not None:
            keep_prob = 1.0 - dp
            dropout_shape = tuple([1] * (k.ndim - 2)) + aw.shape[-2:]
            keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
            multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
            aw = aw * multiplier

        attention = jnp.einsum("bkhsm,bmkd->bskhd", aw, v, optimize=True).reshape(b, qs, qh, vd)

        return AttentionOutput(
            attention_weights=aw,
            attention_outputs=with_sharding_constraint(arr=attention, sharding=a_sharding),
        )

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Delegates to `forward_native`."""
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU forward pass. Delegates to `forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass. Delegates to `forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA GPU forward pass. Delegates to `forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Delegates to `forward_native`."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        bias: Array | None = None,
        init_bias: tp.Callable[[], Array] | None = None,
        deterministic: bool = True,
        dropout_rng: jax.random.PRNGKey = None,
        softmax_aux: Array | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Executes the vanilla attention computation.

        Calls the appropriate backend-specific forward method via `super().__call__`.
        Since all backend methods delegate to `forward_native`, this effectively
        always runs the native JAX implementation.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            mask: Optional attention mask.
            bias: Optional attention bias.
            init_bias: Optional callable to initialize bias.
            deterministic: If True, disables dropout.
            dropout_rng: JAX PRNG key for dropout if deterministic is False.
            **ignore: Additional ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        # Uses the BaseOperation.__call__ which reads self.metadata.backend for dispatch,
        # but all paths in VanillaAttn lead back to forward_native.
        return super().__call__(
            q=q,
            k=k,
            v=v,
            mask=mask,
            bias=bias,
            init_bias=init_bias,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            softmax_aux=softmax_aux,
            **ignore,
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

    attn = VanillaAttn(metadata)
    out = attn(q=q, k=k, v=v, mask=a)
