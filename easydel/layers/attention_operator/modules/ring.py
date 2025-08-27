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
from functools import partial

import jax
import jax.lax as lax
from eformer.escale import with_sharding_constraint
from einops import rearrange
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as Ps

from easydel.kernels.tpu_ops import pallas_ring_attention

from .._attention_impl import AttentionImpl, AttentionMetadata, AttentionOutput, AttentionRegistry
from .vanilla import VanillaAttn


def blockwise_attn(
    query,
    key,
    value,
    bias=None,
    deterministic=True,
    dropout_rng=None,
    attn_pdrop=0.0,
    causal=True,
    query_chunk_size=2048,
    key_chunk_size=2048,
    dtype=jnp.float32,
    policy=jax.checkpoint_policies.nothing_saveable(),
    precision=None,
    float32_logits=True,
    prevent_cse=True,
):
    query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape

    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))

    query = jnp.moveaxis(query, 1, 0)
    key = jnp.moveaxis(key, 1, 0)
    value = jnp.moveaxis(value, 1, 0)

    if bias is not None:
        for bias_dim, broadcast_dim in zip(bias.shape, (batch, num_heads, q_len, kv_len), strict=False):
            assert bias_dim == 1 or bias_dim == broadcast_dim
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None

    _chunk_bias_fn = functools.partial(
        _chunk_attention_bias,
        query_chunk_size,
        key_chunk_size,
        bias,
        deterministic,
        attn_dropout,
        attn_pdrop,
        causal,
        dtype,
    )

    def scan_attention(args):
        query_chunk, query_chunk_idx = args

        @functools.partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            (numerator, denominator, prev_max_score) = carry
            attn_weights = jnp.einsum("bqhd,bkhd->bqhk", query_chunk, key_chunk, precision=precision)
            bias_chunk = _chunk_bias_fn(query_chunk_idx, key_chunk_idx)
            bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
            attn_weights = attn_weights + bias_chunk

            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
            max_score = jnp.maximum(prev_max_score, max_score)
            max_score = jax.lax.stop_gradient(max_score)
            exp_weights = jnp.exp(attn_weights - max_score)
            exp_values = jnp.einsum("bqhv,bvhd->bqhd", exp_weights, value_chunk, precision=precision)
            correction = jnp.exp(prev_max_score - max_score)
            numerator = numerator * correction + exp_values
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)
            return Carry(numerator, denominator, max_score), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            skip_block = jnp.array(False)
            if causal:
                skip_block = query_chunk_idx < key_chunk_idx
            return jax.lax.cond(
                skip_block,
                lambda carry, args: (carry, None),
                scan_kv_block,
                carry,
                args,
            )

        init_carry = Carry(
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
            (-jnp.inf) * jnp.ones((batch, query_chunk_size, num_heads, 1), dtype=query.dtype),
        )
        (numerator, denominator, max_score), _ = lax.scan(
            skip_upper_half, init_carry, xs=(key, value, jnp.arange(0, num_kv))
        )
        outputs = (numerator / denominator).astype(dtype)
        return outputs

    _, res = lax.scan(lambda _, x: ((), scan_attention(x)), (), xs=(query, jnp.arange(0, num_q)))
    res = rearrange(res, "n b c h d -> b (n c) h d")
    return res


class Carry(tp.NamedTuple):
    numerator: jax.Array
    denominator: jax.Array
    max_so_far: jax.Array


def _chunk_attention_bias(
    query_chunk_size,
    key_chunk_size,
    bias,
    deterministic,
    attn_dropout,
    attn_pdrop,
    causal,
    dtype,
    query_chunk_idx,
    key_chunk_idx,
):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *bias.shape[:2],
                min(bias.shape[-2], query_chunk_size),
                min(bias.shape[-1], key_chunk_size),
            ),
        )

    if causal:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
        offset = query_offset - key_offset
        query_idx += offset
        causal_mask_value = (query_idx < key_idx) * jnp.finfo(dtype).min
        chunk_bias += causal_mask_value.reshape(1, 1, *causal_mask_value.shape)

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias += attn_dropout_slice * jnp.finfo(dtype).min
    return chunk_bias.astype(dtype)


# TODO: Recheck this


@AttentionRegistry.register
class RingAttn(AttentionImpl):
    """
    Attention implementation using ring-passing algorithm or blockwise scan.

    This implementation supports:
    - Native (scan-based) blockwise attention via `blockwise_attn`.
    - TPU-specific ring attention using `pallas_ring_attention` kernel.

    It is registered under the name "ring".
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """Returns the registered name: "ring"."""
        return "ring"

    def get_impl_metadata(self) -> AttentionMetadata:
        """Returns the metadata associated with this instance."""
        return self.metadata

    @jax.named_scope("easydel-ringimpl-native-xla")
    def forward_native(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        bias: Array | None = None,
        init_bias: tp.Callable[[], Array] | None = None,
        deterministic: bool = False,
        dropout_rng: jr.PRNGKey = None,
        causal: bool = True,
        **ignore,
    ) -> AttentionOutput:
        """
        Computes attention using the scan-based `blockwise_attn` function.

        Handles optional mask/bias, KV head repetition, and sharding constraints.

        Args:
            q: Query tensor (B, T, H, D).
            k: Key tensor (B, S, H_kv, D).
            v: Value tensor (B, S, H_kv, D).
            mask: Optional boolean attention mask (broadcastable to B, 1, T, S).
            bias: Optional attention bias (broadcastable to B, H, T, S).
            init_bias: Optional callable to initialize bias if mask/bias are None.
            deterministic: If False, enables dropout. Requires `dropout_rng`.
            dropout_rng: JAX PRNG key for dropout if `deterministic` is False.
            causal: Apply causal mask if True.
            **ignore: Ignored keyword arguments.

        Returns:
            AttentionOutput containing the attention result.
        """

        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        dtype = self.metadata.runtime_dtype
        k, v = self.repeat_kv_heads(k, v, q.shape[2] // k.shape[2])

        query_lenght = q.shape[1]
        value_lenght = v.shape[1]

        model_mode = self.get_mode(q=q, BTHD=True)

        q_sharding, k_sharding, v_sharding, b_sharding, m_sharding, a_sharding = self.metadata.get_shardings(model_mode)

        blocksize_k = min(self.metadata.blocksize_k, value_lenght)
        blocksize_q = min(self.metadata.blocksize_q, query_lenght)
        with self.metadata.mesh:
            if mask is None and bias is None and init_bias is not None:
                bias = init_bias()

            if bias is None and mask is not None:
                bias = jnp.where(mask, 0, jnp.finfo(dtype).min)

            output = with_sharding_constraint(
                arr=blockwise_attn(
                    query=with_sharding_constraint(arr=q, sharding=q_sharding),
                    key=with_sharding_constraint(arr=k, sharding=k_sharding),
                    value=with_sharding_constraint(arr=v, sharding=v_sharding),
                    bias=with_sharding_constraint(arr=bias, sharding=b_sharding),
                    deterministic=deterministic,
                    dtype=dtype,
                    dropout_rng=dropout_rng,
                    precision=jax.lax.Precision.DEFAULT,
                    attn_pdrop=self.metadata.dropout_prob,
                    key_chunk_size=blocksize_k,
                    query_chunk_size=blocksize_q,
                    prevent_cse=False,
                    causal=causal,
                    float32_logits=True,
                ),
                sharding=a_sharding,
            )
            return AttentionOutput(attention_weights=None, attention_outputs=output)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Currently delegates to `forward_native` (scan-based)."""
        # TODO: Implement GPU-specific ring attention kernel if available
        return self.forward_cuda(*args, **kwargs)

    @jax.named_scope("easydel-ringimpl-tpu")
    def forward_tpu(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        bias: Array | None = None,
        init_bias: tp.Callable[[], Array] | None = None,
        deterministic: bool = False,
        dropout_rng: jr.PRNGKey = None,
        causal: bool = True,
        **ignore,
    ) -> AttentionOutput:
        """
        Computes Ring Attention on TPU using the `pallas_ring_attention` kernel.

        Handles optional mask/bias, sharding, and passes configuration to the kernel.

        Args:
            q: Query tensor (B, T, H, D).
            k: Key tensor (B, S, H_kv, D).
            v: Value tensor (B, S, H_kv, D).
            mask: Optional boolean attention mask (broadcastable to B, 1, T, S).
            bias: Optional attention bias (broadcastable to B, H, T, S).
            init_bias: Optional callable to initialize bias if mask/bias are None.
            deterministic: If False, potentially enables dropout within the kernel (if supported).
            dropout_rng: JAX PRNG key (may be used by the kernel if dropout is enabled).
            causal: Apply causal mask if True. Passed to the kernel.
            **ignore: Ignored keyword arguments.

        Returns:
            AttentionOutput containing the attention result.
        """
        sm_scale = self.metadata.softmax_scale
        sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        dtype = self.metadata.runtime_dtype

        model_mode = self.get_mode(q=q, BTHD=True)

        q_sharding, k_sharding, v_sharding, b_sharding, m_sharding, a_sharding = self.metadata.get_shardings(model_mode)

        if mask is None and bias is None and init_bias is not None:
            bias = init_bias()

        segment_ids = None

        if bias is None and mask is not None:
            bias = jnp.where(mask, 0, jnp.finfo(dtype).min)

        blocksize_k = min(self.metadata.blocksize_k, k.shape[1])
        blocksize_q = min(self.metadata.blocksize_q, q.shape[1])

        attn_output = shard_map(
            partial(
                pallas_ring_attention,
                axis_name=self.metadata.sequence_axis_name,
                float32_logits=True,
                cache_idx=None,
                query_chunk_size=blocksize_q,
                key_chunk_size=blocksize_k,
                causal_block_size=1 if causal else None,
            ),
            in_specs=(
                self.create_stable_sharding(q_sharding, dep=q),
                self.create_stable_sharding(k_sharding, dep=k),
                self.create_stable_sharding(v_sharding, dep=v),
                self.create_stable_sharding(b_sharding, [0], dep=b),
                self.create_stable_sharding(Ps(q_sharding[0], None), dep=segment_ids),
            ),
            out_specs=self.create_stable_sharding(a_sharding),
            mesh=self.metadata.mesh,
            check_rep=False,
        )(
            q.astype(dtype),
            k.astype(dtype),
            v.astype(dtype),
            bias,
            segment_ids,
        )

        return AttentionOutput(
            attention_weights=None,
            attention_outputs=with_sharding_constraint(
                arr=attn_output,
                sharding=a_sharding,
            ),
        )

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass. Delegates to `forward_native` (scan-based)."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA GPU forward pass. Currently delegates to `forward_native` (scan-based)."""
        # TODO: Implement GPU-specific ring attention kernel if available
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Currently delegates to `forward_native` (scan-based)."""
        # TODO: Implement ROCm-specific ring attention kernel if available
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
        mask: Array | None = None,
        bias: Array | None = None,
        init_bias: tp.Callable[[], Array] | None = None,
        deterministic: bool = False,
        dropout_rng: jr.PRNGKey = None,
        causal: bool = True,
        **ignore,
    ) -> AttentionOutput:
        """
        Executes the Ring Attention computation.

        Currently bypasses the backend dispatch and directly calls `forward_native`.
        (See TODO in the original code).

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            mask: Optional attention mask.
            bias: Optional attention bias.
            init_bias: Optional callable to initialize bias.
            deterministic: If False, enables dropout (requires dropout_rng).
            dropout_rng: JAX PRNG key for dropout if deterministic is False.
            causal: Apply causal mask if True.
            **ignore: Additional ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the results.
        """
        # TODO: Debug Ring Attention then restore super().__call__ dispatch.
        # The original code temporarily forces native execution.
        return self.forward_native(
            q=q,
            k=k,
            v=v,
            mask=mask,
            bias=bias,
            init_bias=init_bias,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            causal=causal,
        )


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    b, qs, ks, qh, kh, d, vd = 1, 1024, 1024, 32, 32, 128, 128
    q = jr.normal(jr.key(0), (b, qs, qh, d), "f2")
    k = jr.normal(jr.key(1), (b, ks, kh, d), "f2")
    v = jr.normal(jr.key(2), (b, ks, kh, vd), "f2")
    cu_mask = VanillaAttn._create_causal_mask(qs)[None, None, :, :].repeat(b, 0)
    metadata = AttentionMetadata(
        runtime_dtype=jnp.bfloat16,
        base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, -1)),
        blocksize_k=128,
        blocksize_q=128,
        backend="cpu",
    )

    vanilla = VanillaAttn(metadata)
    attn = RingAttn(metadata)

    out = attn(q=q, k=k, v=v, mask=cu_mask).attention_outputs
    vout = vanilla(q=q, k=k, v=v, mask=cu_mask).attention_outputs

    print(out[-1, -1, -1, -5:], out[-1, 0, -1, -5:])
    print(vout[-1, -1, -1, -5:], vout[-1, 0, -1, -5:])

    print(jnp.allclose(out, vout, atol=0.125))
