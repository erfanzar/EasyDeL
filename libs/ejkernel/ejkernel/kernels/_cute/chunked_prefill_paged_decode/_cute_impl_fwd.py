# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""CuTe forward implementation for chunked prefill + paged decode attention.

This module implements a CuTe DSL kernel that updates the paged KV cache from
packed token keys/values and then runs unified attention on the updated cache.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import jax
import jax.numpy as jnp

from ejkernel.callib._cute_call import cute_call
from ejkernel.callib._cute_ffi import build_cute_ffi_call, has_cute_ffi_support
from ejkernel.errors import EjkernelRuntimeError

try:
    from ejkernel.kernels._triton.unified_attention import unified_attention as triton_unified_attention
except Exception:  # pragma: no cover
    triton_unified_attention = None

os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")

try:
    _THREADS = max(32, int(os.getenv("EJKERNEL_CUTE_CPPD_THREADS", "256")))
except ValueError:
    _THREADS = 256


def _dtype_to_cutlass(dtype) -> type[cutlass.Numeric]:
    """Map a JAX/NumPy dtype to the corresponding CUTLASS scalar type.

    Supported mappings: ``float16`` → ``Float16``, ``bfloat16`` → ``BFloat16``,
    ``float32`` → ``Float32``, ``int32`` → ``Int32``, ``uint32`` → ``Uint32``.

    Raises:
        TypeError: If *dtype* has no CUTLASS equivalent in this module.
    """
    dt = jnp.dtype(dtype)
    if dt == jnp.float16:
        return cutlass.Float16
    if dt == jnp.bfloat16:
        return cutlass.BFloat16
    if dt == jnp.float32:
        return cutlass.Float32
    if dt == jnp.int32:
        return cutlass.Int32
    if dt == jnp.uint32:
        return cutlass.Uint32
    raise TypeError(f"Unsupported dtype for CuTe DSL: {dt}")


def _fake_tensor(dtype: type[cutlass.Numeric], shape: tuple[int, ...]):
    """Create a compact fake tensor descriptor for CuTe kernel compilation.

    Args:
        dtype: CUTLASS element type.
        shape: Logical tensor shape.

    Returns:
        A :func:`cute.runtime.make_fake_compact_tensor` object with
        C-order (row-major) strides; not a real device buffer.
    """
    stride_order = tuple(range(len(shape) - 1, -1, -1))
    return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=stride_order)


def _compute_token_linear_indices(
    *,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    total_tokens: int,
    block_size: int,
) -> jax.Array:
    """Compute destination linear cache indices for each packed token.

    Args:
        kv_lens: Total KV lengths per sequence, shape ``[num_seqs]``.
        block_tables: Sequence-to-physical-block mapping,
            shape ``[num_seqs, max_blocks_per_seq]``.
        query_start_loc: Cumulative packed-query starts,
            shape ``[num_seqs + 1]``.
        total_tokens: Number of packed query/key/value tokens.
        block_size: Number of tokens per physical KV cache block.

    Returns:
        Int32 tensor of shape ``[total_tokens]`` where each element is a linear
        position into ``cache.reshape(num_blocks * block_size, ...)``.
    """
    token_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    seq_idx = jnp.searchsorted(query_start_loc[1:], token_idx, side="right").astype(jnp.int32)

    q_start = query_start_loc[seq_idx]
    q_end = query_start_loc[seq_idx + 1]
    q_len = q_end - q_start
    ctx_len = kv_lens[seq_idx] - q_len

    seq_pos = token_idx - q_start
    kv_pos = ctx_len + seq_pos
    block_idx = kv_pos // jnp.int32(block_size)
    within = kv_pos - block_idx * jnp.int32(block_size)
    phys_block = block_tables[seq_idx, block_idx]
    return (phys_block * jnp.int32(block_size) + within).astype(jnp.int32)


def _build_kv_update_host_fns() -> tuple[Callable[..., None], Callable[..., None]]:
    """Build runtime and stream-based host functions for KV cache updates.

    Creates a CuTe DSL kernel that scatter-writes packed key/value tokens
    into the correct physical slots of a paged KV cache.  Each CUDA thread
    block processes one token, with threads cooperatively copying across the
    ``(num_kv_heads * head_dim)`` feature elements using a strided loop.

    The grid is ``(total_tokens, 1, 1)`` and the block size is controlled
    by the ``EJKERNEL_CUTE_CPPD_THREADS`` environment variable (default 256).

    Returns:
        Tuple of ``(runtime_launcher, jax_stream_launcher)`` callables that
        accept ``(keys, values, linear_indices, key_cache_in, value_cache_in,
        out_key_cache, out_value_cache)``.
    """

    @cute.kernel
    def _kv_update_kernel(
        keys: cute.Tensor,
        values: cute.Tensor,
        linear_indices: cute.Tensor,
        out_key_cache: cute.Tensor,
        out_value_cache: cute.Tensor,
    ):
        total_tokens = keys.shape[0]
        num_kv_heads = keys.shape[1]
        head_dim = keys.shape[2]
        block_size = out_key_cache.shape[1]

        token_idx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        if token_idx < total_tokens:
            linear_pos = cutlass.Int32(linear_indices[(token_idx,)])
            block_size_i = cutlass.Int32(block_size)
            phys_block = linear_pos // block_size_i
            within_block = linear_pos - phys_block * block_size_i

            feature_i = cutlass.Int32(tidx)
            head_dim_i = cutlass.Int32(head_dim)
            total_features = cutlass.Int32(num_kv_heads * head_dim)
            thread_stride = cutlass.Int32(_THREADS)

            while feature_i < total_features:
                kv_head_idx = feature_i // head_dim_i
                head_dim_idx = feature_i - kv_head_idx * head_dim_i

                out_key_cache[(phys_block, within_block, kv_head_idx, head_dim_idx)] = keys[
                    (token_idx, kv_head_idx, head_dim_idx)
                ]
                out_value_cache[(phys_block, within_block, kv_head_idx, head_dim_idx)] = values[
                    (token_idx, kv_head_idx, head_dim_idx)
                ]

                feature_i = feature_i + thread_stride

    @cute.jit
    def _kv_update_host_runtime(
        keys: cute.Tensor,
        values: cute.Tensor,
        linear_indices: cute.Tensor,
        _key_cache_in: cute.Tensor,
        _value_cache_in: cute.Tensor,
        out_key_cache: cute.Tensor,
        out_value_cache: cute.Tensor,
    ):
        total_tokens = keys.shape[0]
        _kv_update_kernel(
            keys,
            values,
            linear_indices,
            out_key_cache,
            out_value_cache,
        ).launch(
            grid=(total_tokens, 1, 1),
            block=(_THREADS, 1, 1),
        )

    @cute.jit
    def _kv_update_host_jax(
        stream: cuda.CUstream,
        keys: cute.Tensor,
        values: cute.Tensor,
        linear_indices: cute.Tensor,
        _key_cache_in: cute.Tensor,
        _value_cache_in: cute.Tensor,
        out_key_cache: cute.Tensor,
        out_value_cache: cute.Tensor,
    ):
        total_tokens = keys.shape[0]
        _kv_update_kernel(
            keys,
            values,
            linear_indices,
            out_key_cache,
            out_value_cache,
        ).launch(
            grid=(total_tokens, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    return _kv_update_host_runtime, _kv_update_host_jax


def _get_cute_kv_update_call(
    *,
    keys: jax.Array,
    values: jax.Array,
    linear_indices: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Build a CuTe KV-update callable backed by CuTe FFI primitives.

    Compiles the KV-update CuTe kernel via ``build_cute_ffi_call`` with
    in-place aliasing (``key_cache`` and ``value_cache`` are updated in place)
    and returns a callable that performs the scatter-write operation.

    Args:
        keys: Packed key tokens, shape ``[total_tokens, num_kv_heads, head_dim]``.
        values: Packed value tokens, same shape as *keys*.
        linear_indices: Linear destination indices into the flattened cache,
            shape ``[total_tokens]``, int32.
        key_cache: Existing key cache, shape
            ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        value_cache: Existing value cache, same shape as *key_cache*.

    Returns:
        A callable ``(keys, values, linear_indices, key_cache, value_cache)
        -> (updated_key_cache, updated_value_cache)``.

    Raises:
        RuntimeError: If CuTe TVM-FFI support is not available.
    """

    if not has_cute_ffi_support():
        raise RuntimeError(
            "CUTE chunked_prefill_paged_decode requires CuTe primitive support. "
            "Ensure CuTe TVM-FFI support is available (install apache-tvm-ffi)."
        )

    _, host_jax_fn = _build_kv_update_host_fns()
    out_shape = (
        jax.ShapeDtypeStruct(key_cache.shape, key_cache.dtype),
        jax.ShapeDtypeStruct(value_cache.shape, value_cache.dtype),
    )
    primitive_call = build_cute_ffi_call(
        host_jax_fn,
        output_shape_dtype=out_shape,
        input_output_aliases={3: 0, 4: 1},
        compile_options="--enable-tvm-ffi",
    )

    def _call(
        keys_arr: jax.Array,
        values_arr: jax.Array,
        linear_idx_arr: jax.Array,
        key_cache_arr: jax.Array,
        value_cache_arr: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return cute_call(
            keys_arr,
            values_arr,
            linear_idx_arr,
            key_cache_arr,
            value_cache_arr,
            call=primitive_call,
            out_shape=out_shape,
            name="cute_chunked_prefill_kv_update",
        )

    return _call


def _run_unified_attention(
    *,
    queries: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    alibi_slopes: jax.Array | None,
    softmax_aux: jax.Array | None,
    softmax_scale: float,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    seq_threshold_3d: int | None,
    num_par_softmax_segments: int | None,
    num_warps: int | None,
    num_stages: int | None,
) -> jax.Array:
    """Execute paged attention on the updated KV cache via Triton unified attention.

    Delegates to the Triton ``unified_attention`` kernel which supports
    variable-length sequences with paged KV caches.  This function serves
    as the attention stage of the two-phase chunked-prefill-paged-decode
    pipeline (phase 1: KV cache update via CuTe, phase 2: attention via
    Triton).

    Args:
        queries: Packed queries, shape ``[total_tokens, num_q_heads, head_dim]``.
        key_cache: Updated key cache, shape
            ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        value_cache: Updated value cache, same shape as *key_cache*.
        kv_lens: Total KV length per sequence, shape ``[num_seqs]``.
        block_tables: Block table mapping, shape ``[num_seqs, max_blocks]``.
        query_start_loc: Cumulative query starts, shape ``[num_seqs + 1]``.
        alibi_slopes: Optional ALiBi slopes, shape ``[num_q_heads]``.
        softmax_aux: Optional attention-sink logits, shape ``[num_q_heads]``.
        softmax_scale: Attention scale factor.
        sliding_window: Optional sliding-window size.
        logits_soft_cap: Optional logit soft-capping value.
        seq_threshold_3d: Optional Triton tuning hint for 3-D grid threshold.
        num_par_softmax_segments: Optional Triton parallel-softmax segment count.
        num_warps: Optional Triton warp count.
        num_stages: Optional Triton pipeline stage count.

    Returns:
        Attention output tensor, shape ``[total_tokens, num_q_heads, head_dim]``.

    Raises:
        EjkernelRuntimeError: If Triton unified attention is not available.
    """
    if triton_unified_attention is not None:
        return triton_unified_attention(
            queries=queries,
            key_cache=key_cache,
            value_cache=value_cache,
            kv_lens=kv_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            alibi_slopes=alibi_slopes,
            qq_bias=None,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            causal=True,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            seq_threshold_3d=seq_threshold_3d,
            num_par_softmax_segments=num_par_softmax_segments,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    del (
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        alibi_slopes,
        softmax_aux,
        softmax_scale,
        sliding_window,
        logits_soft_cap,
        seq_threshold_3d,
        num_par_softmax_segments,
        num_warps,
        num_stages,
    )
    raise EjkernelRuntimeError(
        "chunked_prefill_paged_decode (platform=cute) requires Triton unified_attention; fallback paths are disabled."
    )


def chunked_prefill_paged_decode_cute(
    *,
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    alibi_slopes: jax.Array | None,
    softmax_aux: jax.Array | None,
    softmax_scale: float,
    causal: bool,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    seq_threshold_3d: int | None,
    num_par_softmax_segments: int | None,
    num_warps: int | None,
    num_stages: int | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Execute the CuTe chunked prefill + paged decode attention pipeline.

    Implements a two-phase pipeline:
    1. CuTe DSL KV cache update: computes per-token linear cache destinations
       via :func:`_compute_token_linear_indices`, then scatter-writes all
       packed keys and values into the paged KV cache in a single GPU kernel.
    2. Triton unified attention: runs causal paged attention on the updated
       cache.

    Args:
        queries: Packed query tokens, shape
            ``(total_tokens, num_q_heads, head_dim)``.
        keys: Packed key tokens to insert, shape
            ``(total_tokens, num_kv_heads, head_dim)``.
        values: Packed value tokens to insert, same shape as *keys*.
        key_cache: Existing paged key cache, shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        value_cache: Existing paged value cache, same shape as *key_cache*.
        kv_lens: Total KV length per sequence, shape ``(num_seqs,)``, int32.
        block_tables: Block table mapping, shape
            ``(num_seqs, max_blocks_per_seq)``, int32.
        query_start_loc: Cumulative query starts, shape ``(num_seqs + 1,)``,
            int32.
        alibi_slopes: Optional ALiBi slopes, shape ``(num_q_heads,)``.
        softmax_aux: Optional attention-sink logits, shape ``(num_q_heads,)``.
        softmax_scale: Attention scale factor (pre-computed; not ``None``
            here — the interface layer resolves the default).
        causal: Whether causal masking is enabled. Must be ``True``.
        sliding_window: Optional sliding-window size.
        logits_soft_cap: Optional logit soft-capping value.
        seq_threshold_3d: Optional Triton decode-kernel threshold hint.
        num_par_softmax_segments: Optional Triton segmented-softmax hint.
        num_warps: Optional Triton warp-count override.
        num_stages: Optional Triton pipeline-stage override.

    Returns:
        A 3-tuple ``(attention_output, updated_key_cache, updated_value_cache)``
        where ``attention_output`` has shape
        ``(total_tokens, num_q_heads, head_dim)``.

    Raises:
        NotImplementedError: If *causal* is ``False``.
        ValueError: If tensor ranks, shapes, or dtypes are invalid, or if the
            input arrays are not on a GPU device.
        EjkernelRuntimeError: If Triton unified attention is not available.
    """
    if not causal:
        raise NotImplementedError("chunked_prefill_paged_decode only supports causal attention.")
    if queries.ndim != 3:
        raise ValueError("queries must be [total_tokens, num_q_heads, head_dim]")
    if keys.ndim != 3 or values.ndim != 3:
        raise ValueError("keys/values must be [total_tokens, num_kv_heads, head_dim]")
    if keys.shape != values.shape:
        raise ValueError("keys/values shape mismatch")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache/value_cache shape mismatch")
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise ValueError("kv_lens/block_tables/query_start_loc must be int32")

    try:
        query_device = queries.device()
    except Exception:
        query_device = None
    if query_device is not None and getattr(query_device, "platform", None) != "gpu":
        raise ValueError("CUTE chunked_prefill_paged_decode requires GPU backend.")

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    if keys.shape[0] != total_tokens:
        raise ValueError("keys/values first dim must match total_tokens")
    num_seqs = int(kv_lens.shape[0])
    if query_start_loc.shape != (num_seqs + 1,):
        raise ValueError("query_start_loc must have shape (num_seqs + 1,)")

    block_size = int(key_cache.shape[1])
    linear_indices = _compute_token_linear_indices(
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        total_tokens=total_tokens,
        block_size=block_size,
    )

    kv_update_call = _get_cute_kv_update_call(
        keys=keys,
        values=values,
        linear_indices=linear_indices,
        key_cache=key_cache,
        value_cache=value_cache,
    )
    new_key_cache, new_value_cache = kv_update_call(
        keys,
        values,
        linear_indices,
        key_cache,
        value_cache,
    )

    out = _run_unified_attention(
        queries=queries,
        key_cache=new_key_cache,
        value_cache=new_value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        alibi_slopes=alibi_slopes,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        seq_threshold_3d=seq_threshold_3d,
        num_par_softmax_segments=num_par_softmax_segments,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    assert out.shape == (total_tokens, num_q_heads, head_dim)
    return out, new_key_cache, new_value_cache
