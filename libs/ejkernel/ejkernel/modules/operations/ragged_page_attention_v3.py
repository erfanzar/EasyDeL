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


"""Ragged Page Attention v3 with fused KV-cache write and automatic optimization.

This module implements **RPA v3**, which combines:

* **Ragged (variable-length) query layout** -- sequences are concatenated
  without padding; boundaries are tracked by ``query_start_loc``.
* **Paged KV cache** -- keys and values are stored in fixed-size pages;
  a block table maps logical page indices to physical page slots.
* **Fused cache update** -- new ``keys``/``values`` tokens are written into
  the KV cache *and* attention is computed against the full (updated) cache
  in the same kernel call.  The updated ``kv_cache`` tensor is returned
  alongside the attention output.

Compared to v2 (``ragged_page_attention_v2``):
    * v2 is read-only: the KV cache must be pre-populated before the call.
    * v3 accepts raw ``keys`` and ``values``, writes them into ``kv_cache``
      via the ``block_tables`` mapping, then computes attention -- enabling
      single-call prefill + decode serving.

Memory Layout:
    Queries: [total_tokens, num_q_heads, head_dim] (ragged, no padding)
    KV Cache: [num_pages, page_size, num_kv_heads_x2_per_kv_packing,
               kv_packing, head_dim_padded]
        Keys and values are interleaved in the ``kv_packing`` dimension.
    Block Tables: [max_num_seqs * pages_per_seq] (flattened 1-D)
    query_start_loc: [max_num_seqs + 1] (inclusive prefix sums)
    distribution: [3] = [num_seqs, pages_per_seq, page_size]

Return Value:
    A tuple ``(attention_output, updated_kv_cache)``:
        * ``attention_output`` has shape [total_tokens, num_q_heads, head_dim].
        * ``updated_kv_cache`` has the same shape as the input ``kv_cache``.

Mathematical Foundation:
    For token i in sequence s:
        start_idx = query_start_loc[s]
        end_idx   = query_start_loc[s + 1]
        output[i] = attention(Q[start_idx:end_idx], K[pages[s]], V[pages[s]])

    where K[pages[s]] and V[pages[s]] now include the newly written tokens.

Platforms:
    GPU: Triton kernel (preferred) or XLA fallback.
    TPU: Pallas kernel.
    Auto: Platform selected from hardware detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import jax
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int32

from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._utils import (
    get_tuned_block_sizes,
    get_tuned_block_sizes_h64,
)
from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import RaggedPageAttentionv3Config


@dataclass(frozen=True)
class _RPAWorkload:
    """Internal dataclass representing workload characteristics for ragged page attention.

    This class encapsulates the key dimensions and data types of a ragged page attention
    workload, used for selecting optimal kernel configurations and autotuning.

    Attributes:
        q_dtype: Data type of query tensors (e.g., jnp.bfloat16, jnp.float32).
        kv_dtype: Data type of KV cache tensors.
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of key-value attention heads (for GQA, num_q_heads % num_kv_heads == 0).
        head_dim: Dimension of each attention head.
        page_size: Number of tokens per page in the paged KV cache.
        max_num_tokens: Maximum number of tokens in the ragged query tensor.
        pages_per_seq: Maximum number of pages allocated per sequence.
    """

    q_dtype: object
    kv_dtype: object
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    max_num_tokens: int
    pages_per_seq: int


_ARGUMENT_ORDER = (
    "queries",
    "keys",
    "values",
    "kv_cache",
    "kv_lens",
    "block_tables",
    "query_start_loc",
    "distribution",
    "softmax_aux",
)
_ARGUMENT_INDEX = {name: idx for idx, name in enumerate(_ARGUMENT_ORDER)}


def _resolve_inv_arg(inv: Invocation, name: str):
    """Resolve an argument from an Invocation by name.

    Looks up arguments first in kwargs, then in positional args based on
    the predefined argument order.

    Args:
        inv: Invocation object containing the function call arguments.
        name: Name of the argument to resolve.

    Returns:
        The resolved argument value.

    Raises:
        KeyError: If the argument is not found in kwargs or positional args.
    """
    if name in inv.kwargs:
        return inv.kwargs[name]
    idx = _ARGUMENT_INDEX.get(name)
    if idx is None or idx >= len(inv.args):
        raise KeyError(name)
    return inv.args[idx]


def _round_up_to_step(value: int, step: int) -> int:
    """Round a value up to the nearest multiple of step.

    Args:
        value: The value to round up.
        step: The step size to round to. If <= 0, returns the value unchanged.

    Returns:
        The smallest multiple of step that is >= value.
    """
    if step <= 0:
        return int(value)
    return ((int(value) + step - 1) // step) * step


def _suggest_block_sizes(workload: _RPAWorkload, aggressive: bool) -> tuple[int, int]:
    """Suggest initial block sizes based on workload characteristics.

    Computes reasonable default block sizes for KV pages and query tokens
    based on the workload dimensions. These serve as starting points for
    autotuning or as fallback heuristic values.

    Args:
        workload: Workload characteristics including head_dim, page_size, etc.
        aggressive: If True, prefer larger block sizes for potentially better
            performance at the cost of higher memory usage.

    Returns:
        Tuple of (num_kv_pages_per_block, num_queries_per_block) representing
        the suggested block sizes.
    """
    pages_per_seq = max(1, workload.pages_per_seq)
    page_size = max(1, workload.page_size)
    page_budget = max(1, 2048 // page_size)

    if page_size <= 64:
        kv_pref = 8 if aggressive else 4
    elif page_size <= 128:
        kv_pref = 6 if aggressive else 3
    elif page_size <= 256:
        kv_pref = 4 if aggressive else 2
    else:
        kv_pref = 2 if aggressive else 1
    kv_pages = max(1, min(pages_per_seq, min(page_budget, kv_pref)))

    token_cap = max(1, min(workload.max_num_tokens, 512))
    if workload.head_dim <= 64:
        q_pref = 96 if aggressive else 64
    elif workload.head_dim <= 128:
        q_pref = 64 if aggressive else 48
    else:
        q_pref = 48 if aggressive else 32
    q_pref = max(8, min(token_cap, q_pref))
    q_pref = max(8, (q_pref // 8) * 8)
    if q_pref > token_cap:
        q_pref = max(8, token_cap)

    return kv_pages, q_pref


def _tpu_kv_candidates(workload: _RPAWorkload) -> tuple[int, ...]:
    """Return TPU-safe KV-page candidates for the workload.

    Wider Gemma4-style heads (256/512) can exceed TPU scoped VMEM when the
    kernel autotuner starts from 16 KV pages per block. Bias those workloads
    toward 8-page blocks first, while keeping broader candidates for narrower
    heads.
    """
    if workload.head_dim >= 256:
        seeds = (8, 4, 2, 1)
    elif workload.head_dim >= 128:
        seeds = (8, 16, 4, 2, 1)
    else:
        seeds = (16, 8, 4, 2, 1)
    kv_limit = max(1, workload.pages_per_seq)
    candidates = tuple(kv for kv in seeds if kv <= kv_limit)
    return candidates or (1,)


def _expand_axis_candidates(
    base: int | None,
    limit: int,
    *,
    seeds: tuple[int, ...],
    min_value: int = 1,
    step: int | None = None,
) -> list[int]:
    """Expand a base value into a list of candidate values for autotuning.

    Generates candidate values by exploring variations around a base value
    (e.g., base/2, base*2) and adding seed values, all constrained to the
    given limits.

    Args:
        base: Initial base value to expand from. If None, only seeds are used.
        limit: Maximum allowed value (candidates will be clamped to this).
        seeds: Tuple of seed values to include as candidates.
        min_value: Minimum allowed value (candidates will be clamped to this).
        step: If provided, candidates will be rounded up to multiples of step.

    Returns:
        List of unique candidate values, ordered by preference.
    """
    limit = int(limit) if limit > 0 else 1
    effective_min = min(max(1, min_value), limit)

    def _quantize(val: float | int) -> int:
        """Round and clamp a value to the nearest valid block size within limits."""
        if step is None or step <= 1:
            quantized = round(val)
        else:
            quantized = _round_up_to_step(round(val), step)
        return max(effective_min, min(limit, max(1, quantized)))

    candidates: list[int] = []

    def _push(val: float | int | None):
        """Quantize and append a candidate value if not already present."""
        if val is None:
            return
        quantized = _quantize(val)
        if quantized not in candidates:
            candidates.append(quantized)

    _push(base)
    if base is not None and base > effective_min:
        _push(base / 2)
    if base is not None and base < limit:
        _push(base * 2)
    for seed in seeds:
        _push(seed)
    _push(limit)
    return candidates


def _is_tpu() -> bool:
    """Check if the current JAX backend is TPU.

    Returns:
        True if the default JAX device is a TPU, False otherwise.
    """
    try:
        return jax.devices()[0].platform == "tpu"
    except Exception:
        return False


def _lookup_tuned_pair(workload: _RPAWorkload) -> tuple[int, int] | None:
    """Look up pre-tuned block sizes for the given workload.

    Attempts to retrieve optimal block sizes from TPU-specific tuning tables
    based on the workload characteristics.

    Args:
        workload: Workload characteristics for the lookup.

    Returns:
        Tuple of (num_kv_pages_per_block, num_queries_per_block) if found,
        None if no tuned configuration exists or not running on TPU.
    """
    if not _is_tpu():
        return None
    try:
        if workload.head_dim == 64:
            bkv, bq = get_tuned_block_sizes_h64(
                workload.q_dtype,
                workload.kv_dtype,
                workload.num_q_heads,
                workload.num_kv_heads,
                workload.head_dim,
                workload.page_size,
                workload.max_num_tokens,
                workload.pages_per_seq,
            )
        else:
            bkv, bq = get_tuned_block_sizes(
                workload.q_dtype,
                workload.kv_dtype,
                workload.num_q_heads,
                workload.num_kv_heads,
                workload.head_dim,
                workload.page_size,
                workload.max_num_tokens,
                workload.pages_per_seq,
            )
        if bkv <= 0 or bq <= 0:
            return None
        return int(bkv), int(bq)
    except (KeyError, ValueError, RuntimeError):
        return None


class RaggedPageAttentionv3(Kernel[RaggedPageAttentionv3Config, tuple[Array, Array]]):
    """Ragged Page Attention v3 with fused KV-cache write.

    Combines variable-length ragged query layout with paged KV cache management
    and a *fused cache update*: new ``keys``/``values`` tokens are written into
    the ``kv_cache`` at the correct paged positions, and attention is computed
    against the full updated cache in the same call.

    Returns a tuple ``(attention_output, updated_kv_cache)`` where:
        * ``attention_output`` has shape [total_tokens, num_q_heads, head_dim].
        * ``updated_kv_cache`` has the same shape as the input ``kv_cache``.

    Features:
        - Zero padding overhead through ragged query layout
        - Fused KV-cache write and attention in a single kernel
        - Paged KV cache with dynamic block allocation
        - Support for variable context lengths per sequence
        - Sliding window attention for long contexts
        - Logit soft capping for numerical stability
        - Optional attention sink mechanism for long-context stability
        - Quantization support via per-tensor q/k/v scale factors
        - Configurable block sizes and VMEM limits (TPU)
        - Multiple platform support (Triton/Pallas/XLA)

    This implementation is particularly efficient for:
        - LLM serving with continuous batching and single-call prefill+decode
        - Variable-length inference workloads
        - Memory-constrained deployment requiring paged cache management
    """

    version = "1"

    def __init__(self):
        """Initialize Ragged Page Attention module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="ragged_page_attention_v3")

    def create_shard_map_wrapper(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        keys: Float[Array, "total_tokens num_kv_heads head_dim"],
        values: Float[Array, "total_tokens num_kv_heads head_dim"],
        kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
        kv_lens: Int32[Array, "max_num_seqs"],
        block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
        query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
        distribution: Int32[Array, "3"],
        softmax_aux: Float[Array, "num_q_heads"] | None = None,
        softmax_scale: float = 1.0,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        vmem_limit_bytes: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedPageAttentionv3Config | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | tuple[PartitionSpec, PartitionSpec] | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed ragged page attention.

        This method creates a sharded computation wrapper that distributes ragged page
        attention across multiple devices using JAX's shard_map. It enables efficient
        parallel execution while handling variable-length sequences and paged KV cache.

        Args:
            queries: Ragged query tensor [total_tokens, num_q_heads, head_dim].
                All sequences concatenated without padding.
            keys: Key tensor [total_tokens, num_kv_heads, head_dim] for cache updates.
            values: Value tensor [total_tokens, num_kv_heads, head_dim] for cache updates.
            kv_cache: Paged KV cache [num_pages, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded].
                Contains interleaved keys and values in paged format.
            kv_lens: Context lengths for each sequence [max_num_seqs].
                Indicates how many tokens are valid in each sequence's KV cache.
            block_tables: Block table mapping [max_num_seqs_times_pages_per_seq].
                Maps logical pages to physical page indices in kv_cache.
            query_start_loc: Start indices for each sequence [max_num_seqs_plus_1].
                query_start_loc[i] to query_start_loc[i+1] defines sequence i's token range.
            distribution: Distribution parameters [3] containing:
                [num_seqs, pages_per_seq, page_size] for kernel execution.
            softmax_scale: Scaling factor applied to attention scores before softmax.
                Default is 1.0, typically set to 1/sqrt(head_dim).
            sliding_window: Optional window size for sliding window attention.
                If None, uses full attention. Otherwise limits attention to sliding_window tokens.
            logits_soft_cap: Optional soft capping value for attention logits.
                Helps prevent extreme values and improves numerical stability.
            q_scale: Optional scaling factor for queries in quantized attention.
            k_scale: Optional scaling factor for keys in quantized attention.
            v_scale: Optional scaling factor for values in quantized attention.
            vmem_limit_bytes: Memory limit for vector memory in bytes (TPU-specific).
                Controls memory usage on TPU accelerators.
            platform: Target platform ("triton", "pallas", "cuda", "xla", "auto").
                If None, uses platform from cfg or auto-detection.
            cfg: Kernel configuration containing block sizes and tuning parameters.
                If None, uses default configuration.
            mesh: JAX device mesh defining the device topology for sharding.
                Must be provided for shard_map execution.
            in_specs: Tuple of PartitionSpec objects defining input tensor sharding.
                Must match the number of input arguments (queries, keys, values, kv_cache,
                kv_lens, block_tables, query_start_loc, distribution).
            out_specs: PartitionSpec defining output tensor sharding.
                Must be provided for shard_map execution.
            check_vma: Whether to check virtual memory alignment in shard_map.
                Default is False.

        Returns:
            Tuple containing:
                - shard_map_fn: The shard_mapped attention function ready for execution
                - call_args: Tuple of arguments to pass to shard_map_fn

        Raises:
            AssertionError: If mesh, in_specs, or out_specs is None.
            AssertionError: If length of in_specs doesn't match number of call arguments.

        Note:
            The shard_map wrapper enables efficient parallel execution of ragged page
            attention across multiple devices while maintaining the ragged layout and
            paged cache structure. This is essential for scaling to large batch sizes
            and long contexts in distributed serving scenarios.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_ragged_page_attn(
            queries: Float[Array, "total_tokens num_q_heads head_dim"],
            keys: Float[Array, "total_tokens num_kv_heads head_dim"],
            values: Float[Array, "total_tokens num_kv_heads head_dim"],
            kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
            kv_lens: Int32[Array, "max_num_seqs"],
            block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
            query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
            distribution: Int32[Array, "3"],
            softmax_aux: Float[Array, "num_q_heads"] | None,
        ) -> tuple[
            Float[Array, "total_tokens num_q_heads head_dim"],
            Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
        ]:
            """Shard-map compatible wrapper that delegates to self.run with captured params."""
            return self.run(
                queries=queries,
                keys=keys,
                values=values,
                kv_cache=kv_cache,
                kv_lens=kv_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                distribution=distribution,
                softmax_aux=softmax_aux,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                vmem_limit_bytes=vmem_limit_bytes,
                platform=platform,
                cfg=cfg,
            )

        call_args = (
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            block_tables,
            query_start_loc,
            distribution,
            softmax_aux,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_ragged_page_attn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, call_args

    def get_impl(self, cfg: RaggedPageAttentionv3Config):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for ragged page attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        platform = detect_platform("ragged_page_attention_v3", cfg.platform, prefer_cuda=True)
        return kernel_registry.get("ragged_page_attention_v3", platform=platform, backend=cfg.backend)

    def run(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        keys: Float[Array, "total_tokens num_kv_heads head_dim"],
        values: Float[Array, "total_tokens num_kv_heads head_dim"],
        kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
        kv_lens: Int32[Array, "max_num_seqs"],
        block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
        query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
        distribution: Int32[Array, "3"],
        softmax_aux: Float[Array, "num_q_heads"] | None = None,
        softmax_scale: float = 1.0,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        vmem_limit_bytes: int | None = None,
        *,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedPageAttentionv3Config,
    ) -> tuple[
        Float[Array, "total_tokens num_q_heads head_dim"],
        Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
    ]:
        """Execute ragged page attention over variable-length sequences.

        Computes attention where queries are in ragged (concatenated) format
        and KV cache is organized in pages, providing maximum memory efficiency
        for serving workloads with variable-length sequences.

        Args:
            queries: Ragged query tensor [total_tokens, num_q_heads, head_dim].
                All sequences concatenated without padding. The start position of
                each sequence is defined by query_start_loc.
            keys: Key tensor [total_tokens, num_kv_heads, head_dim] used for
                updating the KV cache with new tokens.
            values: Value tensor [total_tokens, num_kv_heads, head_dim] used for
                updating the KV cache with new tokens.
            kv_cache: Paged KV cache [num_pages, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded].
                Contains interleaved keys and values in paged format. Physical pages
                are mapped via block_tables.
            kv_lens: Context lengths for each sequence [max_num_seqs].
                Specifies how many tokens are valid in each sequence's KV cache.
            block_tables: Block table mapping [max_num_seqs_times_pages_per_seq].
                Maps logical page indices to physical page indices in kv_cache.
                Flattened from shape [max_num_seqs, pages_per_seq].
            query_start_loc: Start indices for each sequence [max_num_seqs_plus_1].
                query_start_loc[i] to query_start_loc[i+1] defines the token range
                for sequence i in the ragged queries tensor.
            distribution: Three-element vector ``[num_seqs, pages_per_seq,
                page_size]`` packed into shape ``[3]``, dtype ``int32``.
            softmax_scale: Scaling factor applied to attention scores before softmax.
                Default is 1.0, typically set to 1/sqrt(head_dim) for stability.
            sliding_window: Optional window size for sliding window attention.
                If None, uses full attention over all context. If set, limits
                attention to the last sliding_window tokens.
            logits_soft_cap: Optional soft capping value for attention logits.
                Applies tanh-based soft capping to prevent extreme values and
                improve numerical stability.
            q_scale: Optional scaling factor for queries in quantized attention.
                Used when queries are quantized to lower precision.
            k_scale: Optional scaling factor for keys in quantized attention.
                Used when keys are quantized to lower precision.
            v_scale: Optional scaling factor for values in quantized attention.
                Used when values are quantized to lower precision.
            vmem_limit_bytes: Memory limit for vector memory in bytes (TPU-specific).
                Controls VMEM usage on TPU accelerators for large head dimensions.
            platform: Optional platform override ("triton", "pallas", "cuda", "xla", "auto").
                If provided, overrides the platform specified in cfg.
            cfg: Kernel configuration object containing:
                - num_kv_pages_per_block: Number of KV pages processed per block
                - num_queries_per_block: Number of query tokens processed per block
                - num_warps: Number of warps for Triton kernels
                - num_stages: Number of pipeline stages for Triton kernels
                - platform: Target platform
                - backend: Backend specification

        Returns:
            Two-element tuple:
                * Attention output [total_tokens, num_q_heads, head_dim]
                  in ragged format, with the same layout as ``queries``.
                * Updated KV cache with newly written key-value pairs,
                  same shape as the input ``kv_cache``.

        Note:
            The ragged format eliminates all padding overhead by concatenating sequences
            without padding. Combined with paged KV cache, this provides the most
            memory-efficient attention implementation for serving workloads with
            variable-length sequences. The paged cache also enables memory sharing
            across sequences for features like beam search and prefix caching.

        Example:
            >>> import jax.numpy as jnp
            >>>
            >>> queries = jnp.ones((25, 32, 128))
            >>> keys = jnp.ones((25, 8, 128))
            >>> values = jnp.ones((25, 8, 128))
            >>> kv_cache = jnp.zeros((100, 16, 16, 2, 128))
            >>> kv_lens = jnp.array([10, 15])
            >>> block_tables = jnp.arange(200).reshape(-1)
            >>> query_start_loc = jnp.array([0, 10, 25])
            >>> distribution = jnp.array([2, 100, 16])
            >>>
            >>> kernel = RaggedPageAttentionv3()
            >>> cfg = RaggedPageAttentionv3Config(platform="auto")
            >>> out = kernel.run(
            ...     queries, keys, values, kv_cache, kv_lens,
            ...     block_tables, query_start_loc, distribution,
            ...     softmax_scale=0.0883883476483184,
            ...     cfg=cfg
            ... )
            >>> out.shape
        """

        if platform is not None:
            cfg = RaggedPageAttentionv3Config(
                num_kv_pages_per_block=cfg.num_kv_pages_per_block,
                num_queries_per_block=cfg.num_queries_per_block,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            distribution=distribution,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            vmem_limit_bytes=vmem_limit_bytes,
            chunk_prefill_size=cfg.chunk_prefill_size,
            num_kv_pages_per_block=cfg.num_kv_pages_per_block,
            num_queries_per_block=cfg.num_queries_per_block,
        )

    def _extract_workload(self, inv: Invocation[RaggedPageAttentionv3Config, Array]) -> _RPAWorkload | None:
        """Extract workload characteristics from an invocation.

        Parses the input tensors from the invocation to determine the workload
        dimensions needed for configuration selection.

        Args:
            inv: Invocation object containing the attention arguments.

        Returns:
            _RPAWorkload with extracted dimensions, or None if extraction fails.
        """
        try:
            queries = _resolve_inv_arg(inv, "queries")
            keys = _resolve_inv_arg(inv, "keys")
            kv_cache = _resolve_inv_arg(inv, "kv_cache")
            kv_lens = _resolve_inv_arg(inv, "kv_lens")
            block_tables = _resolve_inv_arg(inv, "block_tables")
        except KeyError:
            return None

        try:
            max_num_tokens = int(queries.shape[0])
            num_q_heads = int(queries.shape[1])
            head_dim = int(queries.shape[2])
            num_kv_heads = int(keys.shape[1])
            page_size = int(kv_cache.shape[1])
            kv_lens_dim = int(kv_lens.shape[0])
            block_table_dim = int(block_tables.shape[0])
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

        if kv_lens_dim <= 0 or block_table_dim <= 0:
            return None
        pages_per_seq = block_table_dim // kv_lens_dim
        if pages_per_seq <= 0 or max_num_tokens <= 0:
            return None

        return _RPAWorkload(
            q_dtype=getattr(queries, "dtype", None),
            kv_dtype=getattr(kv_cache, "dtype", None),
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            max_num_tokens=max_num_tokens,
            pages_per_seq=pages_per_seq,
        )

    def _candidate_pairs(
        self,
        inv: Invocation[RaggedPageAttentionv3Config, Array],
        *,
        prefer_tuned: bool,
        max_candidates: int,
    ) -> list[tuple[int, int]]:
        """Generate candidate (kv_pages, query_tokens) block size pairs.

        Combines tuned values (if available) with heuristic-based suggestions
        to produce a list of block size pairs for autotuning.

        Args:
            inv: Invocation object containing the attention arguments.
            prefer_tuned: If True, prioritize pre-tuned block sizes from lookup tables.
            max_candidates: Maximum number of candidate pairs to return.

        Returns:
            List of (num_kv_pages_per_block, num_queries_per_block) tuples.
        """
        workload = self._extract_workload(inv)
        if workload is None:
            return []

        tuned_pair = _lookup_tuned_pair(workload) if prefer_tuned else None
        base_pair = tuned_pair or _suggest_block_sizes(workload, aggressive=prefer_tuned)
        pairs = self._generate_candidate_pairs(workload, base_pair, max_candidates=max_candidates)
        if tuned_pair and tuned_pair not in pairs:
            pairs.insert(0, tuned_pair)
            pairs = pairs[:max_candidates]
        return pairs

    def _generate_candidate_pairs(
        self,
        workload: _RPAWorkload,
        base_pair: tuple[int, int],
        *,
        max_candidates: int,
    ) -> list[tuple[int, int]]:
        """Generate candidate block size pairs from a base configuration.

        Expands the base pair into multiple candidates by exploring variations
        along both the KV and query axes, respecting workload constraints.

        Args:
            workload: Workload characteristics for constraint checking.
            base_pair: Base (kv_pages, query_tokens) pair to expand from.
            max_candidates: Maximum number of pairs to generate.

        Returns:
            List of (num_kv_pages_per_block, num_queries_per_block) tuples.
        """
        base_kv, base_q = base_pair
        kv_limit = max(1, workload.pages_per_seq)
        kv_seed_tuple = _tpu_kv_candidates(workload) if _is_tpu() else (16, 32, 64, 128, 256)
        kv_candidates = _expand_axis_candidates(
            base_kv,
            kv_limit,
            seeds=kv_seed_tuple,
            min_value=1,
            step=None,
        )
        query_limit = max(1, min(workload.max_num_tokens, 512))
        q_seeds = (4, 8, 16, 32, 64) if _is_tpu() else (32, 64, 128, 256)
        q_candidates = _expand_axis_candidates(
            base_q,
            query_limit,
            seeds=q_seeds,
            min_value=1,
            step=4 if _is_tpu() else 16,
        )

        pairs: list[tuple[int, int]] = []
        for kv in kv_candidates:
            for qb in q_candidates:
                if kv <= 0 or qb <= 0:
                    continue
                pair = (kv, qb)
                if pair in pairs:
                    continue
                pairs.append(pair)
                if len(pairs) >= max_candidates:
                    return pairs
        if _is_tpu():
            allowed_kv = set(kv_seed_tuple)
            allowed_q = {4, 8, 16, 32, 64}
            filtered = [(kv, qb) for kv, qb in pairs if kv in allowed_kv and qb in allowed_q]
            if filtered:
                pairs = filtered[:max_candidates]
        return pairs

    def _materialize_configs(
        self,
        pairs: list[tuple[int, int]],
        *,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"],
        backend: Backend | Literal["any"],
        num_warps: int | None,
        num_stages: int | None,
    ) -> list[RaggedPageAttentionv3Config]:
        """Convert block size pairs into configuration objects.

        Creates RaggedPageAttentionv3Config objects from (kv_pages, query_tokens)
        pairs, applying the specified platform and backend settings.

        Args:
            pairs: List of (num_kv_pages_per_block, num_queries_per_block) tuples.
            platform: Target platform for kernel execution.
            backend: Backend specification (e.g., "gpu", "tpu", "any").
            num_warps: Number of warps for Triton kernels (GPU-specific).
            num_stages: Number of pipeline stages for Triton kernels (GPU-specific).

        Returns:
            List of RaggedPageAttentionv3Config objects ready for autotuning.
        """
        if not pairs:
            return [
                RaggedPageAttentionv3Config(
                    num_kv_pages_per_block=None,
                    num_queries_per_block=None,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform=platform,
                    backend=backend,
                )
            ]

        configs: list[RaggedPageAttentionv3Config] = []
        for kv_pages, q_tokens in pairs:
            configs.append(
                RaggedPageAttentionv3Config(
                    num_kv_pages_per_block=kv_pages,
                    num_queries_per_block=q_tokens,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform=platform,
                    backend=backend,
                )
            )
        return configs

    def _build_candidate_configs(
        self,
        inv: Invocation[RaggedPageAttentionv3Config, Array],
        *,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"],
        backend: Backend | Literal["any"],
        num_warps: int | None,
        num_stages: int | None,
        prefer_tuned: bool,
        max_candidates: int,
    ) -> list[RaggedPageAttentionv3Config]:
        """Build candidate configurations for autotuning from an invocation.

        Combines pair generation and materialization to produce a complete list
        of configuration candidates for the given invocation.

        Args:
            inv: Invocation object containing the attention arguments.
            platform: Target platform for kernel execution.
            backend: Backend specification (e.g., "gpu", "tpu", "any").
            num_warps: Number of warps for Triton kernels (GPU-specific).
            num_stages: Number of pipeline stages for Triton kernels (GPU-specific).
            prefer_tuned: If True, prioritize pre-tuned block sizes.
            max_candidates: Maximum number of configurations to generate.

        Returns:
            List of RaggedPageAttentionv3Config objects for autotuning.
        """
        return self._materialize_configs(
            self._candidate_pairs(inv, prefer_tuned=prefer_tuned, max_candidates=max_candidates),
            platform=platform,
            backend=backend,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def heuristic_cfg(self, inv: Invocation[RaggedPageAttentionv3Config, Array]) -> RaggedPageAttentionv3Config:
        """Provide default configuration optimized for ragged page attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with conservative block sizes suitable for
            typical ragged attention workloads with variable sequence lengths
        """
        pairs = self._candidate_pairs(inv, prefer_tuned=True, max_candidates=1)
        kv_block = pairs[0][0] if pairs else None
        q_block = pairs[0][1] if pairs else None
        return RaggedPageAttentionv3Config(
            num_kv_pages_per_block=kv_block,
            num_queries_per_block=q_block,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[RaggedPageAttentionv3Config, Array]):
        """Generate candidate configurations for autotuning.

        Creates configurations optimized for ragged attention scenarios with
        various batch sizes and sequence lengths.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Ragged attention performance depends on the distribution of sequence
            lengths and the page size. Candidates are chosen to work well across
            common serving scenarios.
        """

        return self._build_candidate_configs(
            inv,
            platform="auto",
            backend="any",
            num_warps=None,
            num_stages=None,
            prefer_tuned=True,
            max_candidates=6,
        )

    def candidate_cfgs_gpu(self, inv: Invocation[RaggedPageAttentionv3Config, Array]):
        """Generate GPU candidates for every registered GPU backend.

        Automatic platform selection should benchmark Triton, CUDA, TileLang,
        and XLA candidates at the operation level.  An explicit public
        ``platform=...`` request narrows the list to that backend.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("cuda", "triton", "tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[RaggedPageAttentionv3Config] = []
        if "cuda" in platforms:
            candidates.extend(
                self._build_candidate_configs(
                    inv,
                    platform="cuda",
                    backend="gpu",
                    num_warps=None,
                    num_stages=None,
                    prefer_tuned=True,
                    max_candidates=2,
                )
            )
        if "triton" in platforms:
            candidates.extend(
                self._build_candidate_configs(
                    inv,
                    platform="triton",
                    backend="gpu",
                    num_warps=None,
                    num_stages=None,
                    prefer_tuned=True,
                    max_candidates=10,
                )
            )
        if "tilelang" in platforms:
            candidates.extend(
                self._build_candidate_configs(
                    inv,
                    platform="tilelang",
                    backend="gpu",
                    num_warps=4,
                    num_stages=1,
                    prefer_tuned=True,
                    max_candidates=6,
                )
            )
        if "xla" in platforms:
            candidates.extend(
                self._build_candidate_configs(
                    inv,
                    platform="xla",
                    backend="any",
                    num_warps=None,
                    num_stages=None,
                    prefer_tuned=True,
                    max_candidates=6,
                )
            )
        return candidates or self._build_candidate_configs(
            inv,
            platform="xla",
            backend="any",
            num_warps=None,
            num_stages=None,
            prefer_tuned=True,
            max_candidates=1,
        )

    def candidate_cfgs_tpu(self, inv: Invocation[RaggedPageAttentionv3Config, Array]):
        """Generate candidate configurations for autotuning on TPU (Pallas backend).

        Produces configurations optimized for TPU execution with Pallas backend.
        Uses power-of-2 block sizes within strict bounds to ensure efficient
        TPU tile execution.

        Heuristics:
            - Enforces power-of-2 block sizes for TPU efficiency
            - Restricts Q and KV block sizes to max 64
            - Minimum KV block size: 16, minimum Q block size: 4
            - chunk_prefill_size always set to None

        Args:
            inv: Invocation object containing input tensors and metadata.
                Used to determine optimal block sizes based on workload shape
                including queries, keys, kv_cache, kv_lens, and block_tables.

        Returns:
            List of RaggedPageAttentionv3Config objects optimized for TPU.
            Each configuration uses power-of-2 block sizes within the
            specified bounds. Returns empty list if workload extraction fails.

        Note:
            KV page candidates: 16, 32, 64 (bounded by pages_per_seq)
            Query candidates: 4, 8, 16, 32, 64 (bounded by max_num_tokens, capped at 512)
        """
        workload = self._extract_workload(inv)
        if workload is None:
            return []

        kv_candidates = list(_tpu_kv_candidates(workload))
        q_candidates = [4, 8, 16, 32, 64]

        kv_limit = max(1, workload.pages_per_seq)
        q_limit = max(1, min(workload.max_num_tokens, 512))

        pairs = []
        for kv in kv_candidates:
            if kv > kv_limit:
                continue
            for q in q_candidates:
                if q > q_limit:
                    continue
                pairs.append((kv, q))

        if not pairs:
            pairs.append((16, 4))

        return [
            *self._materialize_configs(
                pairs,
                platform="pallas",
                backend="tpu",
                num_warps=None,
                num_stages=None,
            ),
            *self._materialize_configs(
                pairs[:6],
                platform="xla",
                backend="any",
                num_warps=None,
                num_stages=None,
            ),
        ]

    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu
    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu


_ragged_page_attention_executor: Executor[RaggedPageAttentionv3Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("ragged-page-attention-v3"),
    )
)


def ragged_page_attention_v3(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    /,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    vmem_limit_bytes: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RaggedPageAttentionv3Config | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | tuple[PartitionSpec, PartitionSpec] | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
]:
    """Execute ragged page attention v3 with automatic optimization and optional sharding.

    This is the main entry point for ragged page attention v3, which combines variable-length
    (ragged) sequence processing with paged KV cache management. It provides the most
    memory-efficient attention implementation for LLM serving workloads by eliminating
    padding overhead while enabling flexible memory management through paged caching.

    The function automatically selects and caches optimal kernel configurations based on
    the input shapes and hardware platform. It supports both single-device execution and
    distributed execution via JAX's shard_map when mesh and partition specs are provided.

    Key Features:
        - Zero padding overhead through ragged layout
        - Efficient paged KV cache with flexible memory allocation
        - Automatic kernel selection and autotuning
        - Support for sliding window attention
        - Logit soft capping for numerical stability
        - Quantization support (q_scale, k_scale, v_scale)
        - Multi-platform support (Triton/Pallas/CUDA/XLA)
        - Distributed execution via shard_map
        - Persistent configuration caching

    Args:
        queries: Ragged query tensor [total_tokens, num_q_heads, head_dim].
            All sequences concatenated without padding. Sequence boundaries
            are defined by query_start_loc.
        keys: Key tensor [total_tokens, num_kv_heads, head_dim] for updating
            the KV cache with new tokens.
        values: Value tensor [total_tokens, num_kv_heads, head_dim] for updating
            the KV cache with new tokens.
        kv_cache: Paged KV cache [num_pages, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded].
            Contains interleaved keys and values in paged format. The cache is
            organized as fixed-size pages that are dynamically allocated.
        kv_lens: Context lengths for each sequence [max_num_seqs].
            Indicates how many tokens are valid in each sequence's KV cache.
        block_tables: Block table mapping [max_num_seqs_times_pages_per_seq].
            Maps logical page indices to physical page indices in kv_cache.
            Flattened from shape [max_num_seqs, pages_per_seq].
        query_start_loc: Start indices for each sequence [max_num_seqs_plus_1].
            query_start_loc[i] to query_start_loc[i+1] defines the token range
            for sequence i in the ragged queries tensor. The last element equals
            total_tokens.
        distribution: Three-element vector ``[num_seqs, pages_per_seq,
            page_size]`` packed into shape ``[3]``, dtype ``int32``.
        softmax_scale: Scaling factor applied to attention scores before softmax.
            Default is 1.0, typically set to 1/sqrt(head_dim) for numerical stability.
        sliding_window: Optional window size for sliding window attention.
            If None, uses full attention over all context. If set to a positive integer,
            limits attention to the last sliding_window tokens for each query.
        logits_soft_cap: Optional soft capping value for attention logits.
            Applies soft capping as: logits_soft_cap * tanh(logits / logits_soft_cap)
            to prevent extreme values and improve numerical stability.
        q_scale: Optional scaling factor for queries in quantized attention.
            Used to dequantize queries when they are stored in lower precision.
        k_scale: Optional scaling factor for keys in quantized attention.
            Used to dequantize keys when they are stored in lower precision.
        v_scale: Optional scaling factor for values in quantized attention.
            Used to dequantize values when they are stored in lower precision.
        vmem_limit_bytes: Memory limit for vector memory in bytes (TPU-specific).
            Controls VMEM usage on TPU accelerators, particularly important for
            large head dimensions (e.g., 256).
        platform: Target platform for kernel execution.
            One of "triton" (GPU), "pallas" (TPU), "cuda" (GPU), "xla" (fallback),
            or "auto" (automatic detection). If None, uses platform from cfg or
            auto-detection.
        cfg: Optional kernel configuration object. If None, uses automatic configuration
            selection with autotuning. Can specify:
            - num_kv_pages_per_block: Number of KV pages processed per block
            - num_queries_per_block: Number of query tokens processed per block
            - num_warps: Number of warps for Triton kernels (GPU-specific)
            - num_stages: Number of pipeline stages for Triton kernels (GPU-specific)
            - platform: Target platform
            - backend: Backend specification
        mesh: Optional JAX device mesh for distributed execution.
            If provided along with in_specs and out_specs, executes using shard_map
            for multi-device parallelism.
        in_specs: Optional tuple of PartitionSpec objects defining input tensor sharding.
            Must be provided if mesh is specified. Should contain specs for all 9 inputs:
            (queries, keys, values, kv_cache, kv_lens, block_tables, query_start_loc,
            distribution, softmax_aux).
        out_specs: Optional PartitionSpec defining output tensor sharding.
            Must be provided if mesh is specified.

    Returns:
        Tuple containing:
            - Attention output [total_tokens, num_q_heads, head_dim]: Attention-weighted
              combination of values in ragged format, maintaining the same layout as queries.
            - Updated KV cache [num_pages, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded]:
              The KV cache with new key-value pairs incorporated.

    Raises:
        ValueError: If no suitable kernel implementation is found for the platform.
        AssertionError: If mesh is provided without in_specs or out_specs.
        AssertionError: If in_specs length doesn't match number of input arguments.

    Note:
        Performance Characteristics:
            - Ragged layout eliminates padding overhead, saving memory proportional to
              sequence length variance
            - Paged cache enables memory sharing for beam search and prefix caching
            - Automatic configuration caching avoids re-tuning for similar workloads
            - Sliding window attention reduces complexity from O(n²) to O(n*w) where w
              is the window size

        Memory Layout:
            - Queries: Ragged format [total_tokens, ...] with no padding between sequences
            - KV Cache: Paged format [num_pages, page_size, ...] with page-level granularity
            - Block Tables: Maps logical sequence pages to physical cache pages

        Distributed Execution:
            When mesh, in_specs, and out_specs are provided, the function uses JAX's
            shard_map to distribute computation across devices. This is essential for:
            - Large batch sizes that don't fit on a single device
            - Long contexts requiring memory distribution
            - Multi-node inference serving

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from ejkernel.modules.operations import ragged_page_attention_v3
        >>>
        >>>
        >>> queries = jnp.ones((25, 32, 128), dtype=jnp.bfloat16)
        >>> keys = jnp.ones((25, 8, 128), dtype=jnp.bfloat16)
        >>> values = jnp.ones((25, 8, 128), dtype=jnp.bfloat16)
        >>> kv_cache = jnp.zeros((100, 16, 16, 2, 128), dtype=jnp.bfloat16)
        >>> kv_lens = jnp.array([10, 15], dtype=jnp.int32)
        >>> block_tables = jnp.arange(200, dtype=jnp.int32)
        >>> query_start_loc = jnp.array([0, 10, 25], dtype=jnp.int32)
        >>> distribution = jnp.array([2, 100, 16], dtype=jnp.int32)
        >>>
        >>>
        >>> output, updated_cache = ragged_page_attention_v3(
        ...     queries, keys, values, kv_cache, kv_lens,
        ...     block_tables, query_start_loc, distribution,
        ...     softmax_scale=1.0 / jnp.sqrt(128.0),
        ...     sliding_window=2048,
        ...     logits_soft_cap=30.0,
        ... )
        >>> output.shape
        >>> updated_cache.shape
        >>>
        >>>
        >>> devices = jax.devices()
        >>> mesh = Mesh(devices, axis_names=('data',))
        >>> P = PartitionSpec
        >>> output, updated_cache = ragged_page_attention_v3(
        ...     queries, keys, values, kv_cache, kv_lens,
        ...     block_tables, query_start_loc, distribution,
        ...     mesh=mesh,
        ...     in_specs=(P('data'), P('data'), P('data'), P(None), P('data'), P('data'), P('data'), P(None)),
        ...     out_specs=P('data'),
        ... )

    See Also:
        RaggedPageAttentionv3: The kernel class implementing the attention operation.
        RaggedPageAttentionv3Config: Configuration class for kernel parameters.
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    return _ragged_page_attention_executor(
        RaggedPageAttentionv3(),
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )
