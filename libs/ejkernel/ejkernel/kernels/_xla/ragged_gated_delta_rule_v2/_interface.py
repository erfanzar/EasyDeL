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

"""Public XLA interface and registry entry for the ragged GDN v2 kernel.

Defines the typed, registry-decorated ``ragged_gated_delta_rule_v2`` entry
point. The function is registered for ``(Platform.XLA, Backend.ANY)`` so the
kernel dispatcher can resolve it on any XLA backend (GPU/TPU/CPU), wrapped with
``jaxtyping``/``beartype`` shape-and-dtype checking, and forwards all arguments
unchanged to the underlying XLA forward implementation in ``_xla_impl_fwd``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import ragged_gated_delta_rule_v2 as _ragged_gated_delta_rule_v2_xla


@kernel_registry.register("ragged_gated_delta_rule_v2", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_gated_delta_rule_v2(
    mixed_qkv: Float[Array, "num_tokens mixed_dim"],
    b: Float[Array, "num_tokens num_value_heads"],
    a: Float[Array, "num_tokens num_value_heads"],
    recurrent_state: Float[Array, "num_slots num_value_heads qk_head_dim v_head_dim"],
    A_log: Float[Array, "num_value_heads"],
    dt_bias: Float[Array, "num_value_heads"],
    query_start_loc: Int[Array, "num_requests_plus_1"],
    state_indices: Int[Array, "num_requests"],
    distribution: Int[Array, "three"],
    has_initial_state: Bool[Array, "num_requests"] | None = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = True,
    apply_silu_in_gdr: bool = False,
    use_recurrent_scan_prefill: bool = False,
    mask_initial_state: bool = False,
    kernel_tile_policy: str = "auto",
    use_fused_gdn_decode: bool = False,
    runtime_dtype: object | None = None,
) -> tuple[
    Float[Array, "num_slots num_value_heads qk_head_dim v_head_dim"],
    Float[Array, "num_tokens out_dim"],
]:
    """Run the packed-inference ragged Gated Delta Rule (GDN) v2 XLA kernel.

    Registered entry point for ``(Platform.XLA, Backend.ANY)``. It applies the
    Gated Delta Rule recurrence over a ragged, continuous-batching token buffer
    where many requests of heterogeneous lengths share one contiguous stream.
    Inputs are jaxtyping/beartype checked, then forwarded verbatim to the
    underlying XLA forward implementation, which casts everything to a runtime
    dtype and routes to either the decode-only fast path (all requests advance
    by one token) or the chunked mixed-prefill path (some request has more than
    one new token).

    Args:
        mixed_qkv: Interleaved query/key/value projections in a single flat
            feature axis, shape ``(num_tokens, mixed_dim)`` where
            ``mixed_dim == 2 * n_kq * d_k + n_v * d_v``. The first
            ``n_kq * d_k`` features are queries, the next ``n_kq * d_k`` are
            keys, and the remaining ``n_v * d_v`` are values.
        b: Pre-sigmoid beta (gating) source, shape
            ``(num_tokens, num_value_heads)``; sigmoided inside the kernel.
        a: Pre-softplus alpha (decay) source, shape
            ``(num_tokens, num_value_heads)``; combined with ``A_log`` and
            ``dt_bias`` to form the log-space decay.
        recurrent_state: Global recurrent state pool, shape
            ``(num_slots, num_value_heads, qk_head_dim, v_head_dim)``. Slot 0
            is the null block reserved for padded/invalid tokens. Donated and
            updated in place by the wrapped kernel.
        A_log: Per-value-head log-decay parameter, shape
            ``(num_value_heads,)``.
        dt_bias: Per-value-head delta-time bias, shape ``(num_value_heads,)``.
        query_start_loc: Cumulative per-request token offsets describing
            request boundaries in the packed stream, shape
            ``(num_requests + 1,)``.
        state_indices: Per-request mapping into ``recurrent_state``, shape
            ``(num_requests,)``.
        distribution: ``int32`` triple ``(decode_end, prefill_end, total)``;
            ``decode_end == total`` selects the decode-only branch and
            ``total`` gates which slots have valid outputs/state writes.
        has_initial_state: Optional boolean flags per request indicating which
            carry a non-empty initial recurrent state, shape
            ``(num_requests,)``. Defaults to all-``True`` when ``None``;
            currently only consumed for that default.
        n_kq: Number of key/query heads before GQA-style head expansion.
        n_v: Number of value heads after head expansion (``num_value_heads``).
        d_k: Per-head key/query dimension (``qk_head_dim``).
        d_v: Per-head value dimension (``v_head_dim``).
        chunk_size: Padding/chunking granularity for the mixed-prefill branch.
            Defaults to 64.
        use_qk_norm_in_gdn: Whether to L2-normalize queries and keys before the
            recurrence. Defaults to True.
        apply_silu_in_gdr: When True, applies SiLU to ``mixed_qkv`` before it is
            split into Q/K/V. Defaults to False.
        use_recurrent_scan_prefill: Static flag accepted for API/signature
            compatibility; not consumed by the wrapped implementation. Defaults
            to False.
        mask_initial_state: If ``True``, use ``has_initial_state`` to zero stale
            recurrent slots for fresh prefill requests. Defaults to ``False``.
        kernel_tile_policy: Accepted for API compatibility with TPU Pallas;
            ignored by the XLA implementation.
        use_fused_gdn_decode: Accepted for API compatibility with TPU Pallas;
            ignored by the XLA implementation.
        runtime_dtype: Optional dtype that all float inputs are cast to before
            the recurrence. When ``None``, ``mixed_qkv.dtype`` is used.

    Returns:
        tuple: ``(updated_recurrent_state, output)`` where
        ``updated_recurrent_state`` has shape
        ``(num_slots, num_value_heads, qk_head_dim, v_head_dim)`` and
        ``output`` has shape ``(num_tokens, out_dim)`` with
        ``out_dim == n_v * d_v``. Rows/slots for tokens or requests beyond
        ``distribution[2]`` are left zeroed / unchanged.
    """
    _ = kernel_tile_policy, use_fused_gdn_decode
    return _ragged_gated_delta_rule_v2_xla(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        recurrent_state=recurrent_state,
        A_log=A_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        has_initial_state=has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        apply_silu_in_gdr=apply_silu_in_gdr,
        use_recurrent_scan_prefill=use_recurrent_scan_prefill,
        mask_initial_state=mask_initial_state,
        runtime_dtype=runtime_dtype,
    )
