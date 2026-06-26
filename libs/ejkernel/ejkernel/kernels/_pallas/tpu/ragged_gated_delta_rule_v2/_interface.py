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

"""Ragged GDN v2 TPU Pallas kernel interface and registry entry."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import ragged_gated_delta_rule_v2 as _ragged_gated_delta_rule_v2_pallas


@kernel_registry.register("ragged_gated_delta_rule_v2", Platform.PALLAS, Backend.TPU)
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
    """Run the packed-inference ragged GDN v2 TPU Pallas kernel.

    Registry entry point for the v2 ragged Gated Delta Rule (GDN) forward pass on TPU. It casts the
    floating-point inputs to ``runtime_dtype`` (inside the underlying implementation), supplies a default
    ``has_initial_state`` mask when one is not given, and delegates to the packed-inference forward
    implementation. The call handles a ragged batch of variable-length requests packed contiguously into
    ``num_tokens`` rows, where the prefill and decode segments are distinguished by ``distribution``.

    Args:
        mixed_qkv: Packed query/key/value projections for every token, shape ``(num_tokens, mixed_dim)``
            where ``mixed_dim`` interleaves the ``n_kq`` query/key heads of dim ``d_k`` and ``n_v`` value
            heads of dim ``d_v``.
        b: Per-token, per-value-head beta (write strength) logits, shape ``(num_tokens, num_value_heads)``.
        a: Per-token, per-value-head gating/decay logits, shape ``(num_tokens, num_value_heads)``.
        recurrent_state: Recurrent state pool, shape
            ``(num_slots, num_value_heads, qk_head_dim, v_head_dim)``; read for each request's initial
            state and updated with the post-sequence state.
        A_log: Per-value-head log-decay parameter, shape ``(num_value_heads,)``.
        dt_bias: Per-value-head time-step bias added to the gating term, shape ``(num_value_heads,)``.
        query_start_loc: Cumulative token offsets delimiting each request, shape
            ``(num_requests + 1,)`` in ``int32``; request ``i`` spans
            ``[query_start_loc[i], query_start_loc[i + 1])``.
        state_indices: Mapping from request index to its slot in ``recurrent_state``, shape
            ``(num_requests,)`` in ``int32``.
        distribution: Three-element ``int32`` vector describing the prefill/decode split of the packed
            batch used to schedule the ragged kernel.
        has_initial_state: Optional boolean mask, shape ``(num_requests,)``, marking requests whose
            ``recurrent_state`` slot already holds a valid carry-in state. When ``None``, every request is
            assumed to have a valid initial state.
        n_kq: Number of query/key heads packed into ``mixed_qkv``.
        n_v: Number of value heads packed into ``mixed_qkv``.
        d_k: Per-head query/key dimension.
        d_v: Per-head value dimension.
        chunk_size: Token block size used by the chunked prefill path. Defaults to ``64``.
        use_qk_norm_in_gdn: If ``True``, apply per-head normalization to queries and keys inside the GDN
            update. Defaults to ``True``.
        apply_silu_in_gdr: If ``True``, apply a SiLU activation within the gated delta-rule update.
            Defaults to ``False``.
        use_recurrent_scan_prefill: If ``True``, run prefill through the recurrent scan path instead of the
            chunked path. Defaults to ``False``.
        mask_initial_state: If ``True``, use ``has_initial_state`` to zero stale
            recurrent slots for fresh prefill requests. Defaults to ``False``.
        kernel_tile_policy: TPU Pallas token tile policy for decode.
        use_fused_gdn_decode: Whether to use the fused TPU decode kernel when
            the shape is supported.
        runtime_dtype: Optional dtype to which the floating-point inputs are cast before computation. When
            ``None``, the dtype of ``mixed_qkv`` is used. Defaults to ``None``.

    Returns:
        tuple: ``(updated_state, output)`` where ``updated_state`` has shape
        ``(num_slots, num_value_heads, qk_head_dim, v_head_dim)`` and carries the post-sequence recurrent
        state for each touched slot, and ``output`` has shape ``(num_tokens, out_dim)`` holding the
        per-token GDN attention result.
    """
    return _ragged_gated_delta_rule_v2_pallas(
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
        kernel_tile_policy=kernel_tile_policy,
        use_fused_gdn_decode=use_fused_gdn_decode,
        runtime_dtype=runtime_dtype,
    )
