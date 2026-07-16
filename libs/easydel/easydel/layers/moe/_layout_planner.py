# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Build-time MoE layout planner: pure shape math, loud failures.

The fused MoE dispatch (:meth:`BaseMoeModule._sparse_moe_call`) materializes a
permuted dispatch buffer of ``local_tokens * top_k`` rows by ``hidden_size``
columns on every core. Its size is entirely determined by static shapes and
the mesh layout, so a layout that can never compile (TPU single-allocation
bound) or that will dominate HBM can be detected at trace time instead of
after a 40-minute compile. This module is pure shape math — no device work —
so every estimate is unit-testable on any host.

Calibration (v5p-2048, qwen3.6-35B-A3B, 2026-07-16, all measured):

* ``dp1/fsdp64/ep4/tp4`` with ``fsdp_is_ep_bound=True`` replicated the token
  batch per core; the ``B=512, S=8192, k=8, H=2048`` bucket produced a
  ``bf16[33554432, 2048]`` dispatch buffer (137.4 GB/core) that failed the
  TPU compiler's jellyfish bounds check (``allocation_size_words`` over the
  int32 range) after ~40 pod-minutes.
* The same mesh with ``fsdp_is_ep_bound=False`` and the 131k bucket
  (``B=256, S=131072``) produced a 17.2 GB/core buffer; with roughly three
  copies live across forward+backward it reached the HBM budgeter instead
  (``CompileTimeHbmOom``, 112.6G requested vs 95.7G available).
* ``dp4/fsdp64/ep1/tp4`` with the 131k bucket produced a 4.3 GB/core buffer
  and compiled.
* On the ``dp4/fsdp64/ep1/tp4`` mesh, expert weights sharded only over TP
  (the ``(EP, None, TP)`` at-rest layout with ep=1) left ~15 GB/model of
  expert parameters resident per device; optimizer-state init over that
  residency (f32 mu + f32 nu + bf16 z) then failed allocation
  (``RuntimeBufferAllocationFailure``). ``moe_fsdp_shard_expert_weights``
  divides the residency by the FSDP world size.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from functools import lru_cache

from eformer.loggings import get_logger

logger = get_logger(__name__)

INT32_MAX_ALLOCATION_WORDS: int = 2**31
"""TPU jellyfish bounds check: a single HBM allocation's size in words must
stay within int32 range (``allocation_size_words <= 2**31``)."""

INT32_ALLOCATION_WARN_WORD_BYTES: int = 4
"""Word granularity for the *warning* band. With 4-byte words the bound is
2**33 bytes (8.59 GB). Empirically a 17.2 GB buffer reached the HBM budgeter
rather than failing the bounds check (its compile died at ``CompileTimeHbmOom``
first), so 4-byte words cannot be confirmed as the hard granularity; buffers
over this bound get a loud warning instead of a hard error."""

INT32_ALLOCATION_HARD_WORD_BYTES: int = 32
"""Word granularity for the *hard-error* threshold (2**31 * 32 = 68.7 GB).
Chosen conservatively from the observed data points: the 137.4 GB buffer
tripped the bounds check, while the largest buffer observed to get past it
was 17.2 GB. A hard error must never reject a compilable config, so the
raise threshold sits above every observed pass and below every observed
failure."""

INT32_ALLOCATION_WARN_BOUND_BYTES: int = INT32_MAX_ALLOCATION_WORDS * INT32_ALLOCATION_WARN_WORD_BYTES
INT32_ALLOCATION_HARD_BOUND_BYTES: int = INT32_MAX_ALLOCATION_WORDS * INT32_ALLOCATION_HARD_WORD_BYTES

FWD_BWD_LIVE_BUFFER_MULTIPLIER: int = 3
"""Heuristic count of concurrently-live dispatch-buffer copies across
forward+backward: the forward sorted-activation buffer, its backward
cotangent, and one transient (sort/unsort or matmul operand). Measured on
the v5p run above: a 16 GiB buffer showed up as ~48 GB in the compile-time
HBM report."""

MOE_WEIGHT_RESIDENCY_RECOMMEND_BYTES: int = 128 * 1024 * 1024
"""Per-device expert-weight residency (bytes) above which the planner
recommends ``moe_fsdp_shard_expert_weights`` when fsdp could shard it.
Calibrated per layer: the qwen3.6-35B-A3B probe measured ~403 MB/layer per
device (268 MB gate_up + 134 MB down at tp=4, ep=1) — 15 GB/model at 40
layers, which starved optimizer-state init. 128 MiB/layer keeps small MoE
test models quiet while flagging production-scale expert stacks."""


def _fmt_bytes(num_bytes: int | float) -> str:
    """Format a byte count as a short human-readable decimal string.

    Args:
        num_bytes: Byte count.

    Returns:
        String such as ``"137.4 GB"`` or ``"268.4 MB"`` (decimal units, to
        match how TPU compile logs report buffer sizes).
    """
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1000.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} TB"


@dataclasses.dataclass(frozen=True)
class MoeLayoutEstimate:
    """Static-shape estimate of the fused-MoE memory/communication footprint.

    All byte counts are per device (per core) and per MoE layer unless noted.

    Attributes:
        global_tokens: Total tokens in the (stage-)global batch, ``B * S``.
        local_tokens: Tokens resident per core entering the fused
            ``shard_map`` (after batch sharding over ``dp`` and, when not
            ep-bound, ``fsdp``; replicated over ``ep``/``tp``/``sp``).
        dispatch_tokens: Tokens feeding the permute pipeline per core. Equals
            ``local_tokens`` for the all-to-all path and
            ``local_tokens * ep`` for ring-of-experts (chunking applies
            after the ring all-gather).
        chunk_tokens: Effective chunk size (``min(chunk_size,
            dispatch_tokens)``) when ``moe_chunk_size`` is set, else ``None``.
        dispatch_buffer_bytes: Size of one permuted dispatch buffer per core:
            ``rows * hidden_size * dtype_bytes`` with ``rows =
            (chunk_tokens or dispatch_tokens) * top_k``.
        live_multiplier: Heuristic number of concurrently-live copies of the
            dispatch buffer across forward+backward
            (:data:`FWD_BWD_LIVE_BUFFER_MULTIPLIER`).
        live_dispatch_bytes: ``dispatch_buffer_bytes * live_multiplier``.
        expert_weights_bytes_per_device: Expert parameter bytes resident per
            device at rest given the layout (sharded over ep-ways, tp, and —
            with ``moe_fsdp_shard_expert_weights`` — fsdp).
        weight_gather_bytes_per_layer: Bytes each device receives per layer
            to materialize its shard_map-local expert weights from the
            at-rest sharding (nonzero only when weights are fsdp-sharded at
            rest and gathered at use).
        all_to_all_bytes_per_layer: First-order estimate of per-device token
            communication for the expert combine (ragged all-to-all, or ring
            gather+scatter). Zero when ``ep == 1``.
        exceeds_int32_allocation_bound: ``True`` when the dispatch buffer
            exceeds the conservative hard bound
            (:data:`INT32_ALLOCATION_HARD_BOUND_BYTES`) — such a config has
            never been observed to compile on TPU.
        near_int32_allocation_bound: ``True`` when the dispatch buffer
            exceeds the 4-byte-word interpretation of the bound
            (:data:`INT32_ALLOCATION_WARN_BOUND_BYTES`); warn-worthy but not
            provably fatal.
        summary: One human-readable line describing the layout and its
            dominant costs, with a recommendation when one applies.
    """

    global_tokens: int
    local_tokens: int
    dispatch_tokens: int
    chunk_tokens: int | None
    dispatch_buffer_bytes: int
    live_multiplier: int
    live_dispatch_bytes: int
    expert_weights_bytes_per_device: int
    weight_gather_bytes_per_layer: int
    all_to_all_bytes_per_layer: int
    exceeds_int32_allocation_bound: bool
    near_int32_allocation_bound: bool
    summary: str


def estimate_moe_layout(
    mesh_axis_sizes: Mapping[str, int],
    *,
    tokens_per_device: int,
    num_experts_per_tok: int,
    hidden_size: int,
    num_experts: int,
    expert_params_bytes: int,
    dtype_bytes: int = 2,
    fsdp_is_ep_bound: bool,
    use_ring_of_experts: bool,
    chunk_size: int | None,
    fsdp_shard_expert_weights: bool = False,
) -> MoeLayoutEstimate:
    """Estimate the fused-MoE per-core memory and communication footprint.

    Pure shape math on the logical mesh layout — no device or JAX work. The
    model mirrors ``BaseMoeModule._sparse_moe_call``: the token batch is
    sharded over ``dp`` (plus ``fsdp`` when not ep-bound) and replicated over
    ``ep``/``tp``/``sp``; ring-of-experts additionally all-gathers tokens
    over ``ep`` before dispatch; ``moe_chunk_size`` caps the number of
    tokens permuted at once.

    Args:
        mesh_axis_sizes: Mapping of **logical** axis roles (``"dp"``,
            ``"fsdp"``, ``"ep"``, ``"tp"``, ``"sp"``) to their sizes. Missing
            keys default to 1. Keys are roles, not physical mesh axis names —
            callers with renamed meshes must resolve through their axis
            policy first.
        tokens_per_device: Global batch tokens (``B * S``) divided by the
            total device count of the listed axes (the ideal even split).
            The estimator multiplies the layout's replication factors back
            in.
        num_experts_per_tok: Routing top-k width.
        hidden_size: Model hidden size ``H`` (the dispatch buffer's column
            count).
        num_experts: Total routed expert count ``E``.
        expert_params_bytes: Total bytes of one layer's expert parameters
            (all experts, all projections) at storage dtype.
        dtype_bytes: Bytes per element of the dispatch buffer dtype
            (2 for bf16, 4 for f32).
        fsdp_is_ep_bound: Whether fsdp folds into expert parallelism. When
            ``True`` the token batch is NOT sharded over fsdp (replicated
            per core within each dp replica) — the inference regime.
        use_ring_of_experts: Whether the ring dispatch (ep all-gather +
            psum-scatter combine) is used instead of ragged all-to-all.
        chunk_size: ``moe_chunk_size`` (tokens per chunk) or ``None``.
        fsdp_shard_expert_weights: Whether expert weights are fsdp-sharded
            at rest and gathered at use (``moe_fsdp_shard_expert_weights``).

    Returns:
        A :class:`MoeLayoutEstimate` with per-core, per-layer byte counts
        and the int32 single-allocation verdicts.
    """
    dp = max(1, int(mesh_axis_sizes.get("dp", 1) or 1))
    fsdp = max(1, int(mesh_axis_sizes.get("fsdp", 1) or 1))
    ep = max(1, int(mesh_axis_sizes.get("ep", 1) or 1))
    tp = max(1, int(mesh_axis_sizes.get("tp", 1) or 1))
    sp = max(1, int(mesh_axis_sizes.get("sp", 1) or 1))

    total_devices = dp * fsdp * ep * tp * sp
    global_tokens = int(tokens_per_device) * total_devices

    batch_ways = dp * (1 if fsdp_is_ep_bound else fsdp)
    local_tokens = math.ceil(global_tokens / batch_ways)
    dispatch_tokens = local_tokens * (ep if use_ring_of_experts else 1)

    chunk_tokens: int | None = None
    if chunk_size is not None and int(chunk_size) > 0:
        chunk_tokens = min(int(chunk_size), dispatch_tokens)

    buffer_rows = (chunk_tokens if chunk_tokens is not None else dispatch_tokens) * int(num_experts_per_tok)
    dispatch_buffer_bytes = buffer_rows * int(hidden_size) * int(dtype_bytes)
    live_dispatch_bytes = dispatch_buffer_bytes * FWD_BWD_LIVE_BUFFER_MULTIPLIER

    # Expert-weight residency at rest. The expert (leading) dim is sharded
    # over ep — plus fsdp when either ep-bound (folded) or explicitly
    # fsdp-sharded — and the in/out dims over tp.
    fsdp_shards_weights = fsdp_is_ep_bound or fsdp_shard_expert_weights
    weight_shards = ep * tp * (fsdp if fsdp_shards_weights else 1)
    expert_weights_bytes_per_device = math.ceil(int(expert_params_bytes) / weight_shards)

    # Gather-at-use cost: only the explicit ZeRO-3-style path re-gathers over
    # fsdp per layer; ep-bound weights stay fully sharded inside the expert
    # shard_map and need no gather.
    if fsdp_shard_expert_weights and not fsdp_is_ep_bound and fsdp > 1:
        gathered_bytes = math.ceil(int(expert_params_bytes) / (ep * tp))
        weight_gather_bytes_per_layer = gathered_bytes * (fsdp - 1) // fsdp
    else:
        weight_gather_bytes_per_layer = 0

    # Token communication for the expert combine (first-order, per device,
    # forward only).
    if ep <= 1:
        all_to_all_bytes_per_layer = 0
    elif use_ring_of_experts:
        # all_gather of the local token block over ep (in) + psum_scatter of
        # the combined output over ep (out).
        all_to_all_bytes_per_layer = 2 * local_tokens * (ep - 1) * int(hidden_size) * int(dtype_bytes)
    else:
        # ragged_all_to_all replicating each shard's computed rows to the
        # other ep shards (H is tp-scattered by then).
        row_bytes = math.ceil(int(hidden_size) / tp) * int(dtype_bytes)
        all_to_all_bytes_per_layer = dispatch_tokens * int(num_experts_per_tok) * row_bytes * (ep - 1) // ep

    exceeds_hard = dispatch_buffer_bytes > INT32_ALLOCATION_HARD_BOUND_BYTES
    exceeds_warn = dispatch_buffer_bytes > INT32_ALLOCATION_WARN_BOUND_BYTES

    recommend_fsdp_weights = (
        not fsdp_shards_weights and fsdp > 1 and expert_weights_bytes_per_device > MOE_WEIGHT_RESIDENCY_RECOMMEND_BYTES
    )

    parts = [
        (
            f"MoE layout dp={dp} fsdp={fsdp} ep={ep} tp={tp} sp={sp} "
            f"(fsdp_is_ep_bound={fsdp_is_ep_bound}, ring={use_ring_of_experts}, "
            f"chunk={chunk_tokens if chunk_tokens is not None else 'off'}, "
            f"fsdp_shard_expert_weights={fsdp_shard_expert_weights})"
        ),
        f"{local_tokens:,} tokens/core (x{ep} ring-gathered)" if use_ring_of_experts else f"{local_tokens:,} tokens/core",
        f"dispatch buffer {_fmt_bytes(dispatch_buffer_bytes)}/core "
        f"(~x{FWD_BWD_LIVE_BUFFER_MULTIPLIER} live fwd+bwd = {_fmt_bytes(live_dispatch_bytes)})",
        f"expert weights resident {_fmt_bytes(expert_weights_bytes_per_device)}/device",
        f"weight-gather {_fmt_bytes(weight_gather_bytes_per_layer)}/layer",
        f"token-comm {_fmt_bytes(all_to_all_bytes_per_layer)}/layer",
    ]
    if exceeds_hard:
        parts.append("EXCEEDS TPU int32 single-allocation bound (cannot compile)")
    elif exceeds_warn:
        parts.append("near/over the 4-byte-word int32 allocation bound (may not compile)")
    if recommend_fsdp_weights:
        parts.append(
            "recommendation: set moe_fsdp_shard_expert_weights=True to shard expert weights "
            f"(and their optimizer state) over fsdp={fsdp}"
        )
    summary = "; ".join(parts)

    return MoeLayoutEstimate(
        global_tokens=global_tokens,
        local_tokens=local_tokens,
        dispatch_tokens=dispatch_tokens,
        chunk_tokens=chunk_tokens,
        dispatch_buffer_bytes=dispatch_buffer_bytes,
        live_multiplier=FWD_BWD_LIVE_BUFFER_MULTIPLIER,
        live_dispatch_bytes=live_dispatch_bytes,
        expert_weights_bytes_per_device=expert_weights_bytes_per_device,
        weight_gather_bytes_per_layer=weight_gather_bytes_per_layer,
        all_to_all_bytes_per_layer=all_to_all_bytes_per_layer,
        exceeds_int32_allocation_bound=exceeds_hard,
        near_int32_allocation_bound=exceeds_warn,
        summary=summary,
    )


@lru_cache(maxsize=256)
def _validate_moe_layout_cached(
    mesh_axis_sizes_items: tuple[tuple[str, int], ...],
    tokens_per_device: int,
    num_experts_per_tok: int,
    hidden_size: int,
    num_experts: int,
    expert_params_bytes: int,
    dtype_bytes: int,
    fsdp_is_ep_bound: bool,
    use_ring_of_experts: bool,
    chunk_size: int | None,
    fsdp_shard_expert_weights: bool,
) -> MoeLayoutEstimate:
    """Cached estimate + logging + hard failure for one hashable layout.

    The cache is what makes the module-side hook "once per configuration"
    instead of once per traced layer or per step: 61 identical MoE layers
    hit the cache 60 times and log once. Exceptions are not cached by
    ``lru_cache``, so a fatal layout raises on every call — which is fine,
    the first raise aborts the trace.

    Args:
        mesh_axis_sizes_items: Sorted ``(role, size)`` pairs of the logical
            mesh layout.
        tokens_per_device: See :func:`estimate_moe_layout`.
        num_experts_per_tok: See :func:`estimate_moe_layout`.
        hidden_size: See :func:`estimate_moe_layout`.
        num_experts: See :func:`estimate_moe_layout`.
        expert_params_bytes: See :func:`estimate_moe_layout`.
        dtype_bytes: See :func:`estimate_moe_layout`.
        fsdp_is_ep_bound: See :func:`estimate_moe_layout`.
        use_ring_of_experts: See :func:`estimate_moe_layout`.
        chunk_size: See :func:`estimate_moe_layout`.
        fsdp_shard_expert_weights: See :func:`estimate_moe_layout`.

    Returns:
        The computed :class:`MoeLayoutEstimate`.

    Raises:
        ValueError: When the dispatch buffer exceeds the conservative TPU
            int32 single-allocation bound — that layout has never been
            observed to compile, so failing at trace time saves the compile.
    """
    estimate = estimate_moe_layout(
        dict(mesh_axis_sizes_items),
        tokens_per_device=tokens_per_device,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        num_experts=num_experts,
        expert_params_bytes=expert_params_bytes,
        dtype_bytes=dtype_bytes,
        fsdp_is_ep_bound=fsdp_is_ep_bound,
        use_ring_of_experts=use_ring_of_experts,
        chunk_size=chunk_size,
        fsdp_shard_expert_weights=fsdp_shard_expert_weights,
    )
    logger.info(estimate.summary)
    if estimate.exceeds_int32_allocation_bound:
        raise ValueError(
            f"Fused-MoE dispatch buffer of {estimate.dispatch_buffer_bytes:,} bytes per core "
            f"({(estimate.chunk_tokens or estimate.dispatch_tokens) * num_experts_per_tok:,} rows x "
            f"{hidden_size} cols x {dtype_bytes} B) exceeds the TPU int32 single-allocation bound "
            f"(~{INT32_ALLOCATION_HARD_BOUND_BYTES:,} bytes; jellyfish bounds check "
            f"'allocation_size_words <= int32_max'). This layout cannot compile on TPU. Shrink the "
            f"per-core dispatch buffer by one or more of: (1) set config.moe_chunk_size to cap the "
            f"permuted rows at chunk_size * top_k (e.g. moe_chunk_size=65536); (2) set "
            f"fsdp_is_ep_bound=False so the token batch shards over (dp, fsdp) instead of being "
            f"replicated per core; (3) increase dp/fsdp batch ways in sharding_axis_dims. "
            f"Layout: {estimate.summary}"
        )
    if estimate.near_int32_allocation_bound:
        logger.warning(
            f"Fused-MoE dispatch buffer {_fmt_bytes(estimate.dispatch_buffer_bytes)}/core is over the "
            f"4-byte-word int32 allocation bound ({_fmt_bytes(INT32_ALLOCATION_WARN_BOUND_BYTES)}) and "
            f"~x{estimate.live_multiplier} of it is live across fwd+bwd "
            f"({_fmt_bytes(estimate.live_dispatch_bytes)}); expect compile-time HBM pressure. "
            f"Consider moe_chunk_size or more dp/fsdp batch ways."
        )
    if "recommendation:" in estimate.summary:
        logger.warning(estimate.summary)
    return estimate


def validate_moe_layout(
    mesh_axis_sizes: Mapping[str, int],
    *,
    tokens_per_device: int,
    num_experts_per_tok: int,
    hidden_size: int,
    num_experts: int,
    expert_params_bytes: int,
    dtype_bytes: int = 2,
    fsdp_is_ep_bound: bool,
    use_ring_of_experts: bool,
    chunk_size: int | None,
    fsdp_shard_expert_weights: bool = False,
) -> MoeLayoutEstimate:
    """Estimate an MoE layout, log it at INFO, and hard-fail impossible ones.

    Same arguments as :func:`estimate_moe_layout`. Results are memoized on
    the full argument tuple so repeated calls (one per MoE layer per trace)
    do the math and logging once.

    Returns:
        The computed :class:`MoeLayoutEstimate`.

    Raises:
        ValueError: When the dispatch buffer exceeds the conservative TPU
            int32 single-allocation bound (see
            :data:`INT32_ALLOCATION_HARD_BOUND_BYTES`).
    """
    return _validate_moe_layout_cached(
        tuple(sorted((str(k), int(v)) for k, v in mesh_axis_sizes.items())),
        int(tokens_per_device),
        int(num_experts_per_tok),
        int(hidden_size),
        int(num_experts),
        int(expert_params_bytes),
        int(dtype_bytes),
        bool(fsdp_is_ep_bound),
        bool(use_ring_of_experts),
        None if chunk_size is None else int(chunk_size),
        bool(fsdp_shard_expert_weights),
    )
