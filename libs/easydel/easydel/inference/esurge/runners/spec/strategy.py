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

"""Runner-native speculative decoding as a pluggable strategy.

:class:`DrafterSpeculation` owns every piece of the eSurge runner's
draft/verify/commit machinery that used to live inline on
:class:`~easydel.inference.esurge.runners.model_runner.eSurgeRunner` as
``_spec_*`` methods: drafter invocation, verify/resample JIT caches,
rejection-backoff state, recurrent-candidate commits, KV replay, and the
acceptance counters. The code was moved verbatim; only attribute paths
changed (runner-owned state is reached through the documented ``_runner``
back-reference).
"""

from __future__ import annotations

import math
import time
import typing
from bisect import bisect_left
from dataclasses import dataclass

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

from easydel.utils import flags

from ...config import normalize_draft_schedule, resolve_num_draft_tokens
from ...core.binary_search import apply_topk_mask, apply_topp_mask
from ..executors.model_executor import resolve_lm_logit_postprocess
from .support import build_target_kv_pairs, gather_paged_kv

if typing.TYPE_CHECKING:
    from easydel.inference.speculative import DrafterProtocol

    from ...scheduler import SchedulerOutput
    from ..model_runner import eSurgeRunner
    from ..states import CachedRequestState

logger = get_logger("eSurge")


@dataclass(frozen=True)
class _SpecVerifyMetadata:
    """Host-side token/index contract for one speculative verification window."""

    request_id: str
    row_pos: int
    req_idx: int
    start_pos: int
    real_count: int
    target_local_indices: list[int]
    bonus_local_index: int
    draft_token_positions: list[int]
    scheduled_draft_tokens: list[int]
    buffer_draft_tokens: list[int]

    @property
    def num_drafts(self) -> int:
        """Number of draft tokens described by this verification metadata."""
        return len(self.buffer_draft_tokens)


def window_hidden_row(
    hidden_states: jax.Array,
    *,
    row_pos: int,
    token_offset: int,
    token_index: int,
    total_window_tokens: int,
    padded_reqs: int,
    output_layout: typing.Literal["tokens", "requests"] | None = None,
) -> jax.Array:
    """Select a token or request row using an explicit layout when available.

    ``None`` falls back to a shape heuristic. Runner
    executions supply their bucket's layout, since active token counts alone
    cannot distinguish full outputs from padded gathered request rows.
    """
    hidden_len = int(hidden_states.shape[0])
    full_index = int(token_offset) + int(token_index)
    if output_layout is not None:
        if output_layout not in ("tokens", "requests"):
            raise ValueError(f"Unknown hidden output layout: {output_layout!r}")
        index = full_index if output_layout == "tokens" else int(row_pos)
        if not 0 <= index < hidden_len:
            raise IndexError(f"Hidden row {index} is outside the {output_layout} output of length {hidden_len}.")
        return hidden_states[index]
    if hidden_len >= int(total_window_tokens) and 0 <= full_index < hidden_len:
        return hidden_states[full_index]
    if hidden_len >= int(padded_reqs) and 0 <= int(row_pos) < hidden_len:
        return hidden_states[int(row_pos)]
    return hidden_states[min(max(full_index, 0), hidden_len - 1)]


# Device-array fields carried by a recurrent cache view. These are the only
# leaves whose state advances irreversibly inside a fused speculative forward
# and therefore the only leaves a speculative rollback must restore.
_REC_ARRAY_FIELDS = ("conv_state", "recurrent_state", "positions", "seqlen_offset")
# ``KDACacheView`` (Kimi-Linear / GLM-5-Next KDA layers) extends ``BaseCacheView`` directly and keeps separate Q/K/V
# short-conv windows; none of its per-slot leaves share a name with the generic recurrent view.
_KDA_ARRAY_FIELDS = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state", "positions")


@jax.jit
def _jit_copy_arrays(arrays: list[jax.Array]) -> list[jax.Array]:
    """Copy a flat list of device arrays in a SINGLE fused dispatch.

    Eager ``jnp.array(x, copy=True)`` per leaf costs one host dispatch each; a
    hybrid model has ~4 recurrent leaves per linear layer (~200 for a 64-layer
    27B), so per-leaf copying spent ~30 ms/spec-step purely on dispatch. Copying
    the whole flat list under one ``jax.jit`` collapses that to one dispatch.
    ``jax.jit`` outputs are freshly allocated buffers (no input-output aliasing
    without ``donate_argnums``), so the copies survive the KV-buffer donation the
    fused forward performs on the originals.
    """
    return [jnp.array(a, copy=True) for a in arrays]


def _recurrent_view_of(view: typing.Any, recurrent_cls: type, parallel_cls: type) -> typing.Any:
    """Return the ``RecurrentCacheView`` carried by ``view`` (or ``None``)."""
    if isinstance(view, recurrent_cls):
        return view
    if isinstance(view, parallel_cls) and view.recurrent is not None:
        return view.recurrent
    return None


class _CompressedWindowSnapshot(typing.NamedTuple):
    """Device copy of a compressed-window cache's per-stream state, taken before a speculative verify window.

    ``live_indices`` are the compressed-entry leaves left out of the copy: the rollback keeps the live arrays.
    """

    treedef: typing.Any
    leaves: list[typing.Any]
    array_indices: list[int]
    copied: list[jax.Array]
    live_indices: tuple[int, ...] = ()


# Compressed-entry / indexer-key buffers of the compressed-window caches (``CompressedWindowCacheView`` and
# ``SharedCompressedWindowCacheView``). They are the bulk of the cache bytes and need no rollback: entry ``w`` is
# written by the token that closes its window (``w * rate + rate - 1``) before any read in the same step, and every
# reader masks entries by reach ``(position + 1) // rate`` (decode visibility, block extend bias, and the indexer's
# ``select_candidates`` sets unreachable scores to ``-inf`` before its top-k). An entry left behind by a rejected
# draft therefore sits beyond the reach of every query until the real token at that position overwrites it.
# The replay-free rollback traces need no rollback either: a commit reads only trace slots the verify window itself
# wrote, and a replay rewrites every slot it reads.
_COMPRESSED_WINDOW_LIVE_FIELDS = frozenset(
    {
        "compressor_entries",
        "indexer_entries",
        "compressed_kv",
        "index_k",
        "compressor_trace_kv",
        "compressor_trace_score",
        "compressor_trace_gate",
        "compressor_trace_overlap_kv",
        "compressor_trace_overlap_gate",
        "indexer_trace_kv",
        "indexer_trace_gate",
        "indexer_trace_overlap_kv",
        "indexer_trace_overlap_gate",
    }
)


def _path_field_name(path: tuple[typing.Any, ...]) -> str | None:
    """Name of the dataclass field (or dict key) that owns a pytree leaf, from its key path."""
    if not path:
        return None
    last = path[-1]
    for attr in ("name", "key"):
        value = getattr(last, attr, None)
        if isinstance(value, str):
            return value
    return None


def _is_compressed_window_cache(cache: typing.Any) -> bool:
    """Whether ``cache`` is a per-slot compressed-window cache (DeepSeek-V4 / V4.1)."""
    from easydel.caching import CompressedWindowCache
    from easydel.caching.compressed_window import SharedCompressedWindowCache

    return isinstance(cache, CompressedWindowCache | SharedCompressedWindowCache)


def _snapshot_compressed_window(cache: typing.Any) -> _CompressedWindowSnapshot:
    """Copy the per-stream array leaves of a compressed-window cache in one fused dispatch.

    Ring / open-group compressor / overlap state and ``cache_position`` advance irreversibly on every consumed token,
    so rejected draft tokens are rolled back by restoring a pre-verify copy and replaying the accepted prefix. The
    compressed-entry buffers (:data:`_COMPRESSED_WINDOW_LIVE_FIELDS`) are not copied: the restore keeps them live.
    """
    with_paths, treedef = jax.tree_util.tree_flatten_with_path(cache)
    leaves = [leaf for _, leaf in with_paths]
    live = tuple(
        i
        for i, (path, leaf) in enumerate(with_paths)
        if isinstance(leaf, jax.Array) and _path_field_name(path) in _COMPRESSED_WINDOW_LIVE_FIELDS
    )
    indices = [i for i, leaf in enumerate(leaves) if isinstance(leaf, jax.Array) and i not in live]
    copied = _jit_copy_arrays([leaves[i] for i in indices])
    kept = [None if isinstance(leaf, jax.Array) else leaf for leaf in leaves]
    return _CompressedWindowSnapshot(treedef, kept, indices, copied, live)


def _restore_compressed_window(
    live_cache: typing.Any, snapshot: _CompressedWindowSnapshot, *, copy: bool = True
) -> typing.Any:
    """Rebuild the cache: the snapshot's per-stream leaves, live compressed-entry buffers.

    With ``copy`` (the default) the snapshot is re-copied so it survives the donation of each replay forward. The
    replay-free commit reads the pre-window cache through a non-donating jit, so it passes ``copy=False`` and saves
    one full copy of every per-stream leaf per rolled-back step.
    """
    fresh = _jit_copy_arrays(snapshot.copied) if copy else snapshot.copied
    leaves = list(snapshot.leaves)
    for index, array in zip(snapshot.array_indices, fresh, strict=True):
        leaves[index] = array
    if snapshot.live_indices:
        live_leaves, live_treedef = jax.tree_util.tree_flatten(live_cache)
        if live_treedef != snapshot.treedef:
            raise ValueError("compressed-window cache structure changed between snapshot and restore")
        for index in snapshot.live_indices:
            leaves[index] = live_leaves[index]
    return jax.tree_util.tree_unflatten(snapshot.treedef, leaves)


def _window_exceeds_limit_impl(
    live_positions: list[jax.Array], pre_positions: list[jax.Array], committed: jax.Array, limit: int
) -> jax.Array:
    """Whether any rolled-back row (``committed < live - pre``) of any view has a window longer than ``limit``."""
    exceeds = jnp.zeros((), dtype=bool)
    for live_pos, pre_pos in zip(live_positions, pre_positions, strict=True):
        count = live_pos.astype(jnp.int32) - pre_pos.astype(jnp.int32)
        exceeds = exceeds | jnp.any((committed < count) & (count > limit))
    return exceeds


_window_exceeds_limit = jax.jit(_window_exceeds_limit_impl, static_argnums=(3,))


def snapshot_recurrent_state(kv_pages: typing.Any) -> typing.Any:
    """Snapshot only the recurrent leaves of a hybrid KV cache for spec rollback.

    Speculative rollback for hybrid models only needs the recurrent leaves
    (:data:`_REC_ARRAY_FIELDS`) preserved. The paged full-attention KV is pure
    position-indexed storage (``RaggedPagesCacheView`` has no device-side length
    pointer): a replay forward re-derives the accepted positions deterministically
    from token id + position, and rejected draft positions are never read before a
    later step overwrites them. So the paged tree is rolled back by truncation
    (overwrite), never copied.

    All recurrent leaves across every layer are gathered into one flat list and
    copied in a single fused dispatch (:func:`_jit_copy_arrays`), which is what
    makes the per-spec-step snapshot cheap.

    Args:
        kv_pages: The live cache (expected to be a ``HybridCache`` when a
            snapshot is requested).

    Returns:
        ``(layout, copied_arrays)`` where ``layout`` records, per recurrent view,
        its cache-view index, whether it is a parallel-hybrid view, and the
        per-field index into ``copied_arrays`` (or ``None`` for a non-array
        field); or ``None`` when the cache has no recurrent leaves.
    """
    from easydel.caching import KDACacheView, ParallelHybridCacheView, RecurrentCacheView

    if _is_compressed_window_cache(kv_pages):
        return _snapshot_compressed_window(kv_pages)
    views = getattr(kv_pages, "views", None)
    if views is None:
        return None
    layout: list[tuple[int, bool, dict[str, int | None]]] = []
    flat: list[jax.Array] = []
    for idx, view in enumerate(views):
        rec = _recurrent_view_of(view, RecurrentCacheView, ParallelHybridCacheView)
        fields = _REC_ARRAY_FIELDS
        if rec is None and isinstance(view, KDACacheView):
            rec, fields = view, _KDA_ARRAY_FIELDS
        if rec is None:
            continue
        field_map: dict[str, int | None] = {}
        for field_name in fields:
            val = getattr(rec, field_name, None)
            if isinstance(val, jax.Array):
                field_map[field_name] = len(flat)
                flat.append(val)
            else:
                field_map[field_name] = None
        layout.append((idx, isinstance(view, ParallelHybridCacheView), field_map))
    if not flat:
        return None
    return (layout, _jit_copy_arrays(flat))


def _restore_recurrent_state(live_kv_pages: typing.Any, snapshot: typing.Any) -> typing.Any:
    """Rebuild a hybrid cache from live attention pages + snapshot recurrent state.

    Keeps every full-attention/paged view of ``live_kv_pages`` (the post-forward
    cache) as-is and splices back fresh copies of the recurrent leaves captured by
    :func:`snapshot_recurrent_state`. The snapshot leaves are re-copied (one
    fused dispatch) on every restore so a single snapshot survives the KV-buffer
    donation performed by each replay forward (one snapshot may feed several
    replay calls within a verify window).

    Args:
        live_kv_pages: The current (post-forward) cache whose attention pages are
            reused verbatim.
        snapshot: ``(layout, copied_arrays)`` from
            :func:`snapshot_recurrent_state`; ``None`` returns ``live_kv_pages``
            unchanged.

    Returns:
        A ``HybridCache`` with live attention views and pre-step recurrent state.
    """
    from easydel.caching import HybridCache

    if isinstance(snapshot, _CompressedWindowSnapshot):
        return _restore_compressed_window(live_kv_pages, snapshot)
    if snapshot is None:
        return live_kv_pages
    layout, copied = snapshot
    fresh = _jit_copy_arrays(copied)
    views = list(live_kv_pages.views)
    for view_idx, is_parallel, field_map in layout:
        updates = {name: (fresh[i] if i is not None else None) for name, i in field_map.items()}
        live_view = views[view_idx]
        if is_parallel:
            views[view_idx] = live_view.replace(recurrent=live_view.recurrent.replace(**updates), **updates)
        else:
            views[view_idx] = live_view.replace(**updates)
    return HybridCache(views=views)


class DrafterSpeculation:
    """Runner-native speculative decoding driven by an attached drafter.

    Owns the drafter object, the verify/resample JIT caches, the
    rejection-backoff map, debug traces, and the acceptance counters.

    The strategy collaborates with the runner through a documented
    back-reference (``self._runner``). The moved methods touch well over six
    live runner handles (sequence buffer, request map, executor manager,
    cache metadata, KV-cache groups, token buckets, host scratch buffers,
    device input buffers, pending recurrent-commit queue), several of which
    are re-created by ``eSurgeRunner._setup_variables`` on weight updates —
    so holding the runner handle, rather than snapshotting individual
    collaborators, keeps this code byte-identical in behavior to its
    pre-extraction form.
    """

    def __init__(
        self,
        *,
        runner: eSurgeRunner,
        drafter: DrafterProtocol,
        num_draft_tokens: int,
        draft_schedule: list[tuple[int, int, int]] | None = None,
        debug_max_traces: int = 0,
        adaptive_conf_threshold: float = 0.0,
    ) -> None:
        """Initialize the drafter-backed strategy.

        Args:
            runner: Owning :class:`eSurgeRunner`; documented back-reference
                used to reach live runner state.
            drafter: Speculative drafter implementing
                :class:`~easydel.inference.speculative.DrafterProtocol`.
            num_draft_tokens: Static draft tokens proposed per verify window;
                the compiled maximum, and the fallback for concurrencies the
                schedule does not cover.
            draft_schedule: Optional dynamic draft-token schedule
                (``num_draft_tokens_per_batch_size``): inclusive
                ``(start, end, k)`` live-concurrency ranges resolved once per
                scheduler step via :meth:`resolve_step_draft_budget`. ``None``
                keeps the static count at every concurrency.
            debug_max_traces: Maximum verify-window debug traces to retain;
                zero disables trace capture.
            adaptive_conf_threshold: Top-1 confidence threshold for early-exit
                drafting, in [0, 1]. Zero keeps fixed-length drafting.
        """
        if int(debug_max_traces) < 0:
            raise ValueError("debug_max_traces must be nonnegative")
        if not math.isfinite(float(adaptive_conf_threshold)) or not 0.0 <= float(adaptive_conf_threshold) <= 1.0:
            raise ValueError("adaptive_conf_threshold must be finite and in [0, 1]")
        self._runner = runner
        self.drafter = drafter
        self.num_draft_tokens = int(num_draft_tokens)
        self.draft_schedule = (
            normalize_draft_schedule(draft_schedule, max_num_draft_tokens=self.num_draft_tokens)
            if draft_schedule is not None
            else None
        )
        # Live per-step draft budget; re-resolved every scheduler step from
        # the schedule (static without one). Draft paths read THIS, never the
        # static maximum, so a K transition takes effect on the next window.
        self.live_num_draft_tokens = int(num_draft_tokens)
        self.num_drafts_generated = 0
        self.num_drafts_accepted = 0
        self.num_verify_steps = 0
        # Windows rolled back by :meth:`_commit_compressed_window_without_replay` (no replay forward).
        self.num_replay_free_commits = 0
        self.debug_traces: list[dict[str, typing.Any]] = []
        self.debug_max_traces = int(debug_max_traces)
        self.adaptive_conf_threshold = float(adaptive_conf_threshold)
        self.draft_full_log_probs_by_req: dict[str, jax.Array] = {}
        self._filter_sample_fns: dict[tuple, typing.Callable] = {}
        self._verify_sample_fns: dict[tuple, typing.Callable] = {}
        self._recurrent_commit_fn: typing.Callable | None = None
        self._recurrent_commit_candidate_count: int = 0
        self.reject_backoff_steps = 0
        self._backoff_by_req: dict[str, int] = {}
        # Persist-with-rollback of the inline-MTP KV cache (EAGLE-style), now
        # PER REQUEST for batched serving. The MTP cache is sized to the request
        # pool (``max_num_reqs`` rows) and each concurrent request keeps its own
        # row across verify windows, keyed on its stable KV-pool slot (the
        # runner's ``req_idx``). `_mtp_persist_base_pos_by_row` maps a row to that
        # request's first-window seed position (the RoPE origin of compacted cache
        # slot 0 for that row); `_mtp_persist_req_by_row` maps a row to the request
        # id currently occupying it, so a slot reused by a different request is
        # detected and re-based (its stale K/V dropped) instead of leaking.
        self._mtp_persist_base_pos_by_row: dict[int, int] = {}
        self._mtp_persist_req_by_row: dict[int, str] = {}
        # Per-row record of the last window: its seed position and how many MTP
        # cache slots it wrote from the seed slot onward. A fully accepted window
        # commits one more position than it wrote slots for, so the next window
        # re-bases ``_mtp_persist_base_pos_by_row`` by that gap (see
        # ``_mtp_persist_committed_len``) instead of letting ``slot = pos - base``
        # drift and keep rejected-draft K/V.
        self._mtp_persist_last_seed_by_row: dict[int, int] = {}
        self._mtp_persist_written_by_row: dict[int, int] = {}
        self._warned_missing_target_kv = False
        # STABLE per-request drafter rows. The runner's ``req_idx`` is NOT
        # stable: ``condense``/``reorder_decode_first`` relocate surviving
        # requests to different sequence-buffer rows, and while the paged KV
        # (page-table rows) and recurrent state (``permute_recurrent_slots``)
        # move in lockstep, the drafter's pooled MTP cache does not. Keying the
        # MTP rows on ``req_idx`` therefore cold-started (or, on an A->B->A
        # move, corrupted) a relocated request's MTP context — and relocation
        # frequency grows with concurrency, degrading acceptance as n grows.
        # Instead the strategy allocates each request its own drafter row for
        # its whole lifetime, released when the scheduler reports it finished.
        self._mtp_row_by_req: dict[str, int] = {}
        self._mtp_row_free: list[int] = []
        self._mtp_row_capacity: int = 0
        # Whole-block drafters (DSpark): the hidden rows of the step being verified and, per request, its
        # ``(row offset, first position, token count)`` there; refreshed every step by ``observe_window_hidden``.
        self._whole_block_hidden: jax.Array | None = None
        self._whole_block_window: dict[str, tuple[int, int, int]] = {}

    def _stable_drafter_row(self, req_id: str, fallback_row: int) -> int:
        """Return this request's lifetime-stable row in the pooled MTP cache.

        Allocates a free row on first use and keeps it until the request
        finishes (released in :meth:`on_states_applied`). Falls back to
        reclaiming rows of no-longer-live requests when the free list is
        empty (missed release), and only as a last resort to the caller's
        (unstable) ``req_idx``.

        Args:
            req_id: Request identifier.
            fallback_row: Row to use if allocation is impossible.

        Returns:
            The stable row index in ``[0, max_num_reqs)``.
        """
        rid = str(req_id)
        capacity = max(1, int(getattr(self._runner, "max_num_reqs", 1)))
        if self._mtp_row_capacity != capacity:
            live = set(self._mtp_row_by_req.values())
            self._mtp_row_capacity = capacity
            self._mtp_row_free = [r for r in range(capacity - 1, -1, -1) if r not in live]
        row = self._mtp_row_by_req.get(rid)
        if row is not None:
            return row
        if not self._mtp_row_free:
            # Self-heal: reclaim rows whose owners are no longer live.
            live_reqs = getattr(self._runner, "requests", {}) or {}
            for other_rid in [r for r in self._mtp_row_by_req if r not in live_reqs]:
                self._mtp_row_free.append(self._mtp_row_by_req.pop(other_rid))
        row = self._mtp_row_free.pop() if self._mtp_row_free else int(fallback_row)
        self._mtp_row_by_req[rid] = row
        return row

    def _release_finished_drafter_rows(self, scheduler_output: SchedulerOutput) -> None:
        """Free the stable drafter rows of requests the scheduler finished."""
        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        for rid in finished:
            row = self._mtp_row_by_req.pop(str(rid), None)
            if row is None:
                continue
            self._mtp_row_free.append(row)
            if self._mtp_persist_req_by_row.get(row) == str(rid):
                self._mtp_persist_req_by_row.pop(row, None)
                self._mtp_persist_base_pos_by_row.pop(row, None)
                self._mtp_persist_last_seed_by_row.pop(row, None)
                self._mtp_persist_written_by_row.pop(row, None)

    def _mtp_persist_committed_len(self, row: int, seed_position: int) -> int:
        """Return ``row``'s committed MTP slot count for a window seeded at ``seed_position``.

        Persisted MTP slots map to sequence positions as ``slot = position - base``.
        A window seeded at ``s`` writes one slot per draft step for positions
        ``s, s + 1, ...``. When the verify accepts every draft the sequence
        advances ``k + 1`` positions while only ``k`` slots were written, so the
        position after the last written slot never enters the cache. Without
        re-basing, every later window would compute a committed length one slot
        too large per such gap — ``rollback_to`` clamps only the current window,
        so the offset grows and retains rejected-draft K/V. Here ``base`` is
        advanced by the gap, keeping ``slot = position - base`` exact.

        Records this window's seed as the new reference (nothing written yet);
        callers update ``_mtp_persist_written_by_row`` after drafting.

        Args:
            row: The request's stable drafter row.
            seed_position: Seed position of the window about to be drafted.

        Returns:
            Number of leading slots of ``row`` that hold committed context.
        """
        seed_position = int(seed_position)
        base = int(self._mtp_persist_base_pos_by_row.get(row, seed_position))
        last_seed = self._mtp_persist_last_seed_by_row.get(row)
        written = self._mtp_persist_written_by_row.get(row)
        if last_seed is not None and written is not None:
            gap = (seed_position - int(last_seed)) - int(written)
            if gap > 0:
                base += gap
        self._mtp_persist_base_pos_by_row[row] = base
        self._mtp_persist_last_seed_by_row[row] = seed_position
        self._mtp_persist_written_by_row[row] = 0
        return max(0, seed_position - base)

    def _mtp_persist_checkpoint_skipped_window(self, req_id: str, seed_position: int) -> None:
        """Record a window whose draft was skipped (reject backoff) for persist bookkeeping.

        The skipped call's ``seed_position`` is the first post-verify seed of the
        previous drafted window, i.e. the only point where that window's accepted
        count is known. Checkpointing it (host-only; the device rollback happens
        at the next real draft) lets the next window drop the previous window's
        rejected-draft slots instead of keeping its whole written extent.

        Args:
            req_id: Request whose draft was skipped.
            seed_position: Seed position of the skipped window.
        """
        if not self._mtp_persist_enabled():
            return
        row = self._mtp_row_by_req.get(str(req_id))
        if row is None or self._mtp_persist_req_by_row.get(row) != str(req_id):
            return
        self._mtp_persist_committed_len(row, int(seed_position))

    @property
    def uses_recurrent_candidates(self) -> bool:
        """Whether hybrid recurrent caches keep speculative candidate rows."""
        return bool(getattr(self._runner, "spec_decode_recurrent_candidates", False))

    def resolve_step_draft_budget(self, concurrency: int) -> int:
        """Resolve this step's draft-token budget from the live concurrency.

        Called ONCE per scheduler step by the runner (never per request) with
        the number of requests scheduled in the step. Drafts produced during a
        step are verified on the NEXT step at approximately the same
        concurrency, so the step's scheduled-request count is the natural
        predictor. Gap semantics follow
        :func:`~easydel.inference.esurge.config.resolve_num_draft_tokens`:
        concurrencies no range covers fall back to the static
        ``num_draft_tokens``. The ``EASYDEL_DISABLE_DYNAMIC_SPEC`` flag (read
        at call time) forces the static count for A/B comparisons.

        A budget of ``0`` makes every draft path a no-op for the step: no
        drafter forward runs, requests carry no spec tokens into the next
        step, and the next step therefore takes the plain single-token decode
        path. No hysteresis is applied — the inclusive ranges provide natural
        bands; add a hysteresis knob only if oscillation shows up in practice.

        Args:
            concurrency: Number of requests scheduled in this step.

        Returns:
            The resolved live budget in ``[0, num_draft_tokens]`` (also stored
            on :attr:`live_num_draft_tokens`).
        """
        if self.draft_schedule is None or flags.get_bool(flags.EASYDEL_DISABLE_DYNAMIC_SPEC):
            self.live_num_draft_tokens = int(self.num_draft_tokens)
        else:
            self.live_num_draft_tokens = resolve_num_draft_tokens(
                self.draft_schedule,
                batch_size=int(concurrency),
                num_draft_tokens=self.num_draft_tokens,
            )
        return self.live_num_draft_tokens

    def on_reset(self) -> None:
        """Clear per-request speculative state and reset the drafter.

        Mirrors the speculative portion of ``eSurgeRunner.reset_state``.
        """
        self.draft_full_log_probs_by_req.clear()
        self._backoff_by_req.clear()
        self._mtp_persist_base_pos_by_row.clear()
        self._mtp_persist_req_by_row.clear()
        self._mtp_persist_last_seed_by_row.clear()
        self._mtp_persist_written_by_row.clear()
        self._mtp_row_by_req.clear()
        self._mtp_row_free = []
        self._mtp_row_capacity = 0
        self._whole_block_hidden = None
        self._whole_block_window.clear()
        self.live_num_draft_tokens = int(self.num_draft_tokens)
        if self.drafter is not None:
            try:
                # Size to the request pool (``max_num_reqs`` == sequence-buffer
                # rows), not ``max_num_seqs`` — per-request persistence keys the MTP
                # cache row on ``req_idx``, which ranges over the full pool.
                self.drafter.reset(batch_size=max(1, int(self._runner.max_num_reqs)))
            except Exception:
                logger.debug("Speculative drafter reset failed.", exc_info=True)

    def is_greedy_request(self, req_state: CachedRequestState | None) -> bool:
        """Return whether speculative decoding should use the greedy path for ``req_state``.

        Args:
            req_state: The cached request state to inspect; ``None`` short-circuits to False.

        Returns:
            ``True`` when a drafter is attached and the request's sampling
            temperature is non-positive (greedy).
        """
        if self.drafter is None or req_state is None:
            return False
        sampling_params = req_state.sampling_params
        temperature = float(getattr(sampling_params, "temperature", 0.0) or 0.0)
        return temperature <= 0.0

    def is_spec_request(self, req_state: CachedRequestState | None) -> bool:
        """Return whether ``req_state`` is eligible for speculative decoding.

        Requests using sampling penalties, ``n > 1``, or no attached drafter
        fall back to the standard single-token decode path because the runner's
        spec-verify kernels don't support those features yet.

        Args:
            req_state: Cached request state to inspect.

        Returns:
            ``True`` if the drafter is set, ``num_draft_tokens > 0``, and
            the request uses default-shaped sampling parameters; ``False``
            otherwise.
        """
        if self.drafter is None or req_state is None or self.num_draft_tokens <= 0:
            return False
        sampling_params = req_state.sampling_params
        if getattr(sampling_params, "n", 1) != 1:
            return False
        if getattr(sampling_params, "presence_penalty", 0.0) != 0.0:
            return False
        if getattr(sampling_params, "frequency_penalty", 0.0) != 0.0:
            return False
        if getattr(sampling_params, "repetition_penalty", 1.0) != 1.0:
            return False
        return True

    def filtered_log_probs(
        self,
        logits_or_log_probs: jax.Array,
        req_state: CachedRequestState,
    ) -> jax.Array:
        """Apply the request's filtering+temperature and return log probabilities.

        Used by the eager (non-jit) speculative paths to score draft tokens
        against the target distribution under the same top-k / top-p / min-p
        filters that the runner sampler would have applied for this request.

        Args:
            logits_or_log_probs: Per-row logits (or already-log probabilities)
                with shape ``[..., vocab_size]``.
            req_state: Request state providing the sampling parameters.

        Returns:
            Log-softmax values of the filtered/scaled logits along the last
            axis, ``float32``.
        """
        sampling_params = req_state.sampling_params
        logits = logits_or_log_probs.astype(jnp.float32)
        min_val = -1e10

        top_k = int(getattr(sampling_params, "top_k", 0) or 0)
        if top_k > 0:
            logits = apply_topk_mask(logits, jnp.asarray(top_k, dtype=jnp.int32), min_val)

        top_p = float(getattr(sampling_params, "top_p", 1.0) or 1.0)
        if top_p < 1.0:
            logits = apply_topp_mask(logits, jnp.asarray(top_p, dtype=jnp.float32), min_val)

        temperature = float(getattr(sampling_params, "temperature", 1.0) or 1.0)
        if temperature > 0.0 and temperature != 1.0:
            logits = logits / jnp.asarray(temperature, dtype=jnp.float32)

        min_p = float(getattr(sampling_params, "min_p", 0.0) or 0.0)
        if min_p > 0.0:
            # Min-p in PROBABILITY space (matches the production sampler's
            # `apply_min_p_mask`): keep tokens whose probability is
            # >= min_p * max_prob. Doing this on raw logits is a different,
            # wrong criterion; `min_p == 0` is gated out above so it stays a no-op.
            probs = jax.nn.softmax(logits, axis=-1)
            max_probs = jnp.max(probs, axis=-1, keepdims=True)
            logits = jnp.where(
                probs >= (jnp.asarray(min_p, dtype=jnp.float32) * max_probs),
                logits,
                jnp.full_like(logits, min_val),
            )

        return jax.nn.log_softmax(logits, axis=-1)

    def rng_split(self, n: int = 1) -> list[jax.Array]:
        """Advance the manager-owned RNG by ``n`` and return ``n`` fresh keys.

        Splits ``executor_manager.rng_key`` into ``n + 1`` subkeys, keeps
        the first as the new state, and hands out the remaining ``n``.

        Args:
            n: Number of fresh keys to return.

        Returns:
            List of ``n`` fresh ``jax.Array`` random keys.
        """
        keys = jax.random.split(self._runner.executor_manager.rng_key, int(n) + 1)
        self._runner.executor_manager.rng_key = keys[0]
        return list(keys[1:])

    def sample_distribution(self, log_probs: jax.Array, req_state: CachedRequestState) -> int:
        """Sample one token id from ``log_probs`` using the request's mode.

        Returns the argmax for greedy requests; otherwise draws a categorical
        sample with a freshly split RNG key.

        Args:
            log_probs: Per-position log probabilities with vocabulary as the
                last axis.
            req_state: Cached request state used to detect the greedy path.

        Returns:
            Sampled token id as a Python int.
        """
        if self.is_greedy_request(req_state):
            return int(np.asarray(jnp.argmax(log_probs, axis=-1)).reshape(-1)[0])
        key = self.rng_split(1)[0]
        return int(np.asarray(jax.random.categorical(key, log_probs)).reshape(-1)[0])

    @staticmethod
    def filter_logits_for_sampling(
        logits_or_log_probs: jax.Array,
        *,
        temperature: jax.Array,
        top_p: jax.Array,
        top_k: jax.Array,
        min_p: jax.Array,
    ) -> jax.Array:
        """Apply the runner's supported sampling filters before log-softmax.

        Conditionally applies top-k masking, top-p (nucleus) masking, temperature
        scaling, and min-p masking using ``jax.lax.cond`` so the function stays
        jit-traceable with scalar gate values. Filters that are at their no-op
        value (``top_k <= 0``, ``top_p >= 1``, ``temperature <= 0``,
        ``min_p <= 0``) are skipped.

        Args:
            logits_or_log_probs: Logits (or log probabilities) with shape
                ``[..., vocab_size]``; cast to float32 internally.
            temperature: Scalar device array temperature divisor.
            top_p: Scalar device array nucleus probability threshold.
            top_k: Scalar device array top-k count.
            min_p: Scalar device array min-p relative threshold.

        Returns:
            Filtered logits (float32) with masked entries set to a large
            negative value, ready for ``log_softmax``.
        """
        logits = logits_or_log_probs.astype(jnp.float32)
        min_val = -1e10

        logits = jax.lax.cond(
            top_k > 0,
            lambda legi: apply_topk_mask(legi, top_k.astype(jnp.int32), min_val),
            lambda legi: legi,
            logits,
        )
        logits = jax.lax.cond(
            top_p < 1.0,
            lambda legi: apply_topp_mask(legi, top_p.astype(jnp.float32), min_val),
            lambda legi: legi,
            logits,
        )
        logits = jax.lax.cond(
            temperature > 0.0,
            lambda legi: legi / temperature.astype(jnp.float32),
            lambda legi: legi,
            logits,
        )

        def _apply_min_p(legi):
            """Mask tokens whose probability is below ``min_p * max_prob``.

            Min-p in PROBABILITY space, matching the production sampler's
            ``apply_min_p_mask`` (``prob >= min_p * max_prob``); the ``min_p == 0``
            case is a no-op via the ``jax.lax.cond`` gate below.
            """
            probs = jax.nn.softmax(legi, axis=-1)
            max_probs = jnp.max(probs, axis=-1, keepdims=True)
            return jnp.where(
                probs >= (min_p.astype(jnp.float32) * max_probs),
                legi,
                jnp.full_like(legi, min_val),
            )

        return jax.lax.cond(min_p > 0.0, _apply_min_p, lambda legi: legi, logits)

    def sampling_args(
        self,
        req_state: CachedRequestState,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Pack a request's sampling parameters into device scalars for spec kernels.

        Args:
            req_state: Cached request state to read sampling parameters from.

        Returns:
            Tuple of ``(temperature, top_p, top_k, min_p)`` device scalars in
            the dtypes expected by :meth:`filter_logits_for_sampling`.
        """
        sampling_params = req_state.sampling_params
        temperature = float(getattr(sampling_params, "temperature", 1.0) or 1.0)
        top_p = float(getattr(sampling_params, "top_p", 1.0) or 1.0)
        top_k = int(getattr(sampling_params, "top_k", 0) or 0)
        min_p = float(getattr(sampling_params, "min_p", 0.0) or 0.0)
        return (
            jnp.asarray(temperature, dtype=jnp.float32),
            jnp.asarray(top_p, dtype=jnp.float32),
            jnp.asarray(top_k, dtype=jnp.int32),
            jnp.asarray(min_p, dtype=jnp.float32),
        )

    def filter_and_sample_log_probs(
        self,
        logits_or_log_probs: jax.Array,
        req_state: CachedRequestState,
    ) -> tuple[jax.Array, jax.Array]:
        """Return sampled token IDs and filtered log-probs using one cached JIT.

        Looks up (or compiles and caches, keyed by shape+dtype) a jitted filter +
        categorical sampler, then runs it with the request's sampling scalars and
        a freshly split RNG key.

        Args:
            logits_or_log_probs: Per-row logits (or log probabilities) with shape
                ``[..., vocab_size]``.
            req_state: Request state providing the sampling parameters.

        Returns:
            Tuple ``(token_ids, log_probs)`` where ``token_ids`` are the sampled
            int32 ids and ``log_probs`` are the filtered log-softmax values.
        """
        key = (tuple(logits_or_log_probs.shape), str(logits_or_log_probs.dtype))
        fn = self._filter_sample_fns.get(key)
        if fn is None:

            @jax.jit
            def _fn(logits, rng_key, temperature, top_p, top_k, min_p):
                """JIT-compiled filter + categorical sampler for spec drafts."""
                filtered = DrafterSpeculation.filter_logits_for_sampling(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
                log_probs = jax.nn.log_softmax(filtered, axis=-1)
                token_ids = jax.random.categorical(rng_key, log_probs).astype(jnp.int32)
                return token_ids, log_probs

            fn = _fn
            self._filter_sample_fns[key] = fn

        temperature, top_p, top_k, min_p = self.sampling_args(req_state)
        return fn(
            logits_or_log_probs,
            self.rng_split(1)[0],
            temperature,
            top_p,
            top_k,
            min_p,
        )

    def verify_sampled_window(
        self,
        *,
        target_logits: jax.Array,
        draft_log_probs: jax.Array,
        draft_tokens: list[int],
        req_state: CachedRequestState,
    ) -> tuple[int, int]:
        """Compiled rejection sampler for one non-greedy spec window.

        Looks up (or compiles and caches by shape) a jitted speculative
        rejection sampler that verifies each draft token against the target
        distribution, drawing a residual-distribution replacement on the first
        rejection and a bonus token if all drafts are accepted.

        Args:
            target_logits: Target-model logits for the draft + bonus positions,
                shape ``[num_drafts + 1, vocab_size]``.
            draft_log_probs: Draft-model log probabilities for the draft
                positions, shape ``[num_drafts, vocab_size]``.
            draft_tokens: The drafted token ids being verified.
            req_state: Request state providing the sampling parameters.

        Returns:
            Tuple ``(accepted, corrected)`` where ``accepted`` is the number of
            leading accepted draft tokens and ``corrected`` is the replacement
            or bonus token id (both Python ints).
        """
        draft_token_arr = jnp.asarray(draft_tokens, dtype=jnp.int32)
        key = (
            tuple(target_logits.shape),
            str(target_logits.dtype),
            tuple(draft_log_probs.shape),
            str(draft_log_probs.dtype),
            tuple(draft_token_arr.shape),
        )
        fn = self._verify_sample_fns.get(key)
        if fn is None:

            @jax.jit
            def _fn(target_logits_i, draft_log_probs_i, draft_token_ids, rng_key, temperature, top_p, top_k, min_p):
                """JIT-compiled rejection sampler for one non-greedy spec window."""
                target_filtered = DrafterSpeculation.filter_logits_for_sampling(
                    target_logits_i,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
                target_log_probs = jax.nn.log_softmax(target_filtered, axis=-1)
                num_drafts = draft_token_ids.shape[0]
                keys = jax.random.split(rng_key, num_drafts * 2 + 1)

                def _resample_residual(t_lp, d_lp, key_i):
                    """Draw a single replacement from the ``(p_target - p_draft)+`` residual."""
                    p_target = jnp.exp(t_lp)
                    p_draft = jnp.exp(d_lp)
                    residual = jnp.maximum(p_target - p_draft, 0.0)
                    residual = residual / jnp.maximum(jnp.sum(residual, axis=-1, keepdims=True), 1e-9)
                    return jax.random.categorical(key_i, jnp.log(residual + 1e-9)).astype(jnp.int32)[0]

                def _body(i, carry):
                    """One rejection-sampling step over draft index ``i`` carrying accept state."""
                    accepted, corrected, rejected = carry
                    tok = draft_token_ids[i]
                    target_full = jax.lax.dynamic_slice_in_dim(target_log_probs, i, 1, axis=0)
                    draft_full = jax.lax.dynamic_slice_in_dim(draft_log_probs_i, i, 1, axis=0)
                    tok_2d = tok.reshape(1, 1)
                    d_lp = jnp.take_along_axis(draft_full, tok_2d, axis=-1).squeeze()
                    t_lp = jnp.take_along_axis(target_full, tok_2d, axis=-1).squeeze()
                    accept_prob = jnp.minimum(1.0, jnp.exp(t_lp - d_lp))
                    accept = jax.random.uniform(keys[i], (), dtype=jnp.float32) < accept_prob
                    replacement = _resample_residual(target_full, draft_full, keys[num_drafts + i])
                    still_verifying = ~rejected
                    take_accept = still_verifying & accept
                    take_reject = still_verifying & (~accept)
                    accepted = accepted + take_accept.astype(jnp.int32)
                    corrected = jnp.where(take_reject, replacement, corrected)
                    rejected = rejected | take_reject
                    return accepted, corrected, rejected

                accepted, corrected, rejected = jax.lax.fori_loop(
                    0,
                    num_drafts,
                    _body,
                    (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)),
                )
                bonus = jax.random.categorical(keys[-1], target_log_probs[num_drafts : num_drafts + 1]).astype(
                    jnp.int32
                )[0]
                corrected = jnp.where(rejected, corrected, bonus)
                return accepted, corrected

            fn = _fn
            self._verify_sample_fns[key] = fn

        temperature, top_p, top_k, min_p = self.sampling_args(req_state)
        accepted, corrected = fn(
            target_logits,
            draft_log_probs,
            draft_token_arr,
            self.rng_split(1)[0],
            temperature,
            top_p,
            top_k,
            min_p,
        )
        return (
            int(np.asarray(accepted).reshape(-1)[0]),
            int(np.asarray(corrected).reshape(-1)[0]),
        )

    def build_verify_metadata(
        self,
        *,
        request_id: str,
        row_pos: int,
        req_idx: int,
        start_pos: int,
        real_count: int,
        scheduled_draft_tokens: list[int],
        token_ids_window_cpu: np.ndarray,
        token_offset: int,
    ) -> _SpecVerifyMetadata:
        """Build target-logit/draft-token verification indices.

        A hidden/logit row for token position ``p`` predicts the token at
        ``p + 1``. For a spec window containing ``real_count`` real tokens
        followed by ``N`` drafts, target rows are the local rows for
        ``start + real_count - 1 + i`` and draft tokens are read from the
        materialized input buffer at ``start + real_count + i``.

        Args:
            request_id: Identifier of the request being verified.
            row_pos: Window-local row index of the request.
            req_idx: Sequence-buffer row index of the request.
            start_pos: Absolute sequence position of the first scheduled token.
            real_count: Number of confirmed real tokens before the drafts.
            scheduled_draft_tokens: Draft token ids the scheduler provided.
            token_ids_window_cpu: CPU token buffer for the active window, indexed
                by ``row_pos`` to recover the materialized draft tokens.
            token_offset: Flattened token offset of this row within the window
                (start of its hidden/logit rows).

        Returns:
            A :class:`_SpecVerifyMetadata` capturing the target/bonus local
            indices, draft token positions, and both the scheduled and the
            buffer-materialized draft tokens.
        """
        num_drafts = len(scheduled_draft_tokens)
        target_start = int(token_offset) + int(real_count) - 1
        target_local_indices = [target_start + i for i in range(num_drafts)]
        bonus_local_index = target_start + num_drafts
        draft_token_positions = [int(start_pos) + int(real_count) + i for i in range(num_drafts)]

        row_tokens = np.asarray(token_ids_window_cpu[int(row_pos)])
        buffer_draft_tokens: list[int] = []
        for i, pos in enumerate(draft_token_positions):
            if 0 <= pos < int(row_tokens.shape[0]):
                buffer_draft_tokens.append(int(row_tokens[pos]))
            else:
                buffer_draft_tokens.append(int(scheduled_draft_tokens[i]))

        return _SpecVerifyMetadata(
            request_id=str(request_id),
            row_pos=int(row_pos),
            req_idx=int(req_idx),
            start_pos=int(start_pos),
            real_count=int(real_count),
            target_local_indices=target_local_indices,
            bonus_local_index=int(bonus_local_index),
            draft_token_positions=draft_token_positions,
            scheduled_draft_tokens=[int(t) for t in scheduled_draft_tokens],
            buffer_draft_tokens=buffer_draft_tokens,
        )

    def record_verify_trace(
        self,
        *,
        meta: _SpecVerifyMetadata,
        logits_rows: jax.Array | np.ndarray | None,
        accepted: int,
        corrected_token: int,
        greedy: bool,
        source: str,
    ) -> None:
        """Record a small host trace for speculative acceptance diagnosis.

        No-op unless ``debug_max_traces`` is positive and the trace
        budget is not yet exhausted. Builds a per-draft diagnostic record
        (scheduler-vs-buffer token match, target argmax, and the draft token's
        rank under the target distribution) and appends it to
        ``debug_traces`` while logging it at WARNING.

        Args:
            meta: Verification metadata describing the window's draft tokens.
            logits_rows: Per-draft target logits (jax/numpy) used to compute
                argmax/rank diagnostics, or ``None`` to skip those fields.
            accepted: Number of accepted draft tokens for this window.
            corrected_token: Replacement or bonus token id chosen by verify.
            greedy: Whether the request used the greedy spec path.
            source: Tag identifying the verify code path (e.g. ``"full-hidden"``
                or ``"replay"``).
        """
        if self.debug_max_traces <= 0:
            return
        if len(self.debug_traces) >= self.debug_max_traces:
            return

        rows: list[dict[str, typing.Any]] = []
        logits_np = np.asarray(logits_rows) if logits_rows is not None else None
        for i, draft_token in enumerate(meta.buffer_draft_tokens):
            row: dict[str, typing.Any] = {
                "draft_index": int(i),
                "target_local_index": int(meta.target_local_indices[i]),
                "draft_token_position": int(meta.draft_token_positions[i]),
                "scheduled_draft_token": int(meta.scheduled_draft_tokens[i]),
                "buffer_draft_token": int(draft_token),
                "scheduler_buffer_match": bool(int(meta.scheduled_draft_tokens[i]) == int(draft_token)),
            }
            if logits_np is not None and i < int(logits_np.shape[0]):
                row_logits = logits_np[i]
                draft_id = int(draft_token)
                target_argmax = int(np.argmax(row_logits))
                row["target_argmax"] = target_argmax
                if 0 <= draft_id < int(row_logits.shape[-1]):
                    draft_logit = float(row_logits[draft_id])
                    row["draft_rank_under_target"] = int(np.count_nonzero(row_logits > draft_logit) + 1)
                else:
                    row["draft_rank_under_target"] = None
            rows.append(row)

        trace = {
            "request_id": meta.request_id,
            "row_pos": int(meta.row_pos),
            "req_idx": int(meta.req_idx),
            "start_pos": int(meta.start_pos),
            "real_count": int(meta.real_count),
            "bonus_local_index": int(meta.bonus_local_index),
            "accepted": int(accepted),
            "corrected_token": int(corrected_token),
            "greedy": bool(greedy),
            "source": source,
            "rows": rows,
        }
        self.debug_traces.append(trace)
        logger.warning("Spec decode verify trace: %s", trace)

    def on_states_applied(self, scheduler_output: SchedulerOutput) -> None:
        """Write scheduler-provided speculative tokens into the sequence buffer.

        For each request that has scheduled draft tokens this step, writes the
        prefix of confirmed real tokens followed by the draft tokens into the
        request's row in the sequence buffer. Updates ``num_tokens_no_spec`` so
        the runner remembers where verified tokens end and drafts begin.

        Args:
            scheduler_output: Scheduler decision carrying the per-request
                ``scheduled_spec_decode_tokens`` mapping.
        """
        if self.drafter is None:
            return
        self._release_finished_drafter_rows(scheduler_output)
        if not scheduler_output.scheduled_spec_decode_tokens:
            return
        for req_id, spec_tokens in scheduler_output.scheduled_spec_decode_tokens.items():
            if not spec_tokens:
                continue
            req_state = self._runner.requests.get(req_id)
            req_idx = self._runner.sequence_buffer.req_id_to_index.get(req_id)
            if req_state is None or req_idx is None:
                continue
            scheduled = int(scheduler_output.num_scheduled_tokens.get(req_id, 0))
            start = int(self._runner.sequence_buffer.num_computed_tokens[req_idx])
            real_count = scheduled - len(spec_tokens)
            if scheduled <= 0 or real_count <= 0:
                continue
            all_known_tokens = list(req_state.prompt_token_ids) + list(req_state.output_token_ids)
            real_tokens = all_known_tokens[start : start + real_count]
            if len(real_tokens) != real_count:
                logger.debug(
                    "Skipping spec-token materialization for %s: need %d real tokens at %d, have %d.",
                    req_id,
                    real_count,
                    start,
                    len(real_tokens),
                )
                continue
            full_tokens = real_tokens + [int(t) for t in spec_tokens]
            end = start + len(full_tokens)
            if end > self._runner.max_model_len:
                continue
            self._runner.sequence_buffer.token_ids[req_idx, start:end] = np.asarray(full_tokens, dtype=np.int32)
            self._runner.sequence_buffer.num_tokens_no_spec[req_idx] = start + real_count
            self._runner.sequence_buffer.num_tokens[req_idx] = end

    def project_hidden_rows(self, hidden_rows: jax.Array) -> jax.Array:
        """Project arbitrary-row hidden states through the LM head for spec verify.

        The standard runner LM head is compiled for the active ``padded_num_reqs``
        bucket; speculative verification often inspects a different number of
        rows. This method looks up (or compiles on demand) an LM-head executable
        sized for the actual ``hidden_rows`` and returns the resulting logits.

        Args:
            hidden_rows: Hidden-state rows shaped ``[num_rows, hidden_dim]``.

        Returns:
            Vocabulary logits of shape ``[num_rows, vocab_size]``.
        """
        if bool(getattr(self.drafter, "needs_main_hidden", False)):
            # DSpark: model-step rows are ``last_hidden | main_hidden``; the LM head only sees the first part.
            hidden_rows = hidden_rows[..., : int(self._runner.model.config.get_text_config().hidden_size)]
        return self._runner.executor_manager.project_hidden_rows(hidden_rows)

    def cache_group_index_for_layer(self, layer_idx: int) -> int | None:
        """Resolve the eSurge page-table group that owns a target layer's K/V.

        Args:
            layer_idx: Zero-based transformer layer index to look up.

        Returns:
            The cache-group index whose ``layer_names`` contains
            ``"layer.{layer_idx}"`` (or ``0`` when there are no groups / a single
            unnamed group), or ``None`` when no group claims the layer.
        """
        if not self._runner.kv_cache_groups:
            return 0
        target = f"layer.{int(layer_idx)}"
        for group_idx, group in enumerate(self._runner.kv_cache_groups):
            layer_names = getattr(group, "layer_names", None)
            if layer_names is None and len(self._runner.kv_cache_groups) == 1:
                return group_idx
            if layer_names is not None and target in set(layer_names):
                return group_idx
        return None

    def gather_paged_kv_for_request(
        self,
        view: typing.Any,
        *,
        group_idx: int,
        req_idx: int,
        kv_len: int,
    ) -> tuple[jax.Array, jax.Array] | None:
        """Gather contiguous K/V slices from paged storage for one request.

        Reads the request's page-table row for ``group_idx``, then assembles
        a dense ``[1, kv_len, ...]`` key/value pair by indexing
        ``view.key_pages`` / ``view.value_pages`` with the request's logical
        token positions. Used by speculative drafters that need a contiguous
        target-side K/V cache view for cross-attention.

        Args:
            view: Cache view exposing ``key_pages`` and ``value_pages``.
            group_idx: KV cache page-table group index that owns the layer.
            req_idx: Request row index in the sequence buffer.
            kv_len: Number of K/V positions to gather (clamped to actual size).

        Returns:
            Tuple ``(keys, values)`` of arrays with a leading batch axis of 1,
            or ``None`` when the view lacks paged K/V or no pages exist.
        """
        page_table = self._runner.sequence_buffer.page_table[group_idx]
        page_table_cpu = page_table.get_cpu_tensor()
        page_size = int(
            getattr(getattr(view, "metadata", None), "page_size", self._runner.page_size) or self._runner.page_size
        )
        key_pages = getattr(view, "key_pages", None)
        value_pages = getattr(view, "value_pages", None)
        if key_pages is None or value_pages is None or int(page_table_cpu.shape[1]) == 0:
            return None
        return gather_paged_kv(
            key_pages,
            value_pages,
            page_table_cpu,
            batch_index=int(req_idx),
            kv_len=int(kv_len),
            page_size=page_size,
        )

    def target_kv_payload_for_drafter(
        self,
        *,
        req_id: str,
        seed_position: int,
    ) -> tuple[list[tuple[jax.Array, jax.Array] | None], jax.Array] | None:
        """Build target K/V pairs + static attention mask for an assistant drafter.

        Used by drafters that cross-attend to the target model's cache (e.g. the
        Gemma4 assistant). Maps each drafter layer to a target layer, gathers
        that layer's K/V (either directly from the view or by paging in from
        ragged storage), and builds a causal-style mask covering positions up to
        ``seed_position``.

        Args:
            req_id: Request identifier whose target K/V cache should be read.
            seed_position: Absolute position of the spec window's seed token,
                used to bound the visible context length.

        Returns:
            Tuple ``(pairs, attention_mask)`` where ``pairs`` is one
            ``(keys, values)`` tuple (or ``None``) per drafter layer and
            ``attention_mask`` is a ``[1, 1, 1, kv_len]`` additive float mask (or
            the drafter's ``build_target_attention_mask`` result, e.g. a dict of
            per-attention-type masks), or
            ``None`` when the drafter does not require a target K/V cache, the
            request is unknown, the cache has no views, or no layer yielded K/V.
        """
        if self.drafter is None or not bool(getattr(self.drafter, "requires_target_kv_cache", False)):
            return None

        req_idx = self._runner.sequence_buffer.req_id_to_index.get(req_id)
        if req_idx is None:
            return None
        target_cache = self._runner.executor_manager.kv_pages
        views = getattr(target_cache, "views", None)
        if views is None:
            return None

        resolver = getattr(self.drafter, "resolve_layer_mapping", None)
        if callable(resolver):
            layer_mapping = resolver(target_cache)
        else:
            layer_mapping = list(range(len(views)))
        kv_len = int(self._runner.max_model_len)
        target_context_len = max(1, min(kv_len, int(seed_position) + 1))

        layer_mapping = list(layer_mapping)
        layer_to_group: dict[int, int] = {}
        for layer in layer_mapping:
            if not 0 <= int(layer) < len(views):
                continue
            view = getattr(views[int(layer)], "transformer", views[int(layer)])
            if getattr(view, "key_pages", None) is not None:
                group = self.cache_group_index_for_layer(int(layer))
                if group is not None:
                    layer_to_group[int(layer)] = group
        page_table = getattr(self._runner.sequence_buffer, "page_table", None)
        pairs = build_target_kv_pairs(
            target_cache,
            layer_mapping,
            page_tables=getattr(page_table, "page_tables", page_table),
            layer_to_group=layer_to_group,
            batch_index=int(req_idx),
            kv_len=kv_len,
            page_size=int(getattr(self._runner, "page_size", 1)),
        )

        if not any(pair is not None for pair in pairs):
            return None
        mask_builder = getattr(self.drafter, "build_target_attention_mask", None)
        if callable(mask_builder):
            # Per-attention-type masks (e.g. the Gemma4 assistant's sliding
            # layers only see the last ``sliding_window + 1`` positions).
            return pairs, mask_builder(kv_len=kv_len, context_len=target_context_len)
        mask_positions = jnp.arange(kv_len, dtype=jnp.int32)[None, None, None, :]
        attention_mask = jnp.where(
            mask_positions < jnp.asarray(target_context_len, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(-1.0e10, dtype=jnp.float32),
        )
        return pairs, attention_mask

    def _mtp_persist_enabled(self) -> bool:
        """Whether the inline-MTP drafter should persist its KV across windows.

        Persist-with-rollback keeps the accepted-context K/V across verify
        windows and only rolls it back to the committed position each window
        (EAGLE / vLLM style), instead of a per-window full clear. It follows the
        drafter's ``persist_kv`` flag (default ``True`` for the inline MTP
        drafter — without it acceptance collapses with draft depth) and requires
        the drafter to expose ``rollback_to``. The ``EASYDEL_MTP_PERSIST_KV``
        env var overrides the flag in either direction: ``"1"`` forces
        persistence on, ``"0"`` forces the historical per-window clear.

        Returns:
            ``True`` when persistence should be used for the attached drafter.
        """
        if self.drafter is None:
            return False
        if not callable(getattr(self.drafter, "rollback_to", None)):
            return False
        override = flags.get_str(flags.EASYDEL_MTP_PERSIST_KV)
        if isinstance(override, str) and override.strip():
            return flags.get_bool(flags.EASYDEL_MTP_PERSIST_KV)
        return bool(getattr(self.drafter, "persist_kv", False))

    def _adaptive_conf_threshold(self) -> float:
        """Return the adaptive (early-exit) draft confidence threshold.

        Configured by ``adaptive_conf_threshold`` (a top-1 softmax-probability
        threshold). Zero means OFF: :meth:`draft_next` uses the fixed-K Python
        draft loop. A positive value routes greedy, non-prefix drafts through
        :meth:`Qwen3_5MTPDrafter.draft_adaptive`, whose on-device ``while_loop``
        stops early once confidence drops below this threshold.

        Returns:
            The configured threshold; zero disables early exit.
        """
        return self.adaptive_conf_threshold

    def draft_next(
        self,
        *,
        req_id: str,
        seed_token: int,
        seed_position: int,
        seed_hidden: jax.Array,
        req_state: CachedRequestState | None,
        prefix_input_ids: jax.Array | None = None,
        prefix_hidden_states: jax.Array | None = None,
        prefix_position_ids: jax.Array | None = None,
        row_pos: int = 0,
    ) -> list[int]:
        """Generate up to ``num_draft_tokens`` draft tokens for one request.

        Runs the attached drafter from ``seed_token`` / ``seed_hidden`` at
        ``seed_position``. When the drafter supports prefix drafting and a
        ``prefix_*`` set is provided, the prefix is fed in instead of the
        single seed token to stabilize generation.

        Args:
            req_id: Identifier of the request being drafted for.
            seed_token: Token id at ``seed_position`` (the spec window's seed).
            seed_position: Absolute position of ``seed_token`` in the sequence.
            seed_hidden: Hidden state at ``seed_position``.
            req_state: Cached request state used to look up sampling parameters
                and gate spec eligibility.
            prefix_input_ids: Optional prefix token ids for prefix-drafting.
            prefix_hidden_states: Optional prefix hidden states matching
                ``prefix_input_ids``.
            prefix_position_ids: Optional prefix position ids.
            row_pos: The request's stable KV-pool slot (the runner's ``req_idx``,
                which is constant for the request's lifetime — unlike the runner's
                window-local row index that aliases across windows). Used as the
                request's row in the shared inline-MTP cache when persist-with-
                rollback is enabled, so each concurrent request keeps its own MTP
                K/V across verify windows. Ignored when persistence is off.

        Returns:
            List of drafted token ids (length may be less than the maximum on
            drafter failure or backoff). The window drafts
            :attr:`live_num_draft_tokens` tokens (the per-step dynamic budget;
            equal to the static ``num_draft_tokens`` without a schedule) —
            a live budget of ``0`` returns ``[]`` without any drafter call.
        """
        k_live = int(self.live_num_draft_tokens)
        if self.drafter is None or k_live <= 0 or not self.is_spec_request(req_state):
            return []
        backoff = int(self._backoff_by_req.get(str(req_id), 0))
        if backoff > 0:
            self._backoff_by_req[str(req_id)] = backoff - 1
            self._mtp_persist_checkpoint_skipped_window(str(req_id), int(seed_position))
            return []
        if bool(getattr(self.drafter, "whole_block_draft", False)):
            return self._draft_next_whole_block(
                req_id=str(req_id),
                seed_token=int(seed_token),
                seed_position=int(seed_position),
                req_state=req_state,
                row_pos=int(row_pos),
                k=k_live,
            )
        use_prefix = (
            prefix_input_ids is not None
            and prefix_hidden_states is not None
            and prefix_position_ids is not None
            and bool(getattr(self.drafter, "supports_prefix_draft", False))
        )
        # Block drafters (DSpark / DFlash) predict the whole window from a FIXED
        # anchor + target context: every call gets the anchor followed by the
        # tokens drafted so far (the Markov head's previous tokens), the same
        # target seed hidden, and never its own hidden state fed back.
        block_draft = bool(getattr(self.drafter, "block_draft", False)) and not use_prefix
        advance_position = bool(getattr(self.drafter, "advance_draft_position", True))
        cur_token = (
            jnp.asarray(prefix_input_ids, dtype=jnp.int32)
            if use_prefix
            else jnp.asarray([[int(seed_token)]], dtype=jnp.int32)
        )
        cur_position = int(seed_position) + int(getattr(self.drafter, "draft_position_offset", 0))
        seed_hidden_bsh = jnp.asarray(prefix_hidden_states) if use_prefix else jnp.asarray(seed_hidden)[None, None, :]
        drafted_dev: list[jax.Array] = []
        draft_fulls: list[jax.Array] = []
        greedy = self.is_greedy_request(req_state)
        target_kv_payload = self.target_kv_payload_for_drafter(
            req_id=req_id,
            seed_position=seed_position,
        )
        if bool(getattr(self.drafter, "requires_target_kv_cache", False)) and target_kv_payload is None:
            if not self._warned_missing_target_kv:
                self._warned_missing_target_kv = True
                logger.warning(
                    "Speculative assistant drafter got no target K/V payload (the mapped target layers own no "
                    "cache view — e.g. KV-shared layers — or the request is unknown); drafting is skipped. "
                    "Check the drafter's `layer_mapping`."
                )
            return []
        # Per-verify-window drafter cache boundary. Two policies:
        #
        # * Default (persist off, SP2): FULL clear each window via
        #   ``reset_request`` so a previous request's/window's K/V never pollutes
        #   this window's drafts — leaving it collapses acceptance. Stateless
        #   drafters expose no ``reset_request`` and are skipped.
        # * Persist-with-rollback (``EASYDEL_MTP_PERSIST_KV`` /
        #   drafter ``persist_kv=True``): KEEP the inline-MTP KV across windows
        #   within a request and only roll it back to the committed sequence
        #   position, dropping the previous window's rejected-draft K/V. Each
        #   window's MTP self-attention then sees the full accepted-sequence
        #   context (EAGLE / vLLM). This is now PER REQUEST: the MTP cache is
        #   sized to the request pool and each request owns its own row
        #   (``row_pos`` == the request's stable KV-pool slot), so interleaved
        #   ``draft_next`` calls for different concurrent requests never thrash
        #   each other's cache.
        reset_request = getattr(self.drafter, "reset_request", None)
        persist_on = self._mtp_persist_enabled()
        # ``row_pos`` (the runner's req_idx) is NOT stable across condense /
        # reorder row moves; use the strategy's lifetime-stable row instead so
        # the request's MTP context survives relocations (see __init__ note).
        row = self._stable_drafter_row(req_id, int(row_pos)) if persist_on else int(row_pos)
        if persist_on:
            # Size the pool cache to hold every concurrent request's row. This is
            # idempotent (allocates once, no-op thereafter) so a live request's
            # accumulated K/V is never wiped mid-generation.
            ensure = getattr(self.drafter, "ensure_batch_size", None)
            if callable(ensure):
                ensure(max(1, int(getattr(self._runner, "max_num_reqs", 1))))
            if self._mtp_persist_req_by_row.get(row) != str(req_id):
                # This row is newly assigned to this request — either a fresh
                # request or a slot reused by a different request. Drop only THIS
                # row's stale K/V (rollback to 0) and re-base its committed-origin
                # on the window's first written slot; other rows are untouched.
                # The first window has nothing to persist, so this matches the
                # per-window clear for that request.
                self._mtp_persist_req_by_row[row] = str(req_id)
                if use_prefix and prefix_position_ids is not None:
                    self._mtp_persist_base_pos_by_row[row] = int(np.asarray(prefix_position_ids).reshape(-1)[0])
                else:
                    self._mtp_persist_base_pos_by_row[row] = int(seed_position)
                self._mtp_persist_last_seed_by_row[row] = int(seed_position)
                self._mtp_persist_written_by_row[row] = 0
                self.drafter.rollback_to(0, row_pos=row)
                write_from = 0
            else:
                # Same request, later window: roll only this request's row back to
                # its committed position so the previous window's rejected-draft
                # K/V is dropped while the accepted context is retained.
                # ``seed_position`` advances with the accepted sequence; the
                # compacted committed length is its offset from that request's
                # (gap-adjusted) base, see ``_mtp_persist_committed_len``.
                committed_len = self._mtp_persist_committed_len(row, int(seed_position))
                self.drafter.rollback_to(committed_len, row_pos=row)
                write_from = int(committed_len)
            # Capacity guard: the persisted row is written from ``write_from`` on, and a dense transformer cache
            # clamps writes past its end onto the last slots (``dynamic_update_slice``), silently corrupting the
            # drafter context. Skip drafting for a window that would overflow; verification is unaffected and the
            # persist bookkeeping above already recorded this window (nothing written).
            capacity_fn = getattr(self.drafter, "mtp_cache_capacity", None)
            capacity = capacity_fn() if callable(capacity_fn) else None
            if capacity is not None:
                prefix_slots = int(np.asarray(prefix_position_ids).size) if use_prefix else 0
                if write_from + prefix_slots + int(k_live) > int(capacity):
                    return []
        elif callable(reset_request):
            reset_request(request_id=str(req_id))
        # Adaptive (early-exit) draft length: run the whole K_max recursion as one
        # on-device ``while_loop`` that stops early when the MTP's confidence drops,
        # so K_max can be large without paying for low-value deep drafts. Only the
        # greedy, non-prefix inline-MTP path is eligible (the drafter exposes
        # ``draft_adaptive``); prefix/first-window drafts, sampled drafts, and
        # non-MTP drafters keep the fixed-K Python loop below. Persist-with-rollback
        # is preserved: the per-request ``row`` and the (already rolled-back) MTP
        # cache are threaded through the loop.
        adaptive_conf = self._adaptive_conf_threshold()
        if (
            adaptive_conf > 0.0
            and greedy
            and not use_prefix
            and target_kv_payload is None
            and callable(getattr(self.drafter, "draft_adaptive", None))
        ):
            try:
                drafts_dev, valid_count_dev = self.drafter.draft_adaptive(
                    seed_token=int(seed_token),
                    seed_hidden=seed_hidden_bsh,
                    seed_position=int(seed_position),
                    k_max=k_live,
                    conf_threshold=adaptive_conf,
                    row_pos=(row if persist_on else None),
                )
            except Exception:
                logger.warning(
                    "Adaptive speculative drafter failed; disabling drafts for this request.",
                    exc_info=True,
                )
                return []
            # ONE batched device->host readback of the padded drafts + the valid
            # count; slice the valid prefix on the host (variable-length window).
            drafts_host, valid_host = jax.device_get((drafts_dev, valid_count_dev))
            valid = max(0, int(np.asarray(valid_host).reshape(-1)[0]))
            if persist_on:
                # One MTP cache slot per executed loop iteration.
                self._mtp_persist_written_by_row[row] = valid
            drafted = [int(t) for t in np.asarray(drafts_host).reshape(-1)[:valid]]
            self.num_drafts_generated += len(drafted)
            return drafted
        for draft_idx in range(k_live):
            position_ids = (
                jnp.asarray(prefix_position_ids, dtype=jnp.int32)
                if use_prefix and draft_idx == 0
                else jnp.asarray([[cur_position]], dtype=jnp.int32)
            )
            try:
                draft_kwargs = {
                    "input_ids": cur_token,
                    "target_hidden_states": seed_hidden_bsh,
                    "position_ids": position_ids,
                    "sample": False,
                    "rng_key": None,
                }
                if persist_on:
                    # Only the inline-MTP persist path understands per-request rows;
                    # other drafters keep their original single-row signature.
                    draft_kwargs["row_pos"] = row
                if not greedy and bool(getattr(self.drafter, "supports_return_full_log_probs", False)):
                    draft_kwargs["return_full_log_probs"] = True
                if target_kv_payload is not None:
                    target_kv_cache, attention_mask = target_kv_payload
                    draft_kwargs.update(
                        {
                            "target_kv_cache": target_kv_cache,
                            "attention_mask": attention_mask,
                        }
                    )
                step = self.drafter.draft(**draft_kwargs)
            except Exception:
                logger.warning("Speculative drafter failed; disabling drafts for this request.", exc_info=True)
                return []
            if persist_on:
                # Each chained draft call wrote one more slot at/after the seed slot
                # (the prefix call writes the prefix through the seed slot).
                self._mtp_persist_written_by_row[row] = draft_idx + 1
            if greedy:
                # Keep the drafted token on-device: the next chained MTP step
                # consumes it as a device array, so the K drafts pipeline back to
                # back without a blocking host round-trip each. The whole window is
                # read back to host once, after the loop (values are unchanged).
                tok_dev = step.token_ids.reshape(-1)[:1].astype(jnp.int32)
            else:
                if step.full_log_probs is None or req_state is None:
                    return []
                sampled_ids, filtered = self.filter_and_sample_log_probs(step.full_log_probs, req_state)
                draft_fulls.append(filtered[0])
                tok_dev = sampled_ids.reshape(-1)[:1].astype(jnp.int32)
            drafted_dev.append(tok_dev)
            if block_draft:
                cur_token = jnp.concatenate([cur_token, tok_dev.reshape(1, 1)], axis=1)
                continue
            cur_token = tok_dev.reshape(1, 1)
            if advance_position:
                cur_position += 1
            step_hidden = getattr(step, "hidden_states", None)
            if step_hidden is not None:
                seed_hidden_bsh = step_hidden[:, -1:, :]
        if not drafted_dev:
            return []
        # Single batched device->host transfer for the whole draft window (was one
        # blocking ``int(np.asarray(...))`` per draft token, which serialized the K
        # chained MTP forwards on the host). Argmax/sample values are unchanged.
        drafted = [int(t) for t in np.asarray(jnp.concatenate(drafted_dev, axis=0)).reshape(-1)]
        self.num_drafts_generated += len(drafted)
        if not greedy and draft_fulls:
            self.draft_full_log_probs_by_req[str(req_id)] = jnp.stack(draft_fulls, axis=0)
        return drafted

    @property
    def rolls_back_cache(self) -> bool:
        """Whether rejected drafts need snapshot-restore-replay of the (non-paged) per-slot cache (DSpark)."""
        return bool(getattr(self.drafter, "rolls_back_cache", False))

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed draft tokens the target accepted so far (``0.0`` before any draft)."""
        generated = int(self.num_drafts_generated)
        return float(self.num_drafts_accepted) / generated if generated else 0.0

    def observe_window_hidden(
        self,
        hidden_states: jax.Array | None,
        *,
        req_ids_window: list[str | None],
        scheduled_list: typing.Sequence[int],
        token_offsets: np.ndarray,
        start_positions: np.ndarray,
        scheduled_spec: dict[str, list[int]],
    ) -> None:
        """Hand the hidden rows of the step just executed to a whole-block drafter (DSpark).

        Registers, for every greedy spec-eligible request of the window, where its rows sit in ``hidden_states`` and
        lets the drafter write the main K/V of its committed (real, non-draft) tokens into its rings.

        Args:
            hidden_states: Full step hidden rows ``[N, (1 + T) * D]`` (``None`` when unavailable).
            req_ids_window: Window-local request ids (``None`` for idle rows).
            scheduled_list: Scheduled tokens per window row (real + draft tokens).
            token_offsets: First hidden row of every window row.
            start_positions: Position of every window row's first scheduled token.
            scheduled_spec: Scheduler draft tokens per request id.
        """
        self._whole_block_hidden = hidden_states
        self._whole_block_window = {}
        if self.drafter is None:
            return
        self.drafter.ensure_batch_size(max(1, int(getattr(self._runner, "max_num_reqs", 1))))
        hidden_len = 0 if hidden_states is None else int(hidden_states.shape[0])
        entries: list[tuple[str, int, int, int, int, int]] = []
        for row_pos, rid in enumerate(req_ids_window):
            if rid is None:
                continue
            count = int(scheduled_list[row_pos])
            req_state = self._runner.requests.get(rid)
            if count <= 0 or req_state is None:
                continue
            if not self.is_spec_request(req_state) or not self.is_greedy_request(req_state):
                continue
            offset = int(token_offsets[row_pos])
            start = int(start_positions[row_pos])
            real = max(0, count - len(scheduled_spec.get(rid, ())))
            req_idx = self._runner.sequence_buffer.req_id_to_index.get(rid, row_pos)
            row = self._stable_drafter_row(str(rid), int(req_idx))
            self._whole_block_window[str(rid)] = (offset, start, count)
            entries.append((str(rid), row, offset, start, real if offset + count <= hidden_len else -1, count))
        self.drafter.observe_window(hidden_states, entries)

    def _draft_next_whole_block(
        self,
        *,
        req_id: str,
        seed_token: int,
        seed_position: int,
        req_state: CachedRequestState | None,
        row_pos: int,
        k: int,
    ) -> list[int]:
        """Draft a whole window with a whole-block drafter from the hidden rows of the last observed step."""
        if not self.is_greedy_request(req_state):
            return []
        info = self._whole_block_window.get(req_id)
        hidden = self._whole_block_hidden
        if info is None or hidden is None:
            return []
        offset, start, count = info
        row = self._stable_drafter_row(req_id, row_pos)
        drafted = self.drafter.draft_window(
            req_id=req_id,
            row=row,
            seed_token=seed_token,
            seed_position=seed_position,
            hidden=hidden,
            offset=offset,
            start=start,
            n=count,
            k=k,
        )
        self.num_drafts_generated += len(drafted)
        return drafted

    def can_batch_whole_block_draft(self) -> bool:
        """Whether whole-block drafts (DSpark) of a window can be collected and drafted in one batched call."""
        if self.drafter is None or self.num_draft_tokens <= 0:
            return False
        if flags.get_bool(flags.EASYDEL_DISABLE_BATCHED_DRAFT):
            return False
        return bool(getattr(self.drafter, "whole_block_draft", False)) and callable(
            getattr(self.drafter, "draft_windows", None)
        )

    def draft_whole_block_batched(
        self,
        seeds: list[tuple[str, int, int, CachedRequestState | None, int]],
    ) -> dict[str, list[int]]:
        """Draft many requests with a whole-block drafter in one call (``drafter.draft_windows``).

        Applies, per request and in the same order, the checks :meth:`draft_next` applies before its whole-block branch
        (live budget, spec eligibility, reject backoff) and :meth:`_draft_next_whole_block`'s (greedy, observed window).

        Args:
            seeds: ``(req_id, seed_token, seed_position, req_state, row_pos)`` per request (``row_pos`` is the stable
                sequence-buffer slot, as for :meth:`draft_next`).

        Returns:
            ``{req_id: drafted token ids}`` for every seed (``[]`` where no draft is proposed).
        """
        out: dict[str, list[int]] = {str(seed[0]): [] for seed in seeds}
        k_live = int(self.live_num_draft_tokens)
        draft_windows = getattr(self.drafter, "draft_windows", None)
        hidden = self._whole_block_hidden
        if self.drafter is None or k_live <= 0 or not callable(draft_windows) or hidden is None:
            return out
        entries: list[tuple[str, int, int, int, int, int, int, int]] = []
        for req_id, seed_token, seed_position, req_state, row_pos in seeds:
            rid = str(req_id)
            if not self.is_spec_request(req_state):
                continue
            backoff = int(self._backoff_by_req.get(rid, 0))
            if backoff > 0:
                self._backoff_by_req[rid] = backoff - 1
                self._mtp_persist_checkpoint_skipped_window(rid, int(seed_position))
                continue
            if not self.is_greedy_request(req_state):
                continue
            info = self._whole_block_window.get(rid)
            if info is None:
                continue
            offset, start, count = info
            row = self._stable_drafter_row(rid, int(row_pos))
            entries.append((rid, row, int(seed_token), int(seed_position), offset, start, count, k_live))
        if not entries:
            return out
        for entry, drafted in zip(entries, draft_windows(hidden, entries), strict=True):
            out[entry[0]] = drafted
            self.num_drafts_generated += len(drafted)
        return out

    def can_batch_draft(self, req_state: CachedRequestState | None) -> bool:
        """Whether ``req_state`` can be drafted through the batched pool path.

        The batched drafter (:meth:`draft_next_batched`) runs the greedy inline-MTP
        recurrence for every spec-active request in a decode step together. A
        request is eligible only when the attached drafter exposes
        :meth:`~easydel.inference.speculative.Qwen3_5MTPDrafter.draft_batched`, the
        request is spec-eligible, and it is greedy (temperature ``<= 0``). Sampled
        requests, non-MTP drafters, and the prefix-draft first window fall back to
        the per-request :meth:`draft_next` path.

        Args:
            req_state: The cached request state to inspect.

        Returns:
            ``True`` when the request should be collected for the post-loop batched
            draft; ``False`` when it must use the per-request fallback.
        """
        if self.drafter is None or self.num_draft_tokens <= 0:
            return False
        # Escape hatch to force the legacy per-request draft path (benchmark A/B and
        # debugging); the batched pool path is the default.
        if flags.get_bool(flags.EASYDEL_DISABLE_BATCHED_DRAFT):
            return False
        if not bool(getattr(self.drafter, "supports_batched_draft", False)):
            return False
        if not callable(getattr(self.drafter, "draft_batched", None)):
            return False
        if bool(getattr(self.drafter, "requires_target_kv_cache", False)):
            # Cross-model (assistant) drafters need per-request target K/V payloads;
            # the pooled batched path only handles the inline-MTP cache.
            return False
        if not self.is_spec_request(req_state):
            return False
        return self.is_greedy_request(req_state)

    def draft_next_batched(
        self,
        seeds: list[tuple[str, int, int, int, jax.Array]],
    ) -> dict[str, list[int]]:
        """Draft ``k`` tokens for a batch of requests in ONE pooled MTP pass.

        Synchronous composition of :meth:`begin_batched_draft` (dispatch) and
        :meth:`finish_batched_draft` (readback); see those for the mechanism.

        Args:
            seeds: One ``(req_id, row, seed_token, seed_position, seed_hidden)`` per
                collected request, where ``row`` is the request's stable KV-pool
                slot (the runner ``req_idx``) and ``seed_hidden`` is a ``(H,)`` (or
                ``(1, 1, H)``) device array.

        Returns:
            ``{req_id: [drafted token ids]}`` for every input seed (an empty list
            for a request whose draft was suppressed by the reject-backoff gate).
        """
        return self.finish_batched_draft(self.begin_batched_draft(seeds))

    def begin_batched_draft(
        self,
        seeds: list[tuple[str, int, int, int, typing.Any]],
        *,
        seed_hidden_matrix: jax.Array | None = None,
    ) -> dict:
        """Dispatch a pooled batched MTP draft WITHOUT blocking on the result.

        Replaces per-request :meth:`draft_next` calls for the greedy inline-MTP
        path: all requests collected for a decode step are drafted together over
        the drafter's pooled cache. The seeds are scattered into full
        request-pool ``(N,)`` arrays (idle rows masked out and left untouched by
        the forward), the per-row EAGLE rollback target is computed exactly as
        :meth:`draft_next` does, and the drafter's ``k`` batched forwards are
        ENQUEUED on device. The caller reads the result back later with
        :meth:`finish_batched_draft`, so the drafter's device time overlaps
        whatever host work runs in between (the emission loop).

        Args:
            seeds: One ``(req_id, row, seed_token, seed_position, seed_hidden)``
                per collected request. ``seed_hidden`` is either a ``(H,)`` /
                ``(1, 1, H)`` device array, or — when ``seed_hidden_matrix`` is
                given — an INT row index into that matrix (all seed rows are then
                fetched with one fused gather instead of per-request slices).
            seed_hidden_matrix: Optional ``[rows, H]`` device matrix the int
                seed-hidden indices refer to (the batched-verify pre-pass's
                gathered hidden rows).

        Returns:
            An opaque handle dict for :meth:`finish_batched_draft`.
        """
        out: dict[str, list[int]] = {str(rid): [] for (rid, *_rest) in seeds}
        # Per-step dynamic budget: 0 means speculation is throttled OFF for
        # this step's concurrency — return the empty handle before any device
        # work so a K=0 step carries no drafter overhead at all.
        k_live = int(self.live_num_draft_tokens)
        if (
            not seeds
            or self.drafter is None
            or k_live <= 0
            or not callable(getattr(self.drafter, "draft_batched", None))
        ):
            return {"out": out, "drafts_dev": None, "active_reqs": []}

        n = max(1, int(getattr(self._runner, "max_num_reqs", 1)))
        persist_on = self._mtp_persist_enabled()
        ensure = getattr(self.drafter, "ensure_batch_size", None)
        if callable(ensure):
            ensure(n)

        seed_tokens = np.zeros((n,), dtype=np.int32)
        seed_positions = np.zeros((n,), dtype=np.int32)
        committed_lens = np.zeros((n,), dtype=np.int32)
        collected_mask = np.zeros((n,), dtype=bool)
        active_rows: list[int] = []
        active_hidden: list[jax.Array] = []
        active_hidden_rows: list[int] = []
        active_reqs: list[tuple[str, int]] = []

        for rid, row, seed_token, seed_position, seed_hidden in seeds:
            rid = str(rid)
            # Remap the runner's (unstable) req_idx to this request's
            # lifetime-stable drafter row (see __init__ note).
            row = self._stable_drafter_row(rid, int(row)) if persist_on else int(row)
            if not (0 <= row < n):
                continue
            backoff = int(self._backoff_by_req.get(rid, 0))
            if backoff > 0:
                # Reject-backoff: skip drafting this request this step (mirrors the
                # early-return branch of ``draft_next``).
                self._backoff_by_req[rid] = backoff - 1
                self._mtp_persist_checkpoint_skipped_window(rid, int(seed_position))
                continue
            if persist_on:
                if self._mtp_persist_req_by_row.get(row) != rid:
                    # Fresh request or a slot reused by a different request: re-base
                    # this row's committed origin and drop its stale K/V (roll to 0).
                    self._mtp_persist_req_by_row[row] = rid
                    self._mtp_persist_base_pos_by_row[row] = int(seed_position)
                    self._mtp_persist_last_seed_by_row[row] = int(seed_position)
                    committed = 0
                else:
                    committed = self._mtp_persist_committed_len(row, int(seed_position))
                # ``draft_batched`` writes ``k_live`` slots for every collected row.
                self._mtp_persist_written_by_row[row] = k_live
            else:
                committed = 0
            seed_tokens[row] = int(seed_token)
            seed_positions[row] = int(seed_position)
            committed_lens[row] = int(committed)
            collected_mask[row] = True
            active_rows.append(row)
            if seed_hidden_matrix is not None and isinstance(seed_hidden, (int, np.integer)):
                active_hidden_rows.append(int(seed_hidden))
            else:
                active_hidden.append(jnp.asarray(seed_hidden).reshape(-1))
            active_reqs.append((rid, row))

        if not active_rows:
            return {"out": out, "drafts_dev": None, "active_reqs": []}

        rows_arr = jnp.asarray(active_rows, dtype=jnp.int32)
        if seed_hidden_matrix is not None and active_hidden_rows and not active_hidden:
            # One fused gather of all seed rows out of the pre-pass hidden
            # matrix, instead of one eager device slice per request.
            matrix = jnp.asarray(seed_hidden_matrix)
            stacked_hidden = matrix[jnp.asarray(active_hidden_rows, dtype=jnp.int32)][:, None, :]
        else:
            if seed_hidden_matrix is not None and active_hidden_rows:
                # Mixed int/array seeds (not produced by the runner today):
                # materialize the int rows so both kinds stack uniformly.
                matrix = jnp.asarray(seed_hidden_matrix)
                active_hidden.extend(matrix[int(i)].reshape(-1) for i in active_hidden_rows)
            stacked_hidden = jnp.stack(active_hidden, axis=0)[:, None, :]
        hidden_dim = int(stacked_hidden.shape[-1])
        hidden_full = jnp.zeros((n, 1, hidden_dim), dtype=stacked_hidden.dtype)
        hidden_full = hidden_full.at[rows_arr].set(stacked_hidden)

        drafts_dev = self.drafter.draft_batched(
            seed_tokens=jnp.asarray(seed_tokens),
            seed_hidden=hidden_full,
            seed_positions=jnp.asarray(seed_positions),
            committed_lens=jnp.asarray(committed_lens),
            collected_mask=jnp.asarray(collected_mask),
            # The drafter compile-keys on k (unrolled recurrence); a schedule
            # contributes one compile per distinct k, warmed by
            # ``eSurgeRunner.compile`` so serving never JITs mid-flight.
            num_draft_tokens=k_live,
        )
        return {"out": out, "drafts_dev": drafts_dev, "active_reqs": active_reqs}

    def finish_batched_draft(self, handle: dict) -> dict[str, list[int]]:
        """Read back a batched draft dispatched by :meth:`begin_batched_draft`.

        Blocks on the device drafts (the only synchronous part of the batched
        draft) and maps each pool row back to its request id.

        Args:
            handle: The dict returned by :meth:`begin_batched_draft`.

        Returns:
            ``{req_id: [drafted token ids]}`` for every input seed.
        """
        out = handle["out"]
        drafts_dev = handle.get("drafts_dev")
        if drafts_dev is None:
            return out
        drafts_host = np.asarray(jax.device_get(drafts_dev))
        total = 0
        for rid, row in handle["active_reqs"]:
            drafted = [int(t) for t in drafts_host[row, :]]
            out[rid] = drafted
            total += len(drafted)
        self.num_drafts_generated += total
        return out

    def can_draft_in_verify(self) -> bool:
        """Whether the fused DRAFT-IN-VERIFY program can serve this drafter.

        The fused path folds the greedy verify argmax, the acceptance compare,
        and the ``k`` recursive MTP draft forwards into ONE compiled program
        (:meth:`~easydel.inference.speculative.Qwen3_5MTPDrafter.verify_draft_batched`),
        removing the separate drafter dispatch and one host round-trip per
        decode step. It subsumes the batched-draft path, so it honors BOTH
        escape hatches: ``EASYDEL_DISABLE_DRAFT_IN_VERIFY`` (fall back to the
        two-dispatch flow) and ``EASYDEL_DISABLE_BATCHED_DRAFT`` (fall all the
        way back to per-request drafting).

        Returns:
            ``True`` when the attached drafter exposes the fused program and no
            escape hatch is set. Per-request/per-step gating (greedy, backoff,
            recurrent replay, dynamic-K transitions) stays with the runner.
        """
        if self.drafter is None or self.num_draft_tokens <= 0:
            return False
        if flags.get_bool(flags.EASYDEL_DISABLE_DRAFT_IN_VERIFY):
            return False
        if flags.get_bool(flags.EASYDEL_DISABLE_BATCHED_DRAFT):
            return False
        if not bool(getattr(self.drafter, "supports_draft_in_verify", False)):
            return False
        if not callable(getattr(self.drafter, "verify_draft_batched", None)):
            return False
        if bool(getattr(self.drafter, "requires_target_kv_cache", False)):
            return False
        # The default fused projection is the raw shared LM head. Use the
        # split executor's authoritative contract, including model-owned hooks,
        # rather than a second list of attributes that can miss transforms.
        # An explicit projection override owns its postprocessing and remains
        # eligible; a hook returning None declares an identity transformation.
        if self.draft_in_verify_project_fn() is None:
            model = getattr(self._runner, "model", None)
            if resolve_lm_logit_postprocess(model) is not None:
                return False
        return True

    def draft_in_verify_project_fn(self) -> typing.Callable | None:
        """Traceable verify-projection override for the fused program.

        ``None`` (the default) lets the drafter project the verify rows through
        the shared LM head inside the fused program — the same
        ``apply_lm_head`` math :meth:`project_hidden_rows` runs on the
        two-dispatch path. Tests override this to inject a deterministic
        batch-invariant projection into BOTH paths. The returned callable must
        be STABLE (its identity keys the drafter's compile cache).

        Returns:
            A traceable ``[R, H] -> [R, V]`` callable, or ``None``.
        """
        return None

    def begin_draft_in_verify(
        self,
        entries: list[tuple[_SpecVerifyMetadata, int, int]],
        *,
        gathered_hidden: jax.Array,
        k_pad: int,
    ) -> dict | None:
        """Dispatch the fused verify+draft program WITHOUT blocking on it.

        Replaces the two-dispatch flow (standalone argmax program -> blocking
        argmax readback -> host acceptance -> :meth:`begin_batched_draft`) for
        the greedy batched path: acceptance moves INTO the compiled program, so
        the ``k`` MTP draft forwards are enqueued in the same dispatch as the
        verify projection and the step's only host sync
        (:meth:`finish_draft_in_verify`) returns argmaxes AND next-window
        drafts together.

        Host persist bookkeeping is split around the dispatch: row claims
        (``_mtp_persist_req_by_row``) happen here because freshness does not
        depend on the acceptance result, while a fresh row's committed-origin
        (``_mtp_persist_base_pos_by_row``) is only known after the program
        computes the accepted count, so it is recorded in
        :meth:`finish_draft_in_verify` from the read-back seed position. A
        reused row's stale base is dropped at claim time so an abort between
        the two phases can never make a later fallback path misread it.

        Args:
            entries: One ``(verify_meta, offset, count)`` per collected greedy
                request, where ``offset``/``count`` locate the request's verify
                rows inside ``gathered_hidden`` (the runner pre-pass layout).
            gathered_hidden: ``[R, H]`` gathered verify-row hidden states.
            k_pad: Verify-window width the gather stripe was padded to. The
                fused program is only engaged at steady state
                (``k_pad == live_num_draft_tokens``); the runner gates
                transitions to the fallback flow.

        Returns:
            An opaque handle for :meth:`finish_draft_in_verify`, or ``None``
            when the fused path cannot serve this window (the caller falls back
            to the two-dispatch flow with no state disturbed).
        """
        k_live = int(self.live_num_draft_tokens)
        if not entries or k_live <= 0 or int(k_pad) != k_live or not self.can_draft_in_verify():
            return None
        n = max(1, int(getattr(self._runner, "max_num_reqs", 1)))
        persist_on = self._mtp_persist_enabled()

        # Validation pass (no state mutation): any ineligible row falls the
        # whole window back to the two-dispatch flow.
        rows_by_entry: list[int] = []
        for meta, _off, _cnt in entries:
            rid = str(meta.request_id)
            if int(self._backoff_by_req.get(rid, 0)) > 0:
                return None
            if len(meta.buffer_draft_tokens) > int(k_pad):
                return None
            row = self._stable_drafter_row(rid, int(meta.req_idx)) if persist_on else int(meta.req_idx)
            if not (0 <= row < n):
                return None
            rows_by_entry.append(row)

        ensure = getattr(self.drafter, "ensure_batch_size", None)
        if callable(ensure):
            ensure(n)

        row_offsets = np.zeros((n,), dtype=np.int32)
        draft_tokens = np.zeros((n, int(k_pad)), dtype=np.int32)
        draft_counts = np.zeros((n,), dtype=np.int32)
        seed_pos_base = np.zeros((n,), dtype=np.int32)
        committed_base = np.zeros((n,), dtype=np.int32)
        fresh_mask = np.zeros((n,), dtype=bool)
        collected_mask = np.zeros((n,), dtype=bool)
        active: list[tuple[str, int, int, int]] = []

        for (meta, off, cnt), row in zip(entries, rows_by_entry, strict=True):
            rid = str(meta.request_id)
            base_known = int(meta.start_pos) + int(meta.real_count) - 1
            if persist_on:
                if self._mtp_persist_req_by_row.get(row) != rid:
                    # Fresh request or a slot reused by a different request:
                    # claim the row, drop any stale base (so a fallback after an
                    # abort computes committed == 0 too), and roll to 0
                    # in-program. The base is re-recorded post-readback.
                    self._mtp_persist_req_by_row[row] = rid
                    self._mtp_persist_base_pos_by_row.pop(row, None)
                    self._mtp_persist_last_seed_by_row.pop(row, None)
                    self._mtp_persist_written_by_row.pop(row, None)
                    fresh_mask[row] = True
                else:
                    base = int(self._mtp_persist_base_pos_by_row.get(row, base_known))
                    committed_base[row] = base_known - base
            else:
                fresh_mask[row] = True
            drafts_row = [int(t) for t in meta.buffer_draft_tokens]
            draft_tokens[row, : len(drafts_row)] = drafts_row
            draft_counts[row] = len(drafts_row)
            row_offsets[row] = int(off)
            seed_pos_base[row] = base_known
            collected_mask[row] = True
            active.append((rid, row, int(off), int(cnt)))

        arg_mat, accepted, seed_positions, drafts = self.drafter.verify_draft_batched(
            gathered_hidden=gathered_hidden,
            row_offsets=row_offsets,
            draft_tokens=draft_tokens,
            draft_counts=draft_counts,
            seed_pos_base=seed_pos_base,
            committed_base=committed_base,
            fresh_mask=fresh_mask,
            collected_mask=collected_mask,
            num_draft_tokens=k_live,
            project_fn=self.draft_in_verify_project_fn(),
        )
        return {
            "arg_mat": arg_mat,
            "accepted": accepted,
            "seed_positions": seed_positions,
            "drafts": drafts,
            "active": active,
            "fresh": fresh_mask,
            "persist_on": persist_on,
            "k_live": k_live,
        }

    def finish_draft_in_verify(self, handle: dict) -> dict[str, dict[str, typing.Any]]:
        """Read back a fused verify+draft dispatched by :meth:`begin_draft_in_verify`.

        The step's single host sync: one ``jax.device_get`` over the argmax
        stripes, accepted counts, seed positions, and next-window drafts. Also
        records a fresh row's persist base (its committed-origin) from the
        read-back seed position — the deferred half of the bookkeeping
        :meth:`begin_draft_in_verify` could not do before acceptance was known.

        Args:
            handle: The dict returned by :meth:`begin_draft_in_verify`.

        Returns:
            ``{req_id: {"argmaxes": [count ints], "accepted": int,
            "known_len": int, "drafts": [k ints]}}`` — ``argmaxes`` matches the
            two-dispatch pre-pass slice for the request (its ``count`` verify
            rows), and ``known_len`` is ``start + real + accepted + 1`` (the
            emitted-through length the runner writes drafts after).
        """
        arg_np, acc_np, pos_np, drafts_np = jax.device_get(
            (handle["arg_mat"], handle["accepted"], handle["seed_positions"], handle["drafts"])
        )
        arg_np = np.asarray(arg_np)
        acc_np = np.asarray(acc_np)
        pos_np = np.asarray(pos_np)
        drafts_np = np.asarray(drafts_np)
        persist_on = bool(handle["persist_on"])
        fresh = handle["fresh"]
        out: dict[str, dict[str, typing.Any]] = {}
        total = 0
        for rid, row, _off, cnt in handle["active"]:
            drafted = [int(t) for t in drafts_np[row, :]]
            if persist_on:
                seed_pos = int(pos_np[row])
                if bool(fresh[row]):
                    self._mtp_persist_base_pos_by_row[row] = seed_pos
                    self._mtp_persist_last_seed_by_row[row] = seed_pos
                else:
                    # The in-program rollback already clamped this window to the
                    # written extent; re-base for any fully-accepted gap so the
                    # NEXT window's ``committed_base`` stays exact.
                    self._mtp_persist_committed_len(row, seed_pos)
                self._mtp_persist_written_by_row[row] = int(handle.get("k_live", len(drafted)))
            out[rid] = {
                "argmaxes": [int(t) for t in arg_np[row, : int(cnt)]],
                "accepted": int(acc_np[row]),
                "known_len": int(pos_np[row]) + 2,
                "drafts": drafted,
            }
            total += len(drafted)
        self.num_drafts_generated += total
        return out

    def record_acceptance(self, req_id: str, accepted: int, num_drafts: int) -> None:
        """Update the adaptive drafter stop-loss after a verify window.

        Hybrid recurrent decode can keep alternate states for speculative
        tokens. Without that cache layout, low-acceptance
        prompts can spend more wallclock on MTP + verification than they save.
        This small gate is intentionally drafter-only: after a full rejection,
        skip a configurable number of future drafter calls and let normal
        eSurge decode advance the sequence.

        Args:
            req_id: Request identifier the verify window belonged to.
            accepted: Number of draft tokens accepted this window.
            num_drafts: Number of draft tokens that were verified.
        """
        if self.reject_backoff_steps <= 0 or num_drafts <= 0:
            return
        key = str(req_id)
        if int(accepted) <= 0:
            self._backoff_by_req[key] = int(self.reject_backoff_steps)
        else:
            self._backoff_by_req.pop(key, None)

    def recurrent_candidate_count(self) -> int:
        """Return per-request speculative recurrent candidate rows, if present."""
        if not getattr(self._runner, "spec_decode_recurrent_candidates", False):
            return 0
        try:
            from easydel.caching import HybridCache, KDACacheView, ParallelHybridCacheView, RecurrentCacheView
        except Exception:
            return 0
        cache = self._runner.executor_manager.kv_pages
        if not isinstance(cache, HybridCache):
            return 0
        base_slots = max(1, int(self._runner.max_num_reqs))
        for view in cache.views:
            if isinstance(view, KDACacheView):
                state = view.recurrent_state
                n_slots = int(getattr(state, "shape", (0,))[0]) if state is not None else 0
                if n_slots > base_slots:
                    return max(0, (n_slots - base_slots) // base_slots)
                continue
            rec = None
            if isinstance(view, RecurrentCacheView):
                rec = view
            elif isinstance(view, ParallelHybridCacheView) and view.recurrent is not None:
                rec = view.recurrent
            if rec is None:
                continue
            state = rec.conv_state if rec.conv_state is not None else rec.recurrent_state
            if state is None:
                continue
            n_slots = int(getattr(state, "shape", (0,))[0])
            if n_slots > base_slots:
                return max(0, (n_slots - base_slots) // base_slots)
        return 0

    def cache_has_recurrent_state(self) -> bool:
        """Whether the current eSurge cache contains recurrent/SSM rows (generic recurrent or KDA views)."""
        try:
            from easydel.caching import HybridCache, KDACacheView, ParallelHybridCacheView, RecurrentCacheView
        except Exception:
            return False
        cache = self._runner.executor_manager.kv_pages
        if not isinstance(cache, HybridCache):
            return False
        for view in cache.views:
            if isinstance(view, KDACacheView):
                # KDA layers keep a rolling recurrent row and short-conv windows; with speculative candidate rows the
                # accepted prefix is committed by a row copy, otherwise by snapshot + replay.
                return True
            if isinstance(view, RecurrentCacheView):
                return view.conv_state is not None or view.recurrent_state is not None
            if isinstance(view, ParallelHybridCacheView) and view.recurrent is not None:
                rec = view.recurrent
                return rec.conv_state is not None or rec.recurrent_state is not None
        return False

    def cache_replay_required(self) -> bool:
        """Pure paged-attention caches can ignore stale rejected draft KV rows.

        A future step overwrites rejected draft positions before those positions
        are admitted by ``context_lens``. Recurrent layers are different: their
        state is a single rolling row, so they need either candidate
        rows or an explicit replay from the pre-verify snapshot.
        """
        if self.rolls_back_cache:
            return True
        if not self.cache_has_recurrent_state():
            return False
        return self.recurrent_candidate_count() <= 0

    def commit_recurrent_candidate_state(self, row_pos: int, prefix_len: int) -> bool:
        """Copy a speculative recurrent candidate row into the live recurrent row.

        Hybrid recurrent caches keep extra candidate rows holding the per-step
        prefix states produced during a fused speculative window. After
        verification settles on an accepted ``prefix_len``, this copies the
        matching candidate row (conv and recurrent state) into the request's
        live row via a cached JIT so the next decode step continues from the
        same state as non-speculative generation.

        Args:
            row_pos: Live recurrent row index of the request.
            prefix_len: Number of accepted prefix tokens (1-based selector into
                the candidate rows); must be in ``[1, candidate_count]``.

        Returns:
            ``True`` if the commit was performed; ``False`` when there are no
            candidate rows, ``prefix_len`` is out of range, the cache is not a
            ``HybridCache``, or ``row_pos`` is out of bounds.
        """
        candidate_count = self.recurrent_candidate_count()
        if candidate_count <= 0 or prefix_len <= 0 or prefix_len > candidate_count:
            return False
        try:
            from easydel.caching import HybridCache, KDACacheView, ParallelHybridCacheView, RecurrentCacheView
        except Exception:
            return False
        cache = self._runner.executor_manager.kv_pages
        if not isinstance(cache, HybridCache):
            return False

        base_slots = max(1, int(self._runner.max_num_reqs))
        row_pos = int(row_pos)
        if row_pos < 0 or row_pos >= base_slots:
            return False

        commit_fn = self._recurrent_commit_fn
        if commit_fn is None or int(self._recurrent_commit_candidate_count) != int(candidate_count):

            def _copy_candidate_row(arr: jax.Array | None, row: jax.Array, prefix: jax.Array) -> jax.Array | None:
                """Copy one speculative candidate row into its live destination row."""
                if arr is None:
                    return None
                n_slots = int(arr.shape[0])
                # Prefix-major candidate layout: base + prefix * base_slots + slot.
                candidate_row = base_slots + (prefix - 1) * base_slots + row
                safe_dst = jnp.clip(row, 0, n_slots - 1)
                safe_src = jnp.clip(candidate_row, 0, n_slots - 1)
                valid = (row >= 0) & (row < n_slots) & (prefix > 0) & (prefix <= int(candidate_count))
                valid = valid & (candidate_row >= 0) & (candidate_row < n_slots)
                return jax.lax.cond(
                    valid,
                    lambda x: x.at[safe_dst].set(x[safe_src]),
                    lambda x: x,
                    arr,
                )

            @jax.jit
            def _commit(cache_in: HybridCache, row: jax.Array, prefix: jax.Array) -> HybridCache:
                """JIT-compiled commit copying candidate states across cache views."""
                new_views = list(cache_in.views)
                for idx, view in enumerate(new_views):
                    if isinstance(view, KDACacheView):
                        new_views[idx] = view.update_kda_states(
                            new_q_conv_state=_copy_candidate_row(view.q_conv_state, row, prefix),
                            new_k_conv_state=_copy_candidate_row(view.k_conv_state, row, prefix),
                            new_v_conv_state=_copy_candidate_row(view.v_conv_state, row, prefix),
                            new_recurrent_state=_copy_candidate_row(view.recurrent_state, row, prefix),
                        )
                        continue
                    rec = None
                    parallel_view = None
                    if isinstance(view, RecurrentCacheView):
                        rec = view
                    elif isinstance(view, ParallelHybridCacheView) and view.recurrent is not None:
                        rec = view.recurrent
                        parallel_view = view
                    if rec is None:
                        continue
                    new_conv = _copy_candidate_row(rec.conv_state, row, prefix)
                    new_rec = _copy_candidate_row(rec.recurrent_state, row, prefix)
                    updated_rec = rec.replace(conv_state=new_conv, recurrent_state=new_rec)
                    if parallel_view is not None:
                        new_views[idx] = parallel_view.replace(
                            recurrent=updated_rec,
                            conv_state=new_conv,
                            recurrent_state=new_rec,
                        )
                    else:
                        new_views[idx] = updated_rec
                return HybridCache(views=new_views)

            commit_fn = _commit
            self._recurrent_commit_fn = commit_fn
            self._recurrent_commit_candidate_count = int(candidate_count)

        row_arr = jnp.asarray(row_pos, dtype=jnp.int32)
        prefix_arr = jnp.asarray(int(prefix_len), dtype=jnp.int32)
        self._runner.executor_manager.kv_pages = commit_fn(cache, row_arr, prefix_arr)
        return True

    def queue_or_commit_recurrent_candidate_state(
        self,
        *,
        req_id: str,
        row_pos: int,
        prefix_len: int,
        defer: bool,
    ) -> bool:
        """Make the live recurrent row match the accepted speculative prefix.

        Qwen3.5/Next recurrent layers may execute a verify window through a
        fast multi-token ragged GDN path, while normal greedy decode advances
        the state one token at a time. The speculative candidate rows are the
        single-step prefix states; commit one after every verify window, not
        only on rejection, so the next decode step starts from the same state
        as non-speculative generation.

        Args:
            req_id: Request identifier the commit applies to.
            row_pos: Live recurrent row index of the request.
            prefix_len: Number of accepted prefix tokens (1-based candidate-row
                selector); must be in ``[1, candidate_count]``.
            defer: When True, queue the commit (applied inside the next compiled
                step via ``spec_recurrent_commit_cpu``); when False, commit
                immediately on the host.

        Returns:
            ``True`` if the commit was queued or performed; ``False`` when there
            are no candidate rows or ``prefix_len`` is out of range.
        """
        candidate_count = self.recurrent_candidate_count()
        prefix_len = int(prefix_len)
        if candidate_count <= 0 or prefix_len <= 0 or prefix_len > candidate_count:
            return False
        if defer:
            self._runner.queue_speculative_recurrent_commit(str(req_id), prefix_len)
            return True
        return self.commit_recurrent_candidate_state(
            row_pos=int(row_pos),
            prefix_len=prefix_len,
        )

    def _commit_compressed_window_without_replay(
        self,
        *,
        pre_step_kv_pages: typing.Any,
        commit_scheduled_list: list[int],
        recurrent_slot_indices_cpu: np.ndarray | None,
    ) -> bool:
        """Replay-free rollback of a traced compressed-window cache (DeepSeek-V4 / V4.1 DSpark).

        Rebuilds only the per-stream state (rejected ring slots, ``cache_position``, open-group compressor state)
        from the pre-verify snapshot and the verify-time projection traces, with no model forward. Returns ``False``
        (caller replays) unless the cache is traced and every window row fits the trace and the ring.
        """
        from easydel.caching.compressed_window import commit_speculative_window, speculative_commit_limit

        if not isinstance(pre_step_kv_pages, _CompressedWindowSnapshot) or recurrent_slot_indices_cpu is None:
            return False
        live = self._runner.executor_manager.kv_pages
        limit = speculative_commit_limit(live)
        if limit <= 0:
            return False
        views = [v for v in live.views if v is not None]
        batch = int(views[0].cache_position.shape[0])
        committed = np.full((batch,), np.iinfo(np.int32).max // 2, dtype=np.int32)
        for row in range(len(commit_scheduled_list)):
            slot = int(recurrent_slot_indices_cpu[row])
            if 0 <= slot < batch:
                committed[slot] = int(commit_scheduled_list[row])
        pre = _restore_compressed_window(live, pre_step_kv_pages, copy=False)
        # Verify widths straight from the cache (every view, one fused reduction, one scalar readback): a rolled-back
        # row is exact only when its window fits the ring and the projection trace; otherwise the caller replays.
        live_positions = [lv.cache_position for lv in live.views if lv is not None]
        pre_positions = [pv.cache_position for pv in pre.views if pv is not None]
        committed_device = jnp.asarray(committed)
        if bool(_window_exceeds_limit(live_positions, pre_positions, committed_device, int(limit))):
            return False
        self._runner.executor_manager.kv_pages = commit_speculative_window(live, pre, committed_device)
        self.num_replay_free_commits += 1
        return True

    def commit_cache_replay(
        self,
        *,
        pre_step_kv_pages: typing.Any,
        commit_scheduled_list: list[int],
        window_row_indices: np.ndarray,
        recurrent_slot_indices_cpu: np.ndarray | None,
        num_tokens_static_original: int,
        padded_num_reqs: int,
        token_ids_window_cpu: np.ndarray,
        num_computed_tokens_window_cpu: np.ndarray,
        temperature_window_cpu: np.ndarray,
        top_p_window_cpu: np.ndarray,
        top_k_window_cpu: np.ndarray,
        min_p_window_cpu: np.ndarray,
        frequency_penalties_window_cpu: np.ndarray,
        presence_penalties_window_cpu: np.ndarray,
        repetition_penalties_window_cpu: np.ndarray,
        page_table_window_cpu: np.ndarray,
        page_table_window_version: int | None,
        mrope_position_ids_cpu: np.ndarray | None,
        prefill_embeds_cpu: np.ndarray | None,
        prefill_embeds_mask_cpu: np.ndarray | None,
        visual_pos_masks_cpu: np.ndarray | None,
        deepstack_visual_embeds_cpu: list[np.ndarray] | None,
    ) -> None:
        """Replay the executor step over the committed speculative prefix.

        Restores the pre-step KV pages snapshot and re-runs the model step with
        ``commit_scheduled_list`` token counts per row, which results in the
        cache being advanced exactly through the accepted speculative prefix.
        Used by the recurrent speculative path when the cache does not have
        candidate rows and must be replayed.

        Args:
            pre_step_kv_pages: KV pages snapshot taken before the speculative
                forward pass was launched.
            commit_scheduled_list: Per-row token counts to feed back into the
                replay step (length matches the active window).
            window_row_indices: Window-local row indices for the active rows.
            num_tokens_static_original: Token bucket size used on the original
                forward pass; bounds the replay bucket.
            padded_num_reqs: Request bucket size used on the original step.
            token_ids_window_cpu, num_computed_tokens_window_cpu: Token and
                compute-progress CPU buffers shaped for the window.
            temperature_window_cpu, top_p_window_cpu, top_k_window_cpu,
            min_p_window_cpu, frequency_penalties_window_cpu,
            presence_penalties_window_cpu, repetition_penalties_window_cpu:
                Per-row sampling parameter CPU buffers.
            page_table_window_cpu: KV-cache page-table block ids for the
                active window.
            page_table_window_version: Optional page-table cache version, or
                ``None`` for packed/sparse layouts that cannot be cached.
            mrope_position_ids_cpu, prefill_embeds_cpu, prefill_embeds_mask_cpu,
            visual_pos_masks_cpu, deepstack_visual_embeds_cpu: VLM helper
                buffers; passed through to the replay step.
        """
        if self._commit_compressed_window_without_replay(
            pre_step_kv_pages=pre_step_kv_pages,
            commit_scheduled_list=commit_scheduled_list,
            recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
        ):
            return
        total_commit = int(sum(commit_scheduled_list))
        self._runner.executor_manager.kv_pages = _restore_recurrent_state(
            self._runner.executor_manager.kv_pages, pre_step_kv_pages
        )
        if total_commit <= 0:
            return
        if self.rolls_back_cache:
            # Per-slot compressed-window caches advance a slot's columns in blocks sized from the bucket (one slot's
            # share, ``ceil(bucket / slots)``). The verify bucket's share covers each row's 1 + k verify columns, so
            # it also covers every row's committed prefix and the replay runs one narrow block step. Re-bucketing to
            # the smallest fitting bucket shrinks the share below the longest committed row and forces the wide
            # bucket-wide block instead (measured on DeepSeek-V4-Flash cc=8: replay 205 ms in bucket 32 vs verify 117).
            num_tokens_static = int(num_tokens_static_original)
        else:
            idx = bisect_left(self._runner.num_tokens_paddings, total_commit)
            if idx >= len(self._runner.num_tokens_paddings):
                idx = len(self._runner.num_tokens_paddings) - 1
            num_tokens_static = int(self._runner.num_tokens_paddings[idx])
            if num_tokens_static > int(num_tokens_static_original):
                num_tokens_static = int(num_tokens_static_original)

        commit_scheduled_full_cpu = self._runner.scheduled_token_counts_cpu.copy()
        commit_scheduled_full_cpu.fill(0)
        commit_scheduled_full_cpu[: len(commit_scheduled_list)] = np.asarray(commit_scheduled_list, dtype=np.int32)

        commit_req_num_tokens = np.asarray(num_computed_tokens_window_cpu).copy()
        for i, n in enumerate(commit_scheduled_list):
            commit_req_num_tokens[i] = int(num_computed_tokens_window_cpu[i]) + int(n) + 1

        self._runner.executor_manager.execute(
            num_tokens=num_tokens_static,
            scheduled_full_cpu=commit_scheduled_full_cpu,
            cache_group_inputs=self._runner.cache_group_inputs_for_execute(
                window_row_indices_cpu=window_row_indices,
                num_computed_tokens_cpu=num_computed_tokens_window_cpu,
            ),
            model_side_inputs_cpu=self._runner.side_inputs_for_execute(
                token_ids_cpu=token_ids_window_cpu,
                num_computed_tokens_cpu=num_computed_tokens_window_cpu,
                scheduled_full_cpu=commit_scheduled_full_cpu,
                window_row_indices_cpu=window_row_indices,
                num_tokens_static=num_tokens_static,
            ),
            req_num_tokens_full_cpu=commit_req_num_tokens,
            active_mask_full_cpu=self._runner.active_request_mask_cpu,
            window_row_indices_cpu=window_row_indices,
            recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
            input_ids_buf=self._runner.input_ids_buf,
            position_ids_buf=self._runner.position_ids_buf,
            padded_num_reqs=padded_num_reqs,
            token_ids_cpu=token_ids_window_cpu,
            num_computed_tokens_cpu=num_computed_tokens_window_cpu,
            temperature_cpu=temperature_window_cpu,
            top_p_cpu=top_p_window_cpu,
            top_k_cpu=top_k_window_cpu,
            min_p_cpu=min_p_window_cpu,
            frequency_penalties_cpu=frequency_penalties_window_cpu,
            presence_penalties_cpu=presence_penalties_window_cpu,
            repetition_penalties_cpu=repetition_penalties_window_cpu,
            page_table_cpu=page_table_window_cpu,
            page_table_version=page_table_window_version,
            mrope_position_ids_cpu=mrope_position_ids_cpu,
            prefill_embeds_cpu=prefill_embeds_cpu,
            prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
            visual_pos_masks_cpu=visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
            wait_for_outputs=True,
        )

    def run_suffix_sample(
        self,
        row_pos: int,
        start_len: int,
        suffix_len: int = 1,
        *,
        num_computed_tokens_window_cpu,
        window_row_indices_cpu,
        recurrent_slot_indices_cpu,
        padded_num_reqs,
        token_ids_window_cpu,
        temperature_window_cpu,
        top_p_window_cpu,
        top_k_window_cpu,
        min_p_window_cpu,
        frequency_penalties_window_cpu,
        presence_penalties_window_cpu,
        repetition_penalties_window_cpu,
        page_table_window_cpu,
        page_table_window_version,
        mrope_position_ids_cpu,
        prefill_embeds_cpu,
        prefill_embeds_mask_cpu,
        visual_pos_masks_cpu,
        deepstack_visual_embeds_cpu,
    ) -> tuple[int, jax.Array]:
        """Execute a one-row suffix forward pass and return the sampled token + hidden row.

        Used by speculative decoding after a rejected prefix to re-sample
        the corrective token at the rejection boundary without touching
        any other window row.
        """
        suffix_timer_start = time.time()
        suffix_scheduled = self._runner.scheduled_token_counts_cpu.copy()
        suffix_scheduled.fill(0)
        suffix_scheduled[int(row_pos)] = int(suffix_len)

        suffix_active = self._runner.active_request_mask_cpu.copy()
        suffix_active.fill(False)
        suffix_active[int(row_pos)] = True

        suffix_num_computed = np.asarray(num_computed_tokens_window_cpu).copy()
        suffix_num_computed[int(row_pos)] = int(num_computed_tokens_window_cpu[int(row_pos)]) + int(start_len)

        suffix_req_num_tokens = np.asarray(num_computed_tokens_window_cpu).copy()
        suffix_req_num_tokens[int(row_pos)] = (
            int(num_computed_tokens_window_cpu[int(row_pos)]) + int(start_len) + int(suffix_len)
        )

        total_suffix = int(suffix_len)
        suffix_idx = bisect_left(self._runner.num_tokens_paddings, total_suffix)
        if suffix_idx >= len(self._runner.num_tokens_paddings):
            suffix_idx = len(self._runner.num_tokens_paddings) - 1
        suffix_num_tokens_static = int(self._runner.num_tokens_paddings[suffix_idx])

        out_tokens_suffix, _, _, _, hidden_suffix, _, _ = self._runner.executor_manager.execute(
            num_tokens=suffix_num_tokens_static,
            scheduled_full_cpu=suffix_scheduled,
            cache_group_inputs=self._runner.cache_group_inputs_for_execute(
                window_row_indices_cpu=window_row_indices_cpu,
                num_computed_tokens_cpu=suffix_num_computed,
            ),
            model_side_inputs_cpu=self._runner.side_inputs_for_execute(
                token_ids_cpu=token_ids_window_cpu,
                num_computed_tokens_cpu=suffix_num_computed,
                scheduled_full_cpu=suffix_scheduled,
                window_row_indices_cpu=window_row_indices_cpu,
                num_tokens_static=suffix_num_tokens_static,
            ),
            req_num_tokens_full_cpu=suffix_req_num_tokens,
            active_mask_full_cpu=suffix_active,
            window_row_indices_cpu=window_row_indices_cpu,
            recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
            input_ids_buf=self._runner.input_ids_buf,
            position_ids_buf=self._runner.position_ids_buf,
            padded_num_reqs=padded_num_reqs,
            token_ids_cpu=token_ids_window_cpu,
            num_computed_tokens_cpu=suffix_num_computed,
            temperature_cpu=temperature_window_cpu,
            top_p_cpu=top_p_window_cpu,
            top_k_cpu=top_k_window_cpu,
            min_p_cpu=min_p_window_cpu,
            frequency_penalties_cpu=frequency_penalties_window_cpu,
            presence_penalties_cpu=presence_penalties_window_cpu,
            repetition_penalties_cpu=repetition_penalties_window_cpu,
            page_table_cpu=page_table_window_cpu,
            page_table_version=page_table_window_version,
            mrope_position_ids_cpu=mrope_position_ids_cpu,
            prefill_embeds_cpu=prefill_embeds_cpu,
            prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
            visual_pos_masks_cpu=visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
            wait_for_outputs=True,
        )
        suffix_token = int(np.asarray(out_tokens_suffix)[int(row_pos)])
        suffix_hidden = window_hidden_row(
            hidden_suffix,
            row_pos=int(row_pos),
            token_offset=0,
            token_index=max(0, int(suffix_len) - 1),
            total_window_tokens=int(suffix_len),
            padded_reqs=int(padded_num_reqs),
            output_layout=self._runner.executor_manager.hidden_state_layout(
                hidden_suffix, num_tokens=suffix_num_tokens_static, padded_num_reqs=int(padded_num_reqs)
            ),
        )
        self._runner.record_speculative_execution_time(suffix=time.time() - suffix_timer_start)
        return suffix_token, suffix_hidden

    def replay_prefix_sample(
        self,
        row_pos: int,
        prefix_len: int,
        *,
        pre_step_kv_pages,
        num_computed_tokens_window_cpu,
        num_tokens_static,
        window_row_indices_cpu,
        recurrent_slot_indices_cpu,
        padded_num_reqs,
        token_ids_window_cpu,
        temperature_window_cpu,
        top_p_window_cpu,
        top_k_window_cpu,
        min_p_window_cpu,
        frequency_penalties_window_cpu,
        presence_penalties_window_cpu,
        repetition_penalties_window_cpu,
        page_table_window_cpu,
        page_table_window_version,
        mrope_position_ids_cpu,
        prefill_embeds_cpu,
        prefill_embeds_mask_cpu,
        visual_pos_masks_cpu,
        deepstack_visual_embeds_cpu,
        window_token_offsets,
        total_scheduled,
        rng_after_verify,
    ) -> tuple[int, jax.Array]:
        """Replay the accepted speculative prefix in one fused forward pass.

        Restores the pre-spec KV snapshot, then re-runs the row with
        ``prefix_len`` scheduled tokens so the cache is advanced exactly
        through the accepted positions before the next decode step.
        """
        replay_timer_start = time.time()
        if pre_step_kv_pages is None:
            raise RuntimeError("Speculative replay requires a pre-step KV snapshot.")
        replay_scheduled = self._runner.scheduled_token_counts_cpu.copy()
        replay_scheduled.fill(0)
        replay_scheduled[int(row_pos)] = int(prefix_len)

        replay_active = self._runner.active_request_mask_cpu.copy()
        replay_active.fill(False)
        replay_active[int(row_pos)] = True

        replay_req_num_tokens = np.asarray(num_computed_tokens_window_cpu).copy()
        replay_req_num_tokens[int(row_pos)] = int(num_computed_tokens_window_cpu[int(row_pos)]) + int(prefix_len)

        self._runner.executor_manager.kv_pages = _restore_recurrent_state(
            self._runner.executor_manager.kv_pages, pre_step_kv_pages
        )
        out_tokens_replay, _, _, _, hidden_replay, _, _ = self._runner.executor_manager.execute(
            num_tokens=num_tokens_static,
            scheduled_full_cpu=replay_scheduled,
            cache_group_inputs=self._runner.cache_group_inputs_for_execute(
                window_row_indices_cpu=window_row_indices_cpu,
                num_computed_tokens_cpu=num_computed_tokens_window_cpu,
            ),
            model_side_inputs_cpu=self._runner.side_inputs_for_execute(
                token_ids_cpu=token_ids_window_cpu,
                num_computed_tokens_cpu=num_computed_tokens_window_cpu,
                scheduled_full_cpu=replay_scheduled,
                window_row_indices_cpu=window_row_indices_cpu,
                num_tokens_static=num_tokens_static,
            ),
            req_num_tokens_full_cpu=replay_req_num_tokens,
            active_mask_full_cpu=replay_active,
            window_row_indices_cpu=window_row_indices_cpu,
            recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
            input_ids_buf=self._runner.input_ids_buf,
            position_ids_buf=self._runner.position_ids_buf,
            padded_num_reqs=padded_num_reqs,
            token_ids_cpu=token_ids_window_cpu,
            num_computed_tokens_cpu=num_computed_tokens_window_cpu,
            temperature_cpu=temperature_window_cpu,
            top_p_cpu=top_p_window_cpu,
            top_k_cpu=top_k_window_cpu,
            min_p_cpu=min_p_window_cpu,
            frequency_penalties_cpu=frequency_penalties_window_cpu,
            presence_penalties_cpu=presence_penalties_window_cpu,
            repetition_penalties_cpu=repetition_penalties_window_cpu,
            page_table_cpu=page_table_window_cpu,
            page_table_version=page_table_window_version,
            mrope_position_ids_cpu=mrope_position_ids_cpu,
            prefill_embeds_cpu=prefill_embeds_cpu,
            prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
            visual_pos_masks_cpu=visual_pos_masks_cpu,
            deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
            wait_for_outputs=True,
        )
        replay_token = int(np.asarray(out_tokens_replay)[int(row_pos)])
        # Only this row is active in the replay. Its prefix is packed from
        # token zero, not at its offset in the original multi-row verify window.
        replay_hidden = window_hidden_row(
            hidden_replay,
            row_pos=int(row_pos),
            token_offset=0,
            token_index=int(prefix_len) - 1,
            total_window_tokens=int(prefix_len),
            padded_reqs=int(padded_num_reqs),
            output_layout=self._runner.executor_manager.hidden_state_layout(
                hidden_replay, num_tokens=int(num_tokens_static), padded_num_reqs=int(padded_num_reqs)
            ),
        )
        if rng_after_verify is not None:
            self._runner.executor_manager.rng_key = rng_after_verify
        self._runner.record_speculative_execution_time(replay=time.time() - replay_timer_start)
        return replay_token, replay_hidden

    def replay_prefix_sequential(
        self,
        row_pos: int,
        prefix_len: int,
        *,
        pre_step_kv_pages,
        num_computed_tokens_window_cpu,
        window_row_indices_cpu,
        recurrent_slot_indices_cpu,
        padded_num_reqs,
        token_ids_window_cpu,
        temperature_window_cpu,
        top_p_window_cpu,
        top_k_window_cpu,
        min_p_window_cpu,
        frequency_penalties_window_cpu,
        presence_penalties_window_cpu,
        repetition_penalties_window_cpu,
        page_table_window_cpu,
        page_table_window_version,
        mrope_position_ids_cpu,
        prefill_embeds_cpu,
        prefill_embeds_mask_cpu,
        visual_pos_masks_cpu,
        deepstack_visual_embeds_cpu,
        rng_after_verify,
    ) -> tuple[int, jax.Array]:
        """Replay an accepted speculative prefix one token at a time.

        Used when the model has recurrent layers that cannot be replayed
        in a single fused window (no candidate-row cache available). Runs
        ``prefix_len`` single-token decode steps on top of the pre-spec
        KV snapshot to advance recurrent state exactly to the boundary.
        """
        replay_timer_start = time.time()
        if pre_step_kv_pages is None:
            raise RuntimeError("Speculative sequential replay requires a pre-step KV snapshot.")
        prefix_len = int(prefix_len)
        if prefix_len <= 0:
            raise ValueError("prefix_len must be positive for speculative sequential replay.")

        self._runner.executor_manager.kv_pages = _restore_recurrent_state(
            self._runner.executor_manager.kv_pages, pre_step_kv_pages
        )
        replay_scheduled = self._runner.scheduled_token_counts_cpu.copy()
        replay_active = self._runner.active_request_mask_cpu.copy()
        replay_num_computed = np.asarray(num_computed_tokens_window_cpu).copy()
        replay_req_num_tokens = np.asarray(num_computed_tokens_window_cpu).copy()
        one_token_idx = bisect_left(self._runner.num_tokens_paddings, 1)
        if one_token_idx >= len(self._runner.num_tokens_paddings):
            one_token_idx = len(self._runner.num_tokens_paddings) - 1
        one_token_static = int(self._runner.num_tokens_paddings[one_token_idx])

        replay_token = 0
        replay_hidden = None
        for step_idx in range(prefix_len):
            replay_scheduled.fill(0)
            replay_scheduled[int(row_pos)] = 1
            replay_active.fill(False)
            replay_active[int(row_pos)] = True
            replay_num_computed[int(row_pos)] = int(num_computed_tokens_window_cpu[int(row_pos)]) + int(step_idx)
            replay_req_num_tokens[int(row_pos)] = int(num_computed_tokens_window_cpu[int(row_pos)]) + int(step_idx) + 1

            out_tokens_replay, _, _, _, hidden_replay, _, _ = self._runner.executor_manager.execute(
                num_tokens=one_token_static,
                scheduled_full_cpu=replay_scheduled,
                cache_group_inputs=self._runner.cache_group_inputs_for_execute(
                    window_row_indices_cpu=window_row_indices_cpu,
                    num_computed_tokens_cpu=replay_num_computed,
                ),
                model_side_inputs_cpu=self._runner.side_inputs_for_execute(
                    token_ids_cpu=token_ids_window_cpu,
                    num_computed_tokens_cpu=replay_num_computed,
                    scheduled_full_cpu=replay_scheduled,
                    window_row_indices_cpu=window_row_indices_cpu,
                    num_tokens_static=one_token_static,
                ),
                req_num_tokens_full_cpu=replay_req_num_tokens,
                active_mask_full_cpu=replay_active,
                window_row_indices_cpu=window_row_indices_cpu,
                recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
                input_ids_buf=self._runner.input_ids_buf,
                position_ids_buf=self._runner.position_ids_buf,
                padded_num_reqs=padded_num_reqs,
                token_ids_cpu=token_ids_window_cpu,
                num_computed_tokens_cpu=replay_num_computed,
                temperature_cpu=temperature_window_cpu,
                top_p_cpu=top_p_window_cpu,
                top_k_cpu=top_k_window_cpu,
                min_p_cpu=min_p_window_cpu,
                frequency_penalties_cpu=frequency_penalties_window_cpu,
                presence_penalties_cpu=presence_penalties_window_cpu,
                repetition_penalties_cpu=repetition_penalties_window_cpu,
                page_table_cpu=page_table_window_cpu,
                page_table_version=page_table_window_version,
                mrope_position_ids_cpu=mrope_position_ids_cpu,
                prefill_embeds_cpu=prefill_embeds_cpu,
                prefill_embeds_mask_cpu=prefill_embeds_mask_cpu,
                visual_pos_masks_cpu=visual_pos_masks_cpu,
                deepstack_visual_embeds_cpu=deepstack_visual_embeds_cpu,
                wait_for_outputs=True,
            )
            replay_token = int(np.asarray(out_tokens_replay)[int(row_pos)])
            replay_hidden = window_hidden_row(
                hidden_replay,
                row_pos=int(row_pos),
                token_offset=0,
                token_index=0,
                total_window_tokens=1,
                padded_reqs=int(padded_num_reqs),
                output_layout=self._runner.executor_manager.hidden_state_layout(
                    hidden_replay, num_tokens=one_token_static, padded_num_reqs=int(padded_num_reqs)
                ),
            )

        if rng_after_verify is not None:
            self._runner.executor_manager.rng_key = rng_after_verify
        self._runner.record_speculative_execution_time(replay=time.time() - replay_timer_start)
        if replay_hidden is None:
            raise RuntimeError("Speculative sequential replay did not produce hidden state.")
        return replay_token, replay_hidden
