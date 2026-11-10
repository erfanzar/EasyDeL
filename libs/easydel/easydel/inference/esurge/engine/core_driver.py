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

"""The engine's scheduler: surgecore's Rust ``EngineCore`` behind the step interface the runner reads.

:class:`EngineCoreDriver` owns one ``surgecore.EngineCore`` (request table, per-rank queues, budgets, KV pages, prefix
cache, recurrent snapshots, speculative drafts) and the Python side of it:

* a handle registry: request id <-> core handle, and each request's :class:`AdmissionRecord`, kept until the core has
  reported the request finished;
* a plan adapter: :meth:`schedule` turns the core's ``Plan`` into the :class:`SchedulerOutput` the runner and the
  multi-host followers consume, and :meth:`commit` turns the runner's :class:`ModelRunnerOutput` into the core's
  plan-order commit arrays and the core's ``Events`` into the :class:`EngineCoreOutputs` the output pipeline reads.

The adapter is temporary: once the runner reads the core's rows directly it goes away.
"""

from __future__ import annotations

import itertools
import time
import typing
from collections.abc import Iterable

import numpy as np

from ..benchmarking import zero_schedule_needs_update
from ..core.interface import ChunkedLocalAttentionSpec, FullAttentionSpec, SlidingWindowSpec
from ..core.manager import page_hash_extra_key
from ..engine_types import EngineCoreOutput, EngineCoreOutputs, FinishReason
from ..logger import logger
from ..metrics import get_metrics_collector
from ..outputs import ModelRunnerOutput
from ..request import AdmissionRecord
from ..scheduler.output import CachedRequestData, NewRequestData, SchedulerOutput
from .admission import expand_request_images

if typing.TYPE_CHECKING:
    from ..core.interface import CacheGroupSpec
    from ..runners.model_runner import eSurgeRunner

FINISH_STOP = 1
"""``finish`` reason of a parser-matched stop string."""
FINISH_LENGTH = 2
"""``finish`` reason of a length limit."""
FINISH_ABORT = 3
"""``finish`` reason of an abort."""

_FINISH_REASON = (None, FinishReason.STOP, FinishReason.LENGTH, FinishReason.ABORT)
_GROUP_KIND = {FullAttentionSpec: 0, SlidingWindowSpec: 1, ChunkedLocalAttentionSpec: 2}
_FLAG_GRAMMAR, _FLAG_NO_PREFIX_HIT, _FLAG_VISION = 1, 2, 4
_PLAN_ASYNC_SAFE, _PLAN_STRUCTURED_PENDING = 1, 16
_KIND_NEW, _KIND_RESUMED = 1, 2
_U64 = (1 << 64) - 1
_RECURRENT_LAYER_TYPES = frozenset({"linear_attention", "kda_linear_attention", "parallel_hybrid", "hybrid"})


def runner_has_recurrent_state(runner: typing.Any) -> bool:
    """Whether the runner's model keeps per-request recurrent (linear-attention/SSM) state."""
    flag = getattr(runner, "_has_recurrent_layers", None)
    if flag is not None:
        return bool(flag)
    config = getattr(getattr(runner, "model", None), "config", None)
    if config is None:
        return False
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    layer_types = getattr(text_config, "layer_types", None) or ()
    return any(str(layer_type).lower() in _RECURRENT_LAYER_TYPES for layer_type in layer_types)


def cache_group_rows(kv_cache_groups: list[CacheGroupSpec]) -> np.ndarray:
    """The core's ``int32[G, 4]`` cache-group descriptor: ``(kind, page_size, window_or_chunk, type_index)``."""
    types: dict[str, int] = {}
    rows = []
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        kind = _GROUP_KIND[type(spec)]
        window = (0, getattr(spec, "sliding_window", 0), getattr(spec, "attention_chunk_size", 0))[kind]
        rows.append((kind, int(spec.page_size), int(window), types.setdefault(spec.type_id, len(types))))
    return np.asarray(rows, np.int32).reshape(-1, 4)


class _Step:
    """A schedule the loop will commit: the core's plan, its runner view, and the entries' handles and ids."""

    __slots__ = ("handles", "plan", "rids", "so")

    def __init__(self, so: SchedulerOutput, plan: typing.Any, handles: list[int], rids: list[str]) -> None:
        self.so = so
        self.plan = plan
        self.handles = handles
        self.rids = rids


class EngineCoreDriver:
    """eSurge's scheduler: a surgecore ``EngineCore`` with its handle registry and the runner's plan adapter.

    Args:
        kv_cache_groups: Cache group specs (at least one).
        max_num_seqs: Running rows over all data-parallel ranks.
        max_num_batched_tokens: Per-step token budget of each rank; ``None`` is ``max_model_len``.
        max_model_len: Longest sequence (prompt plus output).
        num_pages: KV pages over all ranks.
        data_parallel_size: In-process data-parallel ranks.
        enable_prefix_caching: Prefix caching.
        max_num_seq_buckets: Request-count buckets; ``None`` is one bucket of the rows per rank.
        async_scheduling: Async scheduling with output placeholders.
        long_prefill_token_threshold: Per-step prefill cap; ``None`` or non-positive is off.
        chunked_prefill_enabled: Allow a prefill to span steps when it does not fit the remaining budget.
        token_safety_margin: Tokens reserved per running request in a KV-aware budget; ``None`` is off.
        policy: ``"fcfs"`` or ``"priority"`` (lower ``priority`` first).
        num_draft_tokens: Speculative drafts per step (0 off).
        use_eagle: EAGLE-style lookahead reservation (only with drafts).
        include_finished_set: Report every freed request in ``EngineCoreOutputs.finished_requests``.
        recurrent_prefix_snapshot_slots: Recurrent-snapshot slots per rank (0 off; needs prefix caching).
        image_expander: The model's image-placeholder expander (``esurge_image_expander_builder``), or ``None``.
        dp_admission_hold: Data-parallel admission alignment, in steps (0 off; used only with more than one rank):
            the ranks start new requests in the same step, holding admission at most this many steps.
    """

    def __init__(
        self,
        *,
        kv_cache_groups: list[CacheGroupSpec],
        max_num_seqs: int,
        max_num_batched_tokens: int | None,
        max_model_len: int,
        num_pages: int,
        data_parallel_size: int = 1,
        enable_prefix_caching: bool = True,
        max_num_seq_buckets: list[int] | None = None,
        async_scheduling: bool = True,
        long_prefill_token_threshold: int | None = None,
        chunked_prefill_enabled: bool = False,
        token_safety_margin: int | None = None,
        policy: typing.Literal["fcfs", "priority"] = "fcfs",
        num_draft_tokens: int = 0,
        use_eagle: bool = False,
        include_finished_set: bool = False,
        recurrent_prefix_snapshot_slots: int = 0,
        image_expander: typing.Any | None = None,
        dp_admission_hold: int = 0,
    ) -> None:
        from surgecore import CoreClosed, EngineCore, InvariantError

        if policy not in ("fcfs", "priority"):
            raise ValueError(f"Unknown scheduling policy: {policy}")
        self.max_model_len = int(max_model_len)
        self.max_num_scheduled_tokens = (
            int(max_num_batched_tokens) if max_num_batched_tokens is not None else self.max_model_len
        )
        self.async_scheduling = bool(async_scheduling)
        self.data_parallel_size = max(1, int(data_parallel_size))
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.policy = policy
        self.num_pages = int(num_pages)
        self.num_draft_tokens = int(num_draft_tokens)
        self.include_finished_set = bool(include_finished_set)
        self.image_expander = image_expander
        self.num_groups = len(kv_cache_groups)
        buckets = [int(b) for b in (max_num_seq_buckets or []) if int(b) > 0]
        self._core = EngineCore(
            max_rows=int(max_num_seqs),
            max_model_len=self.max_model_len,
            num_pages=self.num_pages,
            dp_size=self.data_parallel_size,
            groups=cache_group_rows(kv_cache_groups),
            token_budget=self.max_num_scheduled_tokens,
            long_prefill_threshold=max(0, int(long_prefill_token_threshold or 0)),
            chunked_prefill=bool(chunked_prefill_enabled),
            token_safety_margin=-1 if token_safety_margin is None else int(token_safety_margin),
            policy=1 if policy == "priority" else 0,
            prefix_caching=self.enable_prefix_caching,
            snapshot_slots=int(recurrent_prefix_snapshot_slots) if self.enable_prefix_caching else 0,
            async_scheduling=self.async_scheduling,
            req_buckets=np.asarray(buckets, np.int64) if buckets else None,
            num_draft_tokens=self.num_draft_tokens,
            use_eagle=bool(use_eagle and self.num_draft_tokens > 0),
            include_finished_set=self.include_finished_set,
            dp_admission_hold=max(0, int(dp_admission_hold)),
        )
        self.closed_error: type[BaseException] = CoreClosed
        """Raised by every call after :meth:`shutdown`; the loop exits cleanly on it."""
        self.invariant_error: type[BaseException] = InvariantError
        """Raised when an internal check fails; the core is then unusable (fatal for the engine)."""
        self._handles: dict[str, int] = {}
        self._records: dict[int, AdmissionRecord] = {}
        self._names: dict[int, str] = {}
        self._reports_due: dict[int, int] = {}
        self._steps: dict[int, _Step] = {}
        self._empty: list[str] = []
        """Admitted requests with an empty prompt: dropped (aborted) at the next schedule, as the former scheduler did."""
        self._empty_freed: list[str] = []

    def set_verify_budget(
        self,
        step_ms: dict[int, float],
        *,
        prior: float = 0.7,
        alpha: float = 0.2,
        correction: float = 0.1,
    ) -> None:
        """Turn on the core's load-aware verification budget for synchronous speculative steps.

        Each schedule then verifies 0..k drafts per request, chosen to maximise the expected emitted tokens per
        millisecond of the step's token bucket, and sets ``SchedulerOutput.num_draft_tokens_next``.

        Args:
            step_ms: Measured model-step time in milliseconds per compiled token bucket.
            prior: Starting conditional acceptance of every draft position.
            alpha: Weight of each acceptance observation in the moving averages.
            correction: Weight of each measured step time in the table's online correction (0 off).

        Raises:
            ValueError: No speculative tokens, async scheduling, or a bad table or weight.
        """
        buckets = sorted(int(b) for b in step_ms)
        self._core.set_verify_budget(
            np.asarray(buckets, np.int64),
            np.asarray([float(step_ms[b]) for b in buckets], np.float64),
            prior=float(prior),
            alpha=float(alpha),
            correction=float(correction),
        )

    def set_step_cost(self, step_ms: dict[int, float]) -> None:
        """Turn on the core's bucket-aligned prefill chunks (only with chunked prefill).

        A step a few tokens past a compiled token bucket then gives the excess back from its newest prefill chunks, to
        be scheduled by the next step, when the measured step times predict that the lower bucket plus the deferred
        tokens' share of a later step is faster.

        Args:
            step_ms: Measured model-step time in milliseconds per compiled token bucket.

        Raises:
            ValueError: A bad table (empty, non-ascending or non-positive buckets, non-positive times).
        """
        buckets = sorted(int(b) for b in step_ms)
        self._core.set_step_cost(
            np.asarray(buckets, np.int64),
            np.asarray([float(step_ms[b]) for b in buckets], np.float64),
        )

    @classmethod
    def from_runner(
        cls,
        runner: eSurgeRunner,
        *,
        max_num_batched_tokens: int | None = None,
        enable_prefix_caching: bool = True,
        async_scheduling: bool = True,
        long_prefill_token_threshold: int | None = None,
        num_draft_tokens: int = 0,
        chunked_prefill_enabled: bool = False,
        token_safety_margin: int | None = None,
        include_finished_set: bool = False,
        dp_admission_hold: int = 0,
    ) -> EngineCoreDriver:
        """Build the core for a runner: its cache groups, pages, rows, request buckets, DP size and snapshots.

        Prefix caching is turned off for compressed-window cache models (every token must flow through the per-slot
        recurrence) and for recurrent models without prefix snapshots (a page hit is only valid with a snapshot of the
        recurrent state at the same boundary).
        """
        from ..core.interface import create_kv_cache_specs_from_config

        metadata = runner.metadata
        if enable_prefix_caching and getattr(runner.model, "esurge_cache_family", None) == "compressed_window":
            logger.warning("Disabling prefix caching: compressed-window cache models must process every prompt token.")
            enable_prefix_caching = False
        snapshot_slots = 0
        if enable_prefix_caching and runner_has_recurrent_state(runner):
            snapshot_slots = int(getattr(runner, "recurrent_prefix_snapshot_slots", 0) or 0)
            if snapshot_slots <= 0:
                logger.warning(
                    "Disabling prefix caching: recurrent-state prefix snapshots are unavailable for this runner "
                    "(speculative decoding, compressed-window caches, or cache.recurrent_prefix_snapshots=0)."
                )
                enable_prefix_caching = False
        if max_num_batched_tokens is None:
            max_num_batched_tokens = runner.max_model_len
        kv_cache_groups = getattr(runner, "kv_cache_groups", None)
        if not kv_cache_groups:
            kv_cache_groups = create_kv_cache_specs_from_config(
                config=runner.model.config,
                page_size=metadata.page_size,
                num_kv_heads=metadata.num_kv_heads,
                head_size=getattr(metadata, "k_headdim", None) or getattr(metadata, "head_dim", None),
                dtype=metadata.kvdtype,
                use_mla=False,
            )
        expander_builder = getattr(type(runner.model), "esurge_image_expander_builder", None)
        driver = cls(
            kv_cache_groups=kv_cache_groups,
            max_num_seqs=runner.max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=runner.max_model_len,
            num_pages=metadata.num_pages,
            data_parallel_size=int(getattr(metadata, "data_parallel_size", 1) or 1),
            enable_prefix_caching=enable_prefix_caching,
            max_num_seq_buckets=list(runner.max_num_seq_buckets) if runner.max_num_seq_buckets else None,
            async_scheduling=async_scheduling,
            long_prefill_token_threshold=long_prefill_token_threshold,
            chunked_prefill_enabled=chunked_prefill_enabled,
            token_safety_margin=token_safety_margin,
            num_draft_tokens=int(num_draft_tokens),
            include_finished_set=include_finished_set,
            recurrent_prefix_snapshot_slots=snapshot_slots,
            image_expander=expander_builder(runner.model) if expander_builder is not None else None,
            dp_admission_hold=dp_admission_hold,
        )
        step_ms = getattr(runner, "step_time_table", None)
        if step_ms:
            driver.set_step_cost(step_ms)
        if step_ms and driver.num_draft_tokens > 0 and not driver.async_scheduling:
            driver.set_verify_budget(step_ms)
        return driver

    # ------------------------------------------------------------------------------------------------ admission
    def add_requests(self, records: list[AdmissionRecord]) -> None:
        """Admit requests in order (``n>1`` siblings in one call); atomic: a bad request admits none.

        Raises:
            ValueError: A duplicate live id, images the model cannot expand, an image span longer than the token
                budget, or a prompt that does not fit ``max_model_len``.
        """
        if not records:
            return
        ids = [r.request_id for r in records]
        if len(set(ids)) != len(ids) or any(rid in self._handles or rid in self._empty for rid in ids):
            raise ValueError(f"Requests {ids} include an id that is already scheduled")
        for r in records:
            if r.images is not None and len(r.images) > 0:
                expand_request_images(
                    r,
                    self.image_expander,
                    max_num_scheduled_tokens=self.max_num_scheduled_tokens,
                    max_model_len=self.max_model_len,
                )
        empty = [r.request_id for r in records if not r.prompt_token_ids]
        records = [r for r in records if r.prompt_token_ids]
        self._empty.extend(empty)
        if not records:
            return
        n = len(records)
        prompt_offsets = np.zeros(n + 1, np.int32)
        stop_offsets = np.zeros(n + 1, np.int32)
        span_offsets = np.zeros(n + 1, np.int32)
        max_tokens = np.empty(n, np.int64)
        eos = np.empty(n, np.int32)
        priority = np.empty(n, np.int64)
        arrival = np.empty(n, np.float64)
        ignore_eos = np.zeros(n, np.bool_)
        flags = np.zeros(n, np.uint32)
        temperature = np.ones(n, np.float32)
        top_p = np.ones(n, np.float32)
        min_p = np.zeros(n, np.float32)
        freq = np.zeros(n, np.float32)
        pres = np.zeros(n, np.float32)
        rep = np.ones(n, np.float32)
        top_k = np.zeros(n, np.int32)
        seed = np.full(n, -1, np.int64)
        hash_extra = np.zeros((n, 2), np.uint64)
        prompts: list[int] = []
        stops: list[int] = []
        spans: list[tuple[int, int]] = []
        for i, r in enumerate(records):
            prompts.extend(r.prompt_token_ids)
            prompt_offsets[i + 1] = len(prompts)
            sp = r.sampling_params
            max_tokens[i] = r.max_tokens
            eos[i] = -1 if r.eos_token_id is None else int(r.eos_token_id)
            priority[i] = int(r.priority)
            arrival[i] = float(r.arrival_time)
            if sp is not None:
                ignore_eos[i] = bool(sp.ignore_eos)
                stops.extend(int(t) for t in (sp.stop_token_ids or ()))
                if getattr(sp, "guided_decoding", None) is not None:
                    flags[i] |= _FLAG_GRAMMAR
                if sp.prompt_logprobs is not None:
                    flags[i] |= _FLAG_NO_PREFIX_HIT
                temperature[i] = 1.0 if sp.temperature is None else float(sp.temperature)
                top_p[i], min_p[i], top_k[i] = float(sp.top_p), float(sp.min_p), int(sp.top_k)
                freq[i], pres[i] = float(sp.frequency_penalty), float(sp.presence_penalty)
                rep[i] = float(sp.repetition_penalty)
                seed[i] = -1 if sp.seed is None else int(sp.seed)
            if r.has_vision:
                flags[i] |= _FLAG_VISION
            extra = page_hash_extra_key(r)
            hash_extra[i] = ((extra >> 64) & _U64, extra & _U64)
            stop_offsets[i + 1] = len(stops)
            spans.extend((int(a), int(b)) for a, b in r.image_spans)
            span_offsets[i + 1] = len(spans)
        handles = self._core.add_requests(
            prompt_tokens=np.asarray(prompts, np.int32),
            prompt_offsets=prompt_offsets,
            max_tokens=max_tokens,
            eos_token=eos,
            priority=priority,
            arrival=arrival,
            ignore_eos=ignore_eos,
            flags=flags,
            stop_tokens=np.asarray(stops, np.int32),
            stop_offsets=stop_offsets,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            frequency_penalty=freq,
            presence_penalty=pres,
            repetition_penalty=rep,
            top_k=top_k,
            seed=seed,
            hash_extra=hash_extra,
            image_spans=np.asarray(spans, np.int64).reshape(-1, 2),
            span_offsets=span_offsets,
        )
        for r, h in zip(records, handles.tolist(), strict=True):
            self._handles[r.request_id] = h
            self._records[h] = r
            self._names[h] = r.request_id

    def finish_requests(self, request_ids: str | Iterable[str], reason: int) -> list[str]:
        """Finish live requests in order (``FINISH_STOP`` for a stop string, ``FINISH_ABORT``); returns those freed."""
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        handles: list[int] = []
        for rid in request_ids:
            h = self._handles.get(rid)
            if h is not None and h not in handles:
                handles.append(h)
        if not handles:
            return []
        live = self._core.finish(np.asarray(handles, np.int64), int(reason)).tolist()
        out = []
        for h in live:
            rid = self._names[h]
            self._drop_live(rid, h)
            out.append(rid)
        return out

    # ------------------------------------------------------------------------------------------------ registry
    def _drop_live(self, rid: str, h: int) -> None:
        if self._handles.get(rid) == h:
            del self._handles[rid]

    def _reported(self, h: int) -> None:
        """One of the core's reports that ``h`` was freed (plan ``finished``; ``finished_all`` when requested)."""
        due = self._reports_due.get(h, 2 if self.include_finished_set else 1) - 1
        if due > 0:
            self._reports_due[h] = due
            return
        self._reports_due.pop(h, None)
        rid = self._names.pop(h, None)
        self._records.pop(h, None)
        if rid is not None:
            self._drop_live(rid, h)

    def is_live(self, request_id: str) -> bool:
        """Whether the request is waiting, running or preempted (not yet finished)."""
        return request_id in self._handles or request_id in self._empty

    def live_request_ids(self) -> list[str]:
        """Ids of every unfinished request, in admission order."""
        return [*self._handles, *self._empty]

    def record_of(self, request_id: str) -> AdmissionRecord | None:
        """The admission record of a live request."""
        h = self._handles.get(request_id)
        return None if h is None else self._records.get(h)

    def request_state(self, request_id: str) -> tuple[int, int, int, list[int]] | None:
        """``(status, computed, num_tokens, drafts)`` of a live request, or ``None`` once it finished.

        ``status`` is 0 waiting, 1 running, 2 preempted; ``drafts`` are the speculative drafts it carries into its next
        step (-1 placeholders under async scheduling).
        """
        h = self._handles.get(request_id)
        if h is None:
            return None
        info = self._core.request_info(h)
        if info is None:
            return None
        _rank, status, _num_prompt, num_tokens, computed, _placeholders, _cached = info
        drafts = self._core.drafts_of(h)
        return int(status), int(computed), int(num_tokens), [] if drafts is None else [int(t) for t in drafts]

    def children_of(self, parent_request_id: str) -> list[str]:
        """Ids of the live ``n>1`` samples of ``parent_request_id``."""
        return [rid for rid, h in self._handles.items() if self._records[h].parent_request_id == parent_request_id]

    def counts(self) -> list[int]:
        """Running, waiting, free pages and pending prefill tokens, then the same four per data-parallel rank."""
        return self._core.counts().tolist()

    def _counts(self) -> list[int]:
        return self.counts()

    @property
    def num_running(self) -> int:
        """Running requests over every rank."""
        return int(self._counts()[0])

    @property
    def num_waiting(self) -> int:
        """Waiting and preempted requests over every rank."""
        return int(self._counts()[1])

    @property
    def num_free_pages(self) -> int:
        """Free KV pages over every rank."""
        return int(self._counts()[2])

    @property
    def pending_prefill_tokens(self) -> int:
        """Prompt tokens not yet computed: the running requests' rest plus the waiting requests' prompts."""
        return int(self._counts()[3])

    # ------------------------------------------------------------------------------------------------ steps
    def schedule(self) -> SchedulerOutput:
        """Schedule one step and build the runner's :class:`SchedulerOutput` from the core's plan."""
        start = time.time()
        plan = self._core.schedule()
        handles = plan.handles.tolist()
        names = self._names
        rids = [names[h] for h in handles]
        n, groups, dp = len(handles), self.num_groups, self.data_parallel_size
        scheduled = plan.scheduled.tolist()
        computed = plan.computed.tolist()
        kinds = plan.kinds.tolist()
        pages = plan.new_pages.tolist()
        page_off = plan.new_page_offsets.tolist()
        drafts = plan.draft_tokens.tolist()
        draft_off = plan.draft_offsets.tolist()
        tokens = plan.new_tokens.tolist()
        token_off = plan.new_token_offsets.tolist()
        rank_off = plan.rank_offsets.tolist()
        stats = plan.stats.tolist()

        new_reqs: list[NewRequestData] = []
        c_ids: list[str] = []
        c_resumed: list[bool] = []
        c_tokens: list[list[int]] = []
        c_pages: list[tuple[list[int], ...]] = []
        c_computed: list[int] = []
        c_ranks: list[int | None] = []
        per_rank: dict[int, list[str]] = {}
        for rank in range(len(rank_off) - 1):
            lo, hi = rank_off[rank], rank_off[rank + 1]
            dp_rank = rank if dp > 1 else None
            if dp > 1:
                per_rank[rank] = rids[lo:hi]
            resumed: list[tuple[int, tuple[list[int], ...]]] = []
            for i in range(lo, hi):
                base = i * groups
                page_ids = tuple(pages[page_off[base + g] : page_off[base + g + 1]] for g in range(groups))
                if kinds[i] == _KIND_NEW:
                    rec = self._records[handles[i]]
                    new_reqs.append(
                        NewRequestData(
                            req_id=rids[i],
                            prompt_token_ids=rec.prompt_token_ids,
                            sampling_params=rec.sampling_params,
                            page_ids=page_ids,
                            num_computed_tokens=computed[i],
                            dp_rank=dp_rank,
                            pixel_values=rec.pixel_values,
                            image_grid_thw=rec.image_grid_thw,
                            pixel_values_videos=rec.pixel_values_videos,
                            video_grid_thw=rec.video_grid_thw,
                            mm_features=rec.mm_features,
                        )
                    )
                    continue
                if kinds[i] == _KIND_RESUMED:
                    resumed.append((i, page_ids))
                    continue
                resumed_entry = False
                c_ids.append(rids[i])
                c_resumed.append(resumed_entry)
                c_tokens.append(tokens[token_off[i] : token_off[i + 1]])
                c_pages.append(page_ids)
                c_computed.append(computed[i])
                c_ranks.append(dp_rank)
            for i, page_ids in resumed:
                c_ids.append(rids[i])
                c_resumed.append(True)
                c_tokens.append(tokens[token_off[i] : token_off[i + 1]])
                c_pages.append(page_ids)
                c_computed.append(computed[i])
                c_ranks.append(dp_rank)

        finished: set[str] = set()
        for h in plan.finished.tolist():
            rid = names[h]
            finished.add(rid)
            self._drop_live(rid, h)
        for h in plan.finished.tolist():
            self._reported(h)
        if self._empty:
            for rid in self._empty:
                logger.warning("Dropping request %s: zero schedulable prompt tokens (empty prompt).", rid)
            finished.update(self._empty)
            if self.include_finished_set:
                self._empty_freed.extend(self._empty)
            self._empty = []
        preempted = {names[h] for h in plan.preempted.tolist()}

        def pairs(arr) -> dict[str, int]:
            return {rids[int(e)]: int(s) for e, s in arr.tolist()}

        so = SchedulerOutput(
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=CachedRequestData(
                req_ids=c_ids,
                resumed_from_preemption=c_resumed,
                new_token_ids=c_tokens,
                new_page_ids=c_pages,
                num_computed_tokens=c_computed,
                dp_ranks=c_ranks,
            ),
            num_scheduled_tokens=dict(zip(rids, scheduled, strict=True)),
            total_num_scheduled_tokens=int(sum(scheduled)),
            scheduled_spec_decode_tokens={
                rids[i]: drafts[draft_off[i] : draft_off[i + 1]] for i in range(n) if draft_off[i + 1] > draft_off[i]
            },
            num_common_prefix_pages=plan.common_prefix_pages.tolist(),
            finished_req_ids=finished,
            preempted_req_ids=preempted,
            suggested_bucket=int(plan.req_bucket),
            async_scheduling=self.async_scheduling,
            num_running_reqs=int(stats[1]),
            num_waiting_reqs=int(stats[2]),
            free_pages=int(stats[3]),
            token_budget_initial=int(stats[4]),
            token_budget_remaining=int(stats[5]),
            req_id_to_dp_rank=({rid: rank for rank, ids in per_rank.items() for rid in ids} if dp > 1 else {}),
            max_num_scheduled_tokens_per_dp_rank=int(stats[7]) if dp > 1 else 0,
            req_ids_per_rank=per_rank,
            pending_structured_output_tokens=bool(plan.flags & _PLAN_STRUCTURED_PENDING),
            recurrent_snapshot_restores=pairs(plan.snapshot_restores),
            recurrent_snapshot_saves=pairs(plan.snapshot_saves),
            num_draft_tokens_next=int(plan.num_draft_tokens_next),
            spec_slots={rids[i]: s for i, s in enumerate(plan.spec_slots.tolist()) if s > 0},
            inflight_drafts={rids[i]: d for i, d in enumerate(plan.inflight_drafts.tolist()) if d >= 0},
            async_safe=bool(plan.flags & _PLAN_ASYNC_SAFE),
        )
        if n or zero_schedule_needs_update(so):
            self._steps[id(so)] = _Step(so, plan, handles, rids)
        collector = get_metrics_collector()
        if collector:
            collector.record_scheduler_metrics(
                num_waiting=int(stats[2]),
                num_running=int(stats[1]),
                num_scheduled_tokens=so.total_num_scheduled_tokens,
                num_preempted=int(stats[6]),
                batch_size=n,
                schedule_time=time.time() - start,
            )
            collector.record_cache_metrics(total_pages=self.num_pages, used_pages=self.num_pages - int(stats[3]))
        return so

    def can_prefetch(self, current: SchedulerOutput) -> bool:
        """Whether ``schedule()`` may run while ``current`` is in flight (synchronous overlap).

        Only when no scheduled request has an output placeholder and none has reached the point where its next
        decision depends on the sampled token (``computed >= tokens``).
        """
        step = self._steps.get(id(current))
        if step is None or step.so is not current:
            return False
        for h in step.handles:
            info = self._core.request_info(h)
            if info is None:
                continue
            _, _, _, num_tokens, computed, placeholders, _ = info
            if placeholders > 0 or computed >= num_tokens:
                return False
        return True

    def commit(self, scheduler_output: SchedulerOutput, model_output: ModelRunnerOutput) -> dict[int, EngineCoreOutputs]:
        """Commit a step's runner output (steps commit in schedule order) and return the engine outputs.

        Raises:
            RuntimeError: ``scheduler_output`` was not returned by :meth:`schedule`, or is already committed.
            ValueError: The runner's logprobs do not hold one position per sampled token.
        """
        step = self._steps.pop(id(scheduler_output), None)
        if step is None or step.so is not scheduler_output:
            raise RuntimeError("commit() of a schedule this engine core did not return, or that was already committed")
        index = model_output.req_id_to_index or {}
        sampled = model_output.sampled_token_ids
        spec = model_output.spec_token_ids
        spec_emitted = model_output.spec_emitted
        spec_drafts = model_output.spec_drafts
        logprobs = model_output.logprobs
        logprob_offsets: list[int] = []
        if logprobs is not None:
            logprob_offsets = list(itertools.accumulate((len(t) for t in sampled), initial=0))
            if any(len(field) != logprob_offsets[-1] for field in logprobs):
                raise ValueError("Runner logprobs must contain one position per sampled token in output order.")
        n = len(step.handles)
        if n:
            accepted_by_id = model_output.num_accepted_spec_tokens or {}
            present = np.ones(n, np.uint8)
            accepted = np.full(n, -1, np.int32)
            flat: list[int] = []
            offsets = [0]
            new_drafts: list[int] = []
            draft_offsets = [0]
            for i, rid in enumerate(step.rids):
                out = index.get(rid)
                if out is None:
                    present[i] = 0
                    logger.warning(
                        "Skipping scheduler output for request %s: missing req_id_to_index entry in model output.", rid
                    )
                elif sampled and not 0 <= out < len(sampled):
                    present[i] = 2
                    logger.warning(
                        "Skipping scheduler output for request %s: req_index=%s out of sampled_token_ids range=%s.",
                        rid,
                        out,
                        len(sampled),
                    )
                else:
                    row = (model_output.req_id_to_row_index or {}).get(rid, out)
                    count = accepted_by_id.get(rid)
                    row_tokens = None
                    if count is not None and spec_emitted is not None and 0 <= row < len(spec_emitted):
                        # Only a request that verified drafts this step has its emission packed: its accepted drafts
                        # plus one token. Any other row (a plain decode row riding the same step, or a pooled draft's
                        # seed packed by a path that resolved acceptance on the host) commits the runner's sampled
                        # tokens.
                        row_tokens = [int(t) for t in spec_emitted[row] if int(t) >= 0]
                        if len(row_tokens) != int(count) + 1:
                            row_tokens = None
                    if row_tokens is not None:
                        flat.extend(row_tokens)
                    elif sampled:
                        flat.extend(sampled[out])
                    packed_drafts = None
                    if spec_drafts is not None and 0 <= row < len(spec_drafts):
                        packed_drafts = [int(t) for t in spec_drafts[row] if int(t) >= 0]
                    if packed_drafts:
                        new_drafts.extend(packed_drafts)
                    elif spec is not None and out < len(spec):
                        new_drafts.extend(spec[out])
                    if count is not None:
                        accepted[i] = int(count)
                offsets.append(len(flat))
                draft_offsets.append(len(new_drafts))
            events = self._core.commit(
                step.plan,
                sampled_tokens=np.asarray(flat, np.int32),
                sampled_offsets=np.asarray(offsets, np.int32),
                accepted=accepted,
                new_drafts=None if spec is None else np.asarray(new_drafts, np.int32),
                new_draft_offsets=None if spec is None else np.asarray(draft_offsets, np.int32),
                present=present,
            )
        else:
            events = self._core.commit(step.plan)

        names = dict(zip(step.handles, step.rids, strict=True))
        prompt_logprobs = model_output.prompt_logprobs_dict or {}
        ev_tokens = events.tokens.tolist()
        ev_off = events.token_offsets.tolist()
        ev_finish = events.finish.tolist()
        ev_stop = events.stop_token.tolist()
        ev_cached = events.cached_tokens.tolist()
        outputs: list[EngineCoreOutput] = []
        for k, h in enumerate(events.handles.tolist()):
            rid = names[h]
            new_tokens = ev_tokens[ev_off[k] : ev_off[k + 1]]
            rec = self._records.get(h)
            new_logprobs = None
            if logprobs is not None and rec is not None and rec.sampling_params is not None:
                if rec.sampling_params.logprobs is not None:
                    start = logprob_offsets[index[rid]]
                    new_logprobs = logprobs.slice(start, start + len(new_tokens))
            code = ev_finish[k]
            outputs.append(
                EngineCoreOutput(
                    request_id=rid,
                    new_token_ids=new_tokens,
                    new_logprobs=new_logprobs,
                    finish_reason=_FINISH_REASON[code],
                    new_prompt_logprobs_tensors=prompt_logprobs.get(rid),
                    stop_reason=ev_stop[k] if ev_stop[k] >= 0 else None,
                    num_cached_tokens=ev_cached[k],
                )
            )
            if code:
                self._drop_live(rid, h)
        timestamp = time.perf_counter()
        result: dict[int, EngineCoreOutputs] = {}
        if outputs:
            result[0] = EngineCoreOutputs(outputs=outputs, timestamp=timestamp)
        if self.include_finished_set:
            freed = events.finished_all.tolist()
            if freed or self._empty_freed:
                finished_ids = {self._names[h] for h in freed} | set(self._empty_freed)
                self._empty_freed = []
                for h in freed:
                    self._reported(h)
                if 0 in result:
                    result[0].finished_requests = finished_ids
                else:
                    result[0] = EngineCoreOutputs(finished_requests=finished_ids, timestamp=timestamp)
        return result

    # ------------------------------------------------------------------------------------------------ lifecycle
    def reset_prefix_cache(self) -> bool:
        """Forget every published page and recurrent snapshot; ``False`` while pages are in use."""
        return bool(self._core.reset_prefix_cache())

    def check(self) -> None:
        """Run the core's invariant checks (tests and debugging); raises ``InvariantError`` on a failure."""
        self._core.check()

    def shutdown(self) -> None:
        """Close the core; later calls raise :attr:`closed_error`."""
        self._core.shutdown()


__all__ = ("FINISH_ABORT", "FINISH_LENGTH", "FINISH_STOP", "EngineCoreDriver", "cache_group_rows")
