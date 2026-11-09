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

"""Typed stubs of the compiled extension module ``surgecore._native``.

One delimited section per Rust module (payload, kv, engine, detokenization). Edit only between a section's begin and end markers.
"""

import numpy as np
import numpy.typing as npt

__version__: str

# ---- begin section: detokenization ----

class DetokenizerResult:
    """One incremental decode result; full text is present only at finish or resynchronization."""

    accumulated_text: str | None
    delta_text: str
    last_decoded_index: int
    finished: bool
    detoktook: float

    def __init__(
        self,
        accumulated_text: str | None,
        delta_text: str,
        last_decoded_index: int,
        finished: bool,
        detoktook: float = 0.0,
    ) -> None: ...

class Detokenizer:
    """In-process batched incremental detokenization over a HuggingFace tokenizer.

    Each request dictionary contains ``request_id``, ``tokens_delta``, ``token_offset``, ``finished``,
    ``skip_special_tokens``, ``spaces_between_special_tokens``, optional ``prompt_context`` and ``text_offset``.
    The native state owns the complete token and text history; steady-state calls and results carry only deltas.
    """

    def __init__(self, tokenizer: object, *, context_window: int = 4, max_states: int = 65536) -> None: ...
    def decode_batch(self, requests: list[dict[str, object]]) -> list[DetokenizerResult]: ...
    def reset(self, request_id: str) -> None: ...
    def drain(self) -> None: ...
    def shutdown(self) -> None: ...
    @property
    def num_states(self) -> int: ...

# ---- end section: detokenization ----

# ---- begin section: payload ----

class HostPayloadBuilder:
    """Fills one batch preparer's preallocated staging arrays with a forward window's step inputs.

    ``buffers`` maps staging-array names to writeable, C-contiguous numpy arrays (int32, or float32 for
    ``packed_f32_padded``). They are registered once; every ``build`` writes into them in place with the GIL released.
    """

    def __init__(
        self,
        *,
        buffers: dict[str, npt.NDArray[np.generic]],
        max_num_reqs: int,
        num_reqs_max_model_len: int,
        data_parallel_size: int,
        use_slot_mapping: bool,
        use_request_distribution: bool,
        plain_recurrent_slots: bool,
        recurrent_slot_group_size: int,
        enable_spec_recurrent_commit: bool,
        enable_dp_local_page_path: bool,
        page_size: int,
        max_num_pages_per_req: int,
        slices_per_page: int,
        max_padded_slices: int,
    ) -> None: ...
    def build(
        self,
        num_tokens_static: int,
        scheduled_full: npt.ArrayLike,
        active_mask_full: npt.ArrayLike,
        token_ids: npt.ArrayLike,
        num_computed_tokens: npt.ArrayLike,
        temperature: npt.ArrayLike,
        top_p: npt.ArrayLike,
        top_k: npt.ArrayLike,
        min_p: npt.ArrayLike,
        frequency_penalties: npt.ArrayLike,
        presence_penalties: npt.ArrayLike,
        repetition_penalties: npt.ArrayLike,
        page_table: npt.ArrayLike,
        padded_num_reqs_in: int,
        window_row_indices: npt.ArrayLike | None = None,
        recurrent_slot_indices: npt.ArrayLike | None = None,
        spec_recurrent_commit: npt.ArrayLike | None = None,
        page_table_version: int | None = None,
        cached_page_table_version: int | None = None,
        cached_rows_to_copy: int | None = None,
        num_pages: int = 0,
        metadata_data_parallel_size: int = 1,
    ) -> tuple[int, int, int, bool, int]:
        """Fill the staging arrays for one window.

        Returns:
            ``(num_requests, padded_num_reqs, rows_to_copy, reuse_pages_tables, total_pages)``.

        Raises:
            ValueError: The window does not fit the bucket, the DP layout or the token history.
        """
        ...

def request_distribution_prefix(
    scheduled: npt.ArrayLike, num_computed_tokens: npt.ArrayLike, num_requests: int
) -> tuple[int, int, int]:
    """``(decode_end, prefill_end, num_requests)`` of eSurge's positional request-distribution contract."""
    ...

# ---- end section: payload ----

# ---- begin section: kv ----

class KvCore:
    """One KV page pool with its prefix cache and the page state of every request that touched it.

    ``groups`` holds one ``(kind, page_size, window_or_chunk, type_index)`` per cache group: kind 0 full attention,
    1 sliding window, 2 chunked local attention. Requests are addressed by the slot ``open`` returns.
    """

    def __init__(
        self, *, num_pages: int, groups: list[tuple[int, int, int, int]], max_model_len: int, caching: bool, eagle: bool
    ) -> None: ...
    def open(self, extra: int) -> int: ...
    def num_keys(self, slot: int) -> int: ...
    def keys(self, slot: int, tokens_: list[int] | None) -> list[int]: ...
    def lookup(
        self,
        slot: int,
        tokens_: list[int] | None,
        no_hit: bool,
        max_len: int,
        hint: int | None = None,
        dp: int | None = None,
    ) -> tuple[list[list[int]], int]: ...
    def probe(self, slot: int | None, tokens_: list[int] | None, extra: int, max_len: int) -> int: ...
    def allocate(
        self,
        slot: int,
        computed: int,
        n_new: int,
        n_hit: int,
        hits: list[list[int]] | None,
        lookahead: int,
        hint: int | None = None,
        dp: int | None = None,
    ) -> list[list[int]] | None: ...
    def needed(
        self, slot: int | None, num_tokens: int, hits: list[list[int]] | None = None, max_step: int | None = None
    ) -> int: ...
    def free(self, slot: int) -> bool: ...
    def forget_keys(self, slot: int) -> bool: ...
    def publish(
        self, slot: int, n: int, tokens_: list[int] | None = None, offset: int = 0
    ) -> tuple[int, int] | None: ...
    def page_ids(self, slot: int) -> list[list[int]]: ...
    def common_prefix(self, slot: int | None, num_scheduled: int) -> list[int]: ...
    def release_tail(self, slot: int, keep_tokens: int) -> int: ...
    def reset(self) -> bool: ...
    @property
    def num_pages(self) -> int: ...
    @property
    def num_free(self) -> int: ...
    @property
    def usage(self) -> float: ...
    @property
    def num_cached(self) -> int: ...
    @property
    def caching(self) -> bool: ...
    @property
    def num_open(self) -> int: ...
    def free_order(self) -> list[int]: ...
    def refs(self) -> list[int]: ...
    def check(self) -> None: ...

class SnapshotIndex:
    """Recurrent-snapshot slots: pending until committed; ready slots are recycled least recently used first."""

    def __init__(self, num_slots: int, slot_offset: int = 0) -> None: ...
    @property
    def num_slots(self) -> int: ...
    @property
    def slot_offset(self) -> int: ...
    def reset(self) -> None: ...
    def contains(self, key: int) -> bool: ...
    def is_ready(self, key: int) -> bool: ...
    def lookup(self, key: int) -> int | None: ...
    def reserve(self, key: int, exclude: object = None) -> int | None: ...
    def cancel(self, slot: int) -> None: ...
    def commit(self, slot: int) -> None: ...
    def state(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]: ...

# ---- end section: kv ----

# ---- begin section: engine ----

class InvariantError(RuntimeError):
    """An internal check of the engine core failed; the core is no longer usable (every later call raises it)."""

class CoreClosed(RuntimeError):
    """The engine core was shut down; every later call raises it."""

PLAN_ASYNC_SAFE: int  # 1
PLAN_HAS_PREFILL: int  # 2
PLAN_HAS_VERIFY: int  # 4
PLAN_PAGES_CHANGED: int  # 8
PLAN_STRUCTURED_PENDING: int  # 16

class Plan:
    """One step's scheduling decisions, one entry per scheduled request in decision order.

    Decision order is rank-major; within a rank the running requests in running order, then the requests admitted
    this step in admission order (the order of the former ``SchedulerOutput.num_scheduled_tokens``). Every property
    returns a fresh copy. A plan stays readable after it is committed.
    """

    @property
    def id(self) -> int:
        """The plan's id, consecutive per core."""
        ...
    @property
    def n(self) -> int:
        """Number of entries."""
        ...
    @property
    def flags(self) -> int:
        """``PLAN_*`` bits. bit0 (1) async-dispatch-safe: async scheduling, every entry one token and no grammar
        request pending. bit1 (2) some entry prefills prompt tokens. bit2 (4) some entry verifies speculative drafts.
        bit3 (8) page rows changed (new pages, new or resumed entries, preemptions or finishes). bit4 (16) a
        structured-output request still had an unresolved async placeholder when scheduled (the former
        ``pending_structured_output_tokens``)."""
        ...
    @property
    def handles(self) -> npt.NDArray[np.int64]:
        """``int64[n]``: the scheduled requests' handles."""
        ...
    @property
    def scheduled(self) -> npt.NDArray[np.int32]:
        """``int32[n]``: tokens scheduled per entry, drafts included (the former ``num_scheduled_tokens``)."""
        ...
    @property
    def computed(self) -> npt.NDArray[np.int32]:
        """``int32[n]``: computed tokens per entry before the step (the prefix hit for a new or resumed entry; the
        former ``num_computed_tokens`` of ``NewRequestData`` and ``CachedRequestData``)."""
        ...
    @property
    def kinds(self) -> npt.NDArray[np.uint8]:
        """``uint8[n]``: 0 running, 1 new (``scheduled_new_reqs``), 2 resumed after preemption
        (``resumed_from_preemption``)."""
        ...
    @property
    def emits(self) -> npt.NDArray[np.uint8]:
        """``uint8[n]``: 1 when the entry samples a token this step (async: it received an output placeholder)."""
        ...
    @property
    def new_pages(self) -> npt.NDArray[np.int32]:
        """Global page ids, flat: all of a new or resumed entry's pages, the newly allocated ones of a running entry
        (the former ``page_ids`` / ``new_page_ids``). Global id = ``rank * (num_pages // dp_size) + local``; the
        null page stays 0."""
        ...
    @property
    def new_page_offsets(self) -> npt.NDArray[np.int32]:
        """``int32[n * G + 1]`` offsets into ``new_pages``; entry ``i`` group ``g`` is range ``i * G + g``."""
        ...
    @property
    def rank_offsets(self) -> npt.NDArray[np.int32]:
        """``int32[dp + 1]``: entries of rank ``r`` are ``rank_offsets[r]:rank_offsets[r + 1]``."""
        ...
    @property
    def draft_tokens(self) -> npt.NDArray[np.int32]:
        """Speculative drafts each entry verifies this step, flat, after the budget trim (the former
        ``scheduled_spec_decode_tokens``)."""
        ...
    @property
    def draft_offsets(self) -> npt.NDArray[np.int32]:
        """``int32[n + 1]`` offsets into ``draft_tokens``; an entry without drafts has an empty range."""
        ...
    @property
    def new_tokens(self) -> npt.NDArray[np.int32]:
        """Known tokens each running or resumed entry feeds, flat: its tokens ``[computed, computed + scheduled -
        drafts)`` clipped to the tokens it has (the former ``CachedRequestData.new_token_ids``; under async shorter
        than ``scheduled`` by the placeholders; empty for a new entry)."""
        ...
    @property
    def new_token_offsets(self) -> npt.NDArray[np.int32]:
        """``int32[n + 1]`` offsets into ``new_tokens``."""
        ...
    @property
    def preempted(self) -> npt.NDArray[np.int64]:
        """``int64[P]``: requests preempted this step, in preemption order (the former ``preempted_req_ids``)."""
        ...
    @property
    def finished(self) -> npt.NDArray[np.int64]:
        """``int64[F]``: requests finished since the previous plan (commit stops, ``finish``, zero-token drops; the
        former ``finished_req_ids``)."""
        ...
    @property
    def snapshot_restores(self) -> npt.NDArray[np.int32]:
        """``int32[Rs, 2]``: ``(entry, global slot)`` recurrent-snapshot restores before the forward."""
        ...
    @property
    def snapshot_saves(self) -> npt.NDArray[np.int32]:
        """``int32[Sv, 2]``: ``(entry, global slot)`` recurrent-snapshot saves after the forward."""
        ...
    @property
    def common_prefix_pages(self) -> npt.NDArray[np.int32]:
        """``int32[G]``: leading pages per group shared by every request rank 0 scheduled (full-attention groups; the
        former ``num_common_prefix_pages``)."""
        ...
    @property
    def req_bucket(self) -> int:
        """The request-count bucket selected after scheduling, summed over ranks (the former ``suggested_bucket``)."""
        ...
    @property
    def num_draft_tokens_next(self) -> int:
        """Drafts the runner produces for the next step (0 skips the drafter): the configured count, or the
        verification budget's choice when one is set."""
        ...
    @property
    def spec_slots(self) -> npt.NDArray[np.int32]:
        """``int32[n]``: output slots per entry, ``1 + drafts`` for an entry that samples, else 0. An async step counts
        every slot as accepted until its commit; the device corrects a later step's start by the rejected drafts."""
        ...
    @property
    def inflight_drafts(self) -> npt.NDArray[np.int64]:
        """``int64[n]``: drafts of earlier uncommitted steps counted as accepted by an optimistic start; -1 when
        the entry starts from committed state. Zero means the earlier step verified no drafts."""
        ...
    @property
    def stats(self) -> npt.NDArray[np.int64]:
        """``int64[8 + 4 * dp]``: scheduled tokens, running, waiting, free pages, initial budget, remaining budget,
        preemptions, the busiest rank's tokens (``max_num_scheduled_tokens_per_dp_rank``); then per rank: scheduled
        tokens, running, waiting, free pages."""
        ...

class Events:
    """A commit's outcome in plan order: one entry per request that received at least one token. Owned arrays."""

    @property
    def n(self) -> int:
        """Number of entries."""
        ...
    @property
    def handles(self) -> npt.NDArray[np.int64]:
        """``int64[E]``: the requests that received tokens."""
        ...
    @property
    def token_offsets(self) -> npt.NDArray[np.int32]:
        """``int32[E + 1]`` offsets into ``tokens``."""
        ...
    @property
    def tokens(self) -> npt.NDArray[np.int32]:
        """The tokens appended per entry, truncated after a stop (the former ``EngineCoreOutput.new_token_ids``)."""
        ...
    @property
    def logprob_starts(self) -> npt.NDArray[np.int32]:
        """``int32[E]``: each entry's first position in the commit's full sampled list, before truncation
        (``sampled_offsets[i]`` of the plan entry)."""
        ...
    @property
    def finish(self) -> npt.NDArray[np.int8]:
        """``int8[E]``: 0 running, 1 stop (EOS or stop id), 2 length (``max_tokens`` or ``max_model_len``), 3 abort."""
        ...
    @property
    def stop_token(self) -> npt.NDArray[np.int32]:
        """``int32[E]``: the stop id that finished the request (the former ``stop_reason``), -1 otherwise (EOS too)."""
        ...
    @property
    def cached_tokens(self) -> npt.NDArray[np.int32]:
        """``int32[E]``: prefix-hit tokens of the request's first admission (``num_cached_tokens``)."""
        ...
    @property
    def finished_all(self) -> npt.NDArray[np.int64]:
        """``int64[F]``: every request freed since the previous commit, in free order, rank-major (only with
        ``include_finished_set``, else empty; the former ``EngineCoreOutputs.finished_requests``)."""
        ...

class EngineCore:
    """eSurge's host state machine: request table, per-rank queues, budgets, KV pages, prefix cache, snapshots.

    Every call copies its inputs, releases the GIL, then takes the core's mutex. A bad argument raises
    ``ValueError`` and changes nothing; an internal failure raises ``InvariantError`` and poisons the core.

    Args:
        max_rows: Running rows over all ranks (the runner's ``max_num_seqs``); each rank gets ``max_rows // dp_size``.
        max_model_len: Longest sequence (prompt plus output).
        num_pages: KV pages over all ranks; divisible by ``dp_size`` when it is above 1.
        dp_size: Data-parallel ranks (in process; 1 is a single scheduler).
        groups: ``int32[G, 3]`` or ``int32[G, 4]``: per cache group ``(kind, page_size, window_or_chunk[,
            type_index])``; kind 0 full attention, 1 sliding window, 2 chunked local. Without the type column, equal
            rows share a type.
        token_budget: Per-step token budget of each rank (``max_num_batched_tokens``).
        long_prefill_threshold: Per-step prefill cap; 0 is off.
        chunked_prefill: Allow a prefill to span steps when it does not fit the remaining budget.
        token_safety_margin: Tokens reserved for safety in a KV-aware budget; -1 is off (``None`` today).
        policy: 0 FCFS, 1 priority (lower ``priority`` first; highest ``(priority, arrival)`` preempted first).
        prefix_caching: Prefix caching.
        snapshot_slots: Recurrent-snapshot slots per rank (0 off; on only with prefix caching); rank ``r`` owns
            global slots ``[r * snapshot_slots, (r + 1) * snapshot_slots)``.
        async_scheduling: Async scheduling with output placeholders.
        req_buckets: Request-count buckets (``max_num_seq_buckets``); ``None`` is ``[max_rows // dp_size]``.
        num_draft_tokens: Speculative tokens per step (0 off): a request's drafts enter its scheduled tokens and an
            async step gives each request that samples ``1 + drafts`` placeholders.
        use_eagle: EAGLE-style drafting (only with ``num_draft_tokens > 0``): every allocation reserves
            ``num_draft_tokens`` lookahead slots and a prefix hit drops its last page.
        include_finished_set: Report every freed request in ``Events.finished_all``.
    """

    def __init__(
        self,
        *,
        max_rows: int,
        max_model_len: int,
        num_pages: int,
        dp_size: int,
        groups: npt.ArrayLike,
        token_budget: int,
        long_prefill_threshold: int = 0,
        chunked_prefill: bool = False,
        token_safety_margin: int = -1,
        policy: int = 0,
        prefix_caching: bool = True,
        snapshot_slots: int = 0,
        async_scheduling: bool = True,
        req_buckets: npt.ArrayLike | None = None,
        num_draft_tokens: int = 0,
        use_eagle: bool = False,
        include_finished_set: bool = False,
    ) -> None: ...
    def add_requests(
        self,
        *,
        prompt_tokens: npt.ArrayLike,
        prompt_offsets: npt.ArrayLike,
        max_tokens: npt.ArrayLike,
        min_tokens: npt.ArrayLike | None = None,
        eos_token: npt.ArrayLike | None = None,
        priority: npt.ArrayLike | None = None,
        arrival: npt.ArrayLike | None = None,
        ignore_eos: npt.ArrayLike | None = None,
        flags: npt.ArrayLike | None = None,
        stop_tokens: npt.ArrayLike | None = None,
        stop_offsets: npt.ArrayLike | None = None,
        temperature: npt.ArrayLike | None = None,
        top_p: npt.ArrayLike | None = None,
        min_p: npt.ArrayLike | None = None,
        frequency_penalty: npt.ArrayLike | None = None,
        presence_penalty: npt.ArrayLike | None = None,
        repetition_penalty: npt.ArrayLike | None = None,
        top_k: npt.ArrayLike | None = None,
        dp_rank: npt.ArrayLike | None = None,
        seed: npt.ArrayLike | None = None,
        hash_extra: npt.ArrayLike | None = None,
        image_spans: npt.ArrayLike | None = None,
        span_offsets: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.int64]:
        """Admits ``R`` requests in order, each routed after the previous one is queued; returns ``int64[R]``
        handles. Atomic: a request that fails validation raises ``ValueError`` and none is admitted.

        Args:
            prompt_tokens: ``int32[T]``: every prompt, image placeholders already expanded.
            prompt_offsets: ``int32[R + 1]`` offsets into ``prompt_tokens``; every prompt non-empty and shorter than
                ``max_model_len``.
            max_tokens: ``int64[R]`` output-token limits, positive.
            min_tokens: ``int64[R]``, stored only (not enforced, as today); default 0.
            eos_token: ``int32[R]`` EOS id, -1 none; default -1.
            priority: ``int64[R]``, lower first under the priority policy; default 0.
            arrival: ``float64[R]`` arrival times (the priority tie-break); default 0.
            ignore_eos: ``bool[R]``; an ignored EOS among the stop ids does not stop either; default False.
            flags: ``uint32[R]``: bit0 (1) structured output, bit1 (2) no prefix-cache hit (prompt logprobs),
                bit2 (4) vision inputs, bit3 (8) speculation-eligible (stored); default 0.
            stop_tokens: ``int32[S]`` stop ids of every request, flat.
            stop_offsets: ``int32[R + 1]`` offsets into ``stop_tokens``.
            temperature: ``float32[R]``, stored; default 1.
            top_p: ``float32[R]``, stored; default 1.
            min_p: ``float32[R]``, stored; default 0.
            frequency_penalty: ``float32[R]``, stored; default 0.
            presence_penalty: ``float32[R]``, stored; default 0.
            repetition_penalty: ``float32[R]``, stored; default 1.
            top_k: ``int32[R]``, stored; default 0.
            dp_rank: ``int64[R]`` rank to place each request on, -1 lets the core route it (most prefix-cached
                tokens, else fewest pending prefill tokens, ties to the lower rank); default -1.
            seed: ``int64[R]``, -1 none, stored; default -1.
            hash_extra: ``uint64[R, 2]``: the 128-bit multimodal page-key identity (high, low), 0 for text; mixed
                into every page key (today ``_extra_key``).
            image_spans: ``int64[I, 2]`` half-open, sorted image spans ``(start, stop)`` of prompt positions; a
                prefill chunk never ends inside one, and none may exceed ``token_budget``.
            span_offsets: ``int32[R + 1]`` offsets into ``image_spans``.
        """
        ...
    def schedule(self) -> Plan:
        """Schedules one step over every rank. A plan with entries is in flight until it is committed."""
        ...
    def commit(
        self,
        plan: Plan,
        *,
        sampled_tokens: npt.ArrayLike | None = None,
        sampled_offsets: npt.ArrayLike | None = None,
        accepted: npt.ArrayLike | None = None,
        new_drafts: npt.ArrayLike | None = None,
        new_draft_offsets: npt.ArrayLike | None = None,
        present: npt.ArrayLike | None = None,
    ) -> Events:
        """Commits a plan's runner output; plans with entries commit in the order ``schedule`` returned them (else
        ``ValueError``). Every array is per plan entry, in plan order. A commit of a plan without entries only
        reports ``finished_all``.

        Args:
            plan: The oldest plan in flight.
            sampled_tokens: ``int32[T]``: each entry's sampled list, flat: its accepted drafts then the bonus token
                (the former ``sampled_token_ids[out_index]``); empty for a prefill chunk. Tokens are appended one by
                one with the stop check after each, truncated at a stop. ``None`` with ``sampled_offsets=None`` is
                no tokens.
            sampled_offsets: ``int32[n + 1]`` offsets into ``sampled_tokens``.
            accepted: ``int32[n]`` accepted-draft counts, -1 absent (the former ``num_accepted_spec_tokens``); a
                count gives back ``len(plan drafts) - accepted`` computed tokens. ``None`` is absent for all.
            new_drafts: ``int32[D]``: each entry's drafts for the next step, flat (the former ``spec_token_ids``);
                ``None`` means the runner returned none and each request keeps its drafts. Cleared on a stop.
            new_draft_offsets: ``int32[n + 1]`` offsets into ``new_drafts``; given together with ``new_drafts``.
            present: ``uint8[n]``: 1 present; 0 absent from the runner output (nothing applied, no page publish, its
                snapshot save cancelled); 2 listed in the runner output without a sampled row (its snapshot save
                committed, nothing else applied). ``None`` is all present.
        """
        ...
    def finish(self, handles: npt.ArrayLike, reason: int) -> npt.NDArray[np.int64]:
        """Finishes live requests in order (``reason`` 1 stop string, 2 length, 3 abort): frees their pages at once;
        returns the handles that were live (stale and repeated handles are skipped)."""
        ...
    def reset_prefix_cache(self) -> bool:
        """Forgets every published page and snapshot, rank by rank; ``False`` at the first rank whose pages are in
        use."""
        ...
    def counts(self) -> npt.NDArray[np.int64]:
        """``int64[4 + 4 * dp]``: running, waiting, free pages, pending prefill tokens; then the same per rank."""
        ...
    def record(self, plan: Plan) -> bytes:
        """The plan as a step record (zigzag LEB128 varints, version 4)."""
        ...
    def digest(self) -> int:
        """A hash of every live request's status, tokens, counters and global page ids (never page keys)."""
        ...
    def check(self) -> None:
        """Checks the invariants; raises ``InvariantError`` (and poisons the core) when one fails."""
        ...
    def num_inflight(self) -> int:
        """Plans scheduled with entries and not yet committed."""
        ...
    def rank_of(self, handle: int) -> int | None:
        """The data-parallel rank of a live request, or ``None``."""
        ...
    def request_info(self, handle: int) -> tuple[int, int, int, int, int, int, int] | None:
        """``(rank, status, num_prompt, num_tokens, computed, placeholders, cached_tokens)`` of a live request
        (status 0 waiting, 1 running, 2 preempted; ``cached_tokens`` -1 before its first admission), or ``None``."""
        ...
    def drafts_of(self, handle: int) -> npt.NDArray[np.int32] | None:
        """The drafts a live request carries into its next step (-1 placeholders under async), or ``None``."""
        ...
    def set_verify_budget(
        self,
        buckets: npt.ArrayLike,
        step_ms: npt.ArrayLike,
        *,
        prior: float = 0.7,
        alpha: float = 0.2,
        correction: float = 0.1,
    ) -> None:
        """Turns on the load-aware verification budget of synchronous speculative steps.

        Each schedule then cuts every running request's drafts to its share (0..k), adding draft positions in
        decreasing estimated cumulative acceptance while the expected emitted tokens per millisecond of the step's
        bucket grow, and sets ``Plan.num_draft_tokens_next``. Commits update per-request acceptance estimates, the
        global prior and the step-time correction.

        Args:
            buckets: ``int64[B]``, ascending: the compiled token buckets of the model step.
            step_ms: ``float64[B]``: their measured step times in milliseconds.
            prior: Starting conditional acceptance of every draft position.
            alpha: Weight of each acceptance observation in the moving averages.
            correction: Weight of each measured step time in the table's correction (0 off).

        Raises:
            ValueError: No speculative tokens, or a bad table or weight.
        """
        ...
    def verify_state(self) -> tuple[list[float], float] | None:
        """The budget's global acceptance prior per draft position and its step-time correction (ms), or ``None``."""
        ...
    def shutdown(self) -> None:
        """Closes the core; later calls raise ``CoreClosed``."""
        ...

# ---- end section: engine ----
