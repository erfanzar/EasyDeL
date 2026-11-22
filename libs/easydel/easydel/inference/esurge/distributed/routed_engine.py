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

"""A data-parallel front over independent eSurge replicas.

:class:`RoutedEngine` presents the engine-adapter surface while fanning
requests across N members — local ``eSurge`` instances and/or
:class:`RemoteEngineHandle` clients — with the routing policy lifted from
the in-core DP routing: **prefix affinity first, then join-shortest-queue**.
Affinity is tracked router-side in a :class:`PrefixAffinityMap` (LRU of
block-aligned prompt-prefix hashes → member) instead of synchronously
probing every replica's prefix cache; the JSQ signal is
``pending_prefill_tokens``, read directly from local members and from the
heartbeat-piggybacked ``QueueStats`` for remote ones.

Health is fail-fast per member: a dead replica (heartbeat loss, shutdown)
is excluded from routing and its in-flight requests surface as aborted /
raised errors on their callers; there is no migration or retry. The other
members keep serving.
"""

from __future__ import annotations

import hashlib
import inspect
import threading
import typing
from collections import OrderedDict

from ..logger import logger
from .request_plane import RequestPlaneError

if typing.TYPE_CHECKING:
    from easydel.inference.sampling_params import SamplingParams

    from ..esurge_engine import RequestOutput

#: Fallback affinity block width (characters) when a member reports no page size.
_DEFAULT_BLOCK_CHARS = 512
#: Longest prefix (in blocks) tracked per prompt.
_MAX_BLOCKS_PER_PROMPT = 32


class PrefixAffinityMap:
    """LRU map from block-aligned prompt-prefix hashes to member index.

    Routing wants "the replica that most recently saw this prefix" without
    a synchronous cache probe per replica. Prompts are cut into fixed-width
    character blocks (a router-side proxy for page-aligned token blocks);
    each cumulative prefix hash maps to the member that last served it, and
    lookups return the member holding the longest matching prefix.

    Args:
        block_chars: Prefix block width in characters.
        capacity: Maximum retained prefix hashes (LRU-evicted).
    """

    def __init__(self, *, block_chars: int = _DEFAULT_BLOCK_CHARS, capacity: int = 16384) -> None:
        self._block_chars = max(1, int(block_chars))
        self._capacity = int(capacity)
        self._lock = threading.Lock()
        self._map: OrderedDict[str, int] = OrderedDict()

    def _prefix_hashes(self, prompt: str) -> list[str]:
        hashes = []
        hasher = hashlib.sha256()
        for start in range(0, min(len(prompt), self._block_chars * _MAX_BLOCKS_PER_PROMPT), self._block_chars):
            block = prompt[start : start + self._block_chars]
            if len(block) < self._block_chars and start > 0:
                break  # only full blocks participate after the first
            hasher.update(block.encode())
            hashes.append(hasher.hexdigest()[:24])
        return hashes

    def record(self, prompt: str, member_index: int) -> None:
        """Remember that ``member_index`` served this prompt's prefixes."""
        with self._lock:
            for prefix_hash in self._prefix_hashes(prompt):
                self._map.pop(prefix_hash, None)
                self._map[prefix_hash] = member_index
            while len(self._map) > self._capacity:
                self._map.popitem(last=False)

    def lookup(self, prompt: str) -> tuple[int | None, int]:
        """Find the member holding this prompt's longest known prefix.

        Args:
            prompt: The incoming prompt text.

        Returns:
            ``(member_index | None, matched_blocks)``.
        """
        best: int | None = None
        matched = 0
        with self._lock:
            for depth, prefix_hash in enumerate(self._prefix_hashes(prompt), start=1):
                member = self._map.get(prefix_hash)
                if member is None:
                    break
                best = member
                matched = depth
        return best, matched

    def drop_member(self, member_index: int) -> None:
        """Forget every prefix pinned to a (dead) member."""
        with self._lock:
            for key in [key for key, value in self._map.items() if value == member_index]:
                self._map.pop(key, None)


class _SlotCounter:
    """Per-member-instance in-flight request counter.

    Pinned to the member instance rather than its slot index so that
    :meth:`RoutedEngine.replace_member` gives the replacement a fresh counter
    and unwinding requests from the old member never decrement the replacement.
    """

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0


class _ReleasedIterator:
    """Iterator wrapper releasing a router in-flight slot on exhaustion, close, or GC.

    Unlike a generator function, closing or discarding this wrapper before the
    first ``next()`` still closes the wrapped iterator and releases the slot.
    """

    def __init__(self, iterator: typing.Iterator, release: typing.Callable[[], None]) -> None:
        self._iterator = iterator
        self._release = release
        self._closed = False

    def __iter__(self) -> _ReleasedIterator:
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            inner_close = getattr(self._iterator, "close", None)
            if inner_close is not None:
                inner_close()
        finally:
            self._release()

    def __enter__(self) -> _ReleasedIterator:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _member_alive(member) -> bool:
    """Read public liveness metadata, assuming unknown engine-shaped members are alive."""
    alive = getattr(member, "alive", None)
    return True if alive is None else bool(alive)


def _member_pending_prefill_tokens(member) -> int:
    """JSQ cost signal for a member (remote stats or a local engine-core count)."""
    remote = getattr(member, "pending_prefill_tokens", None)
    if remote is not None:
        return int(remote)
    core = getattr(member, "core", None)
    if core is None:
        return 0
    try:
        return int(core.pending_prefill_tokens)
    except Exception:
        return 0


class RoutedEngine:
    """Prefix-affinity + JSQ router over engine-shaped members.

    Args:
        members: Engines to route across — any mix of local ``eSurge``
            instances and :class:`RemoteEngineHandle` clients (anything
            exposing the adapter surface).
        affinity_block_chars: Prefix block width for the affinity map;
            defaults to a page-size-derived estimate when a member reports
            one (≈4 characters per token), else ``512`` characters.
        affinity_max_imbalance: How many more router-local in-flight requests
            the prefix-pinned member may hold than the least-loaded live member
            before affinity yields to join-shortest-queue. A prefix shared by a
            burst (one system prompt, one benchmark prompt) otherwise lands
            every request on one member while the others idle.
    """

    def __init__(
        self,
        members: list,
        *,
        affinity_block_chars: int | None = None,
        affinity_max_imbalance: int = 4,
        planned_members: int | None = None,
    ) -> None:
        if not members:
            raise ValueError("RoutedEngine needs at least one member")
        if int(affinity_max_imbalance) < 0:
            raise ValueError(f"affinity_max_imbalance must be >= 0, got {affinity_max_imbalance}")
        if planned_members is not None and int(planned_members) < len(members):
            raise ValueError(f"planned_members={planned_members} is below the {len(members)} members given")
        self.members = list(members)
        self.planned_members = None if planned_members is None else int(planned_members)
        self.affinity_max_imbalance = int(affinity_max_imbalance)
        if affinity_block_chars is None:
            page_size = next((int(getattr(m, "page_size", 0) or 0) for m in self.members), 0)
            affinity_block_chars = page_size * 4 if page_size > 0 else _DEFAULT_BLOCK_CHARS
        self.affinity = PrefixAffinityMap(block_chars=affinity_block_chars)
        self._route_lock = threading.Lock()
        # Router-local in-flight counts (one _SlotCounter per member instance):
        # the JSQ primary signal. Remote QueueStats ride 1s heartbeats, so a
        # burst arriving between beats would otherwise dump entirely on the
        # lowest-indexed member.
        self._slot_counters: list[_SlotCounter] = [_SlotCounter() for _ in self.members]

    @property
    def _inflight(self) -> list[int]:
        """Snapshot of router-local in-flight request counts per slot."""
        with self._route_lock:
            return [counter.count for counter in self._slot_counters]

    def add_member(self, member) -> int:
        """Attach a member that came up after the router started serving (a late replica).

        Returns:
            The new member's index.
        """
        with self._route_lock:
            # In-flight first: a reader that sees the member under the lock also sees its counter.
            self._slot_counters.append(_SlotCounter())
            self.members.append(member)
            return len(self.members) - 1

    def replace_member(self, index: int, member) -> None:
        """Put a fresh member in a dead member's slot (a replica that came back on the same address)."""
        with self._route_lock:
            self.members[index] = member
            self._slot_counters[index] = _SlotCounter()
        self.affinity.drop_member(index)

    def _pick_member_locked(self, pinned: int | None) -> tuple[int | None, list[int]]:
        """Pick the best live member index under ``self._route_lock``.

        Returns:
            ``(chosen_index | None, dead_indices)``; ``chosen_index`` is ``None``
            when no member is alive.
        """
        alive: list[int] = []
        dead: list[int] = []
        for index, member in enumerate(self.members):
            if _member_alive(member):
                alive.append(index)
            else:
                dead.append(index)
        if not alive:
            return None, dead
        if pinned is not None and pinned in alive:
            least = min(self._slot_counters[index].count for index in alive)
            if self._slot_counters[pinned].count - least <= self.affinity_max_imbalance:
                return pinned, dead
        chosen = min(
            alive,
            key=lambda index: (
                self._slot_counters[index].count,
                _member_pending_prefill_tokens(self.members[index]),
                index,
            ),
        )
        return chosen, dead

    def _alive_indices(self) -> list[int]:
        with self._route_lock:
            alive: list[int] = []
            dead: list[int] = []
            for index, member in enumerate(self.members):
                if _member_alive(member):
                    alive.append(index)
                else:
                    dead.append(index)
        for index in dead:
            self.affinity.drop_member(index)
        if not alive:
            raise RequestPlaneError("no live replica: every routed engine member is down")
        return alive

    def _acquire(self, member_index: int) -> None:
        with self._route_lock:
            self._slot_counters[member_index].count += 1

    def _release(self, member_index: int) -> None:
        with self._route_lock:
            counter = self._slot_counters[member_index]
            counter.count = max(0, counter.count - 1)

    def _release_counter(self, counter: _SlotCounter) -> None:
        with self._route_lock:
            counter.count = max(0, counter.count - 1)

    def _select_and_acquire(self, prompt: str) -> tuple[int, typing.Any, _SlotCounter]:
        """Atomically select a live member, increment its counter, and record prefix affinity."""
        pinned, _depth = self.affinity.lookup(prompt)
        with self._route_lock:
            member_index, dead = self._pick_member_locked(pinned)
            if member_index is not None:
                member = self.members[member_index]
                counter = self._slot_counters[member_index]
                counter.count += 1
        for index in dead:
            self.affinity.drop_member(index)
        if member_index is None:
            raise RequestPlaneError("no live replica: every routed engine member is down")
        self.affinity.record(prompt, member_index)
        return member_index, member, counter

    def select_member(self, prompt: str) -> int:
        """Pick the member for one prompt: bounded prefix affinity, then JSQ.

        The prefix-pinned member wins while its router-local in-flight count
        is at most ``affinity_max_imbalance`` above the least-loaded live
        member's; beyond that the request joins the shortest queue (the
        caller then re-pins the prefix there). The queue key is (router-local
        in-flight, owner-reported pending prefill tokens, index): the local
        count is instantaneous, the remote signal covers load from other
        clients.

        Args:
            prompt: The incoming prompt text.

        Returns:
            Index into :attr:`members`.
        """
        pinned, _depth = self.affinity.lookup(prompt)
        with self._route_lock:
            member_index, dead = self._pick_member_locked(pinned)
        for index in dead:
            self.affinity.drop_member(index)
        if member_index is None:
            raise RequestPlaneError("no live replica: every routed engine member is down")
        return member_index

    @property
    def esurge_name(self) -> str:
        """The replica group's display name (first member's)."""
        return str(getattr(self.members[0], "esurge_name", "routed-esurge"))

    @property
    def max_model_len(self) -> int:
        """Group context limit (min across members)."""
        return min(int(member.max_model_len) for member in self.members)

    @property
    def max_num_seqs(self) -> int:
        """Aggregate concurrency limit: the sum across members, or the planned total while members attach late.

        A server sizes its generation slots from this once, at start-up; a router that starts with the first ready
        replica of ``planned_members`` reports the full deployment so late replicas are not capped away.
        """
        attached = sum(int(member.max_num_seqs) for member in self.members)
        if self.planned_members is None or len(self.members) >= self.planned_members:
            return attached
        return max(attached, max(int(member.max_num_seqs) for member in self.members) * self.planned_members)

    @property
    def tokenizer(self):
        """A tokenizer for the served model (first member's)."""
        return self.members[0].tokenizer

    @property
    def num_pending_requests(self) -> int:
        """Aggregate waiting-queue depth across live members."""
        with self._route_lock:
            members = list(self.members)
        return sum(int(getattr(member, "num_pending_requests", 0)) for member in members if _member_alive(member))

    @property
    def num_running_requests(self) -> int:
        """Aggregate running-set size across live members."""
        with self._route_lock:
            members = list(self.members)
        return sum(int(getattr(member, "num_running_requests", 0)) for member in members if _member_alive(member))

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | list[str] | None = None,
        use_tqdm: bool = False,
        tool_parser_request: typing.Any | None = None,
    ) -> list[RequestOutput]:
        """Route each prompt to its member and gather the finished outputs.

        Prompts of one call may land on different members; outputs return
        in prompt order.

        Args:
            prompts: One prompt or a batch.
            sampling_params: Generation parameters, shared by the batch.
            request_id: Optional explicit id(s), forwarded to the member.
            use_tqdm: Forwarded to local members.
            tool_parser_request: Forwarded to members that support it.

        Returns:
            One finished ``RequestOutput`` per prompt, in order.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        elif len(prompts) == 0:
            raise ValueError("Empty prompt list provided")
        if request_id is None:
            request_ids: list[str | None] = [None] * len(prompts)
        elif isinstance(request_id, str):
            if len(prompts) != 1:
                raise ValueError("request_id must be a list when providing multiple prompts.")
            request_ids = [request_id]
        else:
            request_ids = list(request_id)
            if len(request_ids) != len(prompts):
                raise ValueError("Length of request_id list must match number of prompts.")

        # Validate the explicit/auto id mix up front, on the caller's thread.
        # A mixed batch used to raise inside a daemon `_run` thread before its
        # try/finally, so the error was swallowed (caller saw `None` outputs)
        # and the acquired JSQ slots leaked. Fail cleanly before acquiring any.
        provided = [rid for rid in request_ids if rid is not None]
        if provided and len(provided) != len(request_ids):
            raise ValueError("Mix of explicit and auto request ids in one routed batch is unsupported")

        by_counter: dict[int, tuple[typing.Any, _SlotCounter, list[int]]] = {}
        acquired: list[_SlotCounter] = []
        try:
            for position, prompt in enumerate(prompts):
                _member_index, member, counter = self._select_and_acquire(prompt)
                acquired.append(counter)
                key = id(counter)
                if key not in by_counter:
                    by_counter[key] = (member, counter, [position])
                else:
                    by_counter[key][2].append(position)
        except BaseException:
            for counter in acquired:
                self._release_counter(counter)
            raise

        results: list[RequestOutput | None] = [None] * len(prompts)
        errors: list[Exception] = []
        threads: list[threading.Thread] = []
        groups = list(by_counter.values())

        def _run(member, counter: _SlotCounter, positions: list[int]) -> None:
            # The whole body runs under the release-protecting try/finally so
            # the acquired in-flight slots free on every exit path.
            try:
                member_prompts = [prompts[p] for p in positions]
                member_ids = [request_ids[p] for p in positions]
                explicit = [rid for rid in member_ids if rid is not None]
                kwargs: dict[str, typing.Any] = {}
                if explicit:
                    kwargs["request_id"] = explicit if len(explicit) > 1 else explicit[0]
                outputs = member.generate(
                    member_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=use_tqdm,
                    tool_parser_request=tool_parser_request,
                    **kwargs,
                )
                for position, output in zip(positions, outputs, strict=True):
                    results[position] = output
            except Exception as exc:  # surfaced after the join below
                errors.append(exc)
            finally:
                for _ in positions:
                    self._release_counter(counter)

        if len(groups) == 1:
            member, counter, positions = groups[0]
            _run(member, counter, positions)
        else:
            started_count = 0
            try:
                for member, counter, positions in groups:
                    thread = threading.Thread(target=_run, args=(member, counter, positions), daemon=True)
                    thread.start()
                    threads.append(thread)
                    started_count += 1
            except BaseException:
                for _member, counter, positions in groups[started_count:]:
                    for _ in positions:
                        self._release_counter(counter)
                for thread in threads:
                    thread.join()
                raise
            for thread in threads:
                thread.join()
        if errors:
            raise errors[0]
        return typing.cast("list[RequestOutput]", results)

    def stream(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        tool_parser_request: typing.Any | None = None,
    ) -> typing.Iterator[RequestOutput]:
        """Route one streaming request to its member and relay its updates."""
        if isinstance(prompts, list):
            if len(prompts) == 0:
                raise ValueError("Empty prompt list provided")
            prompt = prompts[0]
        else:
            prompt = prompts
        _member_index, member, counter = self._select_and_acquire(prompt)
        try:
            iterator = member.stream(
                prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                tool_parser_request=tool_parser_request,
            )
            yield from iterator
        finally:
            self._release_counter(counter)

    def _chat_routing_key(self, messages: list[dict]) -> str:
        """A stable prompt-shaped key for affinity routing of chat calls."""
        try:
            return str(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        except Exception:
            return repr(messages)

    def chat(self, messages: list[dict], *args, **kwargs):
        """Route a chat call to one member (affinity by templated prompt)."""
        _member_index, member, counter = self._select_and_acquire(self._chat_routing_key(messages))
        try:
            result = member.chat(messages, *args, **kwargs)
        except BaseException:
            self._release_counter(counter)
            raise
        if hasattr(result, "__next__"):
            # Streaming chat: the slot frees when the iterator ends, closes, or is GC'd.
            return _ReleasedIterator(result, lambda: self._release_counter(counter))
        self._release_counter(counter)
        return result

    @staticmethod
    def _stream_usage_kwargs(stream_fn, usage_callback) -> dict[str, typing.Any]:
        """Forward optional progress to supporting members, preserving legacy signatures."""
        if usage_callback is None:
            return {}
        try:
            parameters = inspect.signature(stream_fn).parameters
        except (TypeError, ValueError):
            return {}
        # As on the API server, generic **kwargs is not an advertisement:
        # a legacy adapter may pass it through to a member without this API.
        parameter = parameters.get("usage_callback")
        if parameter is not None and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return {"usage_callback": usage_callback}
        return {}

    def iter_chat_completion_stream(
        self,
        *,
        messages: list[dict],
        usage_callback: typing.Callable[[int, int], None] | None = None,
        **kwargs,
    ):
        """Route chat chunks and optional cumulative usage progress to one member."""
        _member_index, member, counter = self._select_and_acquire(self._chat_routing_key(messages))
        try:
            stream_fn = member.iter_chat_completion_stream
            yield from stream_fn(messages=messages, **kwargs, **self._stream_usage_kwargs(stream_fn, usage_callback))
        finally:
            self._release_counter(counter)

    def iter_responses_stream(
        self,
        *,
        messages: list[dict],
        usage_callback: typing.Callable[[int, int], None] | None = None,
        **kwargs,
    ):
        """Route Responses frames and optional cumulative usage progress to one member."""
        _member_index, member, counter = self._select_and_acquire(self._chat_routing_key(messages))
        try:
            stream_fn = member.iter_responses_stream
            yield from stream_fn(messages=messages, **kwargs, **self._stream_usage_kwargs(stream_fn, usage_callback))
        finally:
            self._release_counter(counter)

    def abort_request(self, request_id: str) -> None:
        """Cancel a request wherever it landed (no-op on non-owners)."""
        for member in self.members:
            try:
                member.abort_request(request_id)
            except Exception:
                logger.debug("Routed abort failed on one member", exc_info=True)

    def close(self) -> None:
        """Close remote members (local engines are left to their owners)."""
        for member in self.members:
            close = getattr(member, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    logger.debug("Routed member close failed", exc_info=True)
