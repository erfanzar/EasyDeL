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
the DP scheduler: **prefix affinity first, then join-shortest-queue**.
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


def _member_alive(member) -> bool:
    """Best-effort liveness for local engines and remote handles alike."""
    alive = getattr(member, "alive", None)
    if alive is not None:
        return bool(alive)
    return bool(getattr(member, "_generation_alive", True))


def _member_pending_prefill_tokens(member) -> int:
    """JSQ cost signal for a member (remote stats or a local scheduler read)."""
    remote = getattr(member, "pending_prefill_tokens", None)
    if remote is not None:
        return int(remote)
    scheduler = getattr(member, "scheduler", None)
    if scheduler is None:
        return 0
    try:
        total = 0
        for request in scheduler.running:
            total += max(0, int(request.num_prompt_tokens) - int(request.num_computed_tokens))
        for request in scheduler.waiting:
            total += int(request.num_prompt_tokens)
        return total
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
    """

    def __init__(self, members: list, *, affinity_block_chars: int | None = None) -> None:
        if not members:
            raise ValueError("RoutedEngine needs at least one member")
        self.members = list(members)
        if affinity_block_chars is None:
            page_size = next((int(getattr(m, "page_size", 0) or 0) for m in self.members), 0)
            affinity_block_chars = page_size * 4 if page_size > 0 else _DEFAULT_BLOCK_CHARS
        self.affinity = PrefixAffinityMap(block_chars=affinity_block_chars)
        self._route_lock = threading.Lock()
        # Router-local in-flight counts: the JSQ primary signal. Remote
        # QueueStats ride 1s heartbeats, so a burst arriving between beats
        # would otherwise dump entirely on the lowest-indexed member.
        self._inflight = [0] * len(self.members)

    # routing
    def _alive_indices(self) -> list[int]:
        alive = []
        for index, member in enumerate(self.members):
            if _member_alive(member):
                alive.append(index)
            else:
                self.affinity.drop_member(index)
        if not alive:
            raise RequestPlaneError("no live replica: every routed engine member is down")
        return alive

    def _acquire(self, member_index: int) -> None:
        with self._route_lock:
            self._inflight[member_index] += 1

    def _release(self, member_index: int) -> None:
        with self._route_lock:
            self._inflight[member_index] = max(0, self._inflight[member_index] - 1)

    def _released_iterator(self, iterator: typing.Iterator, member_index: int) -> typing.Iterator:
        """Wrap a streaming result so its in-flight slot frees when it ends."""
        try:
            yield from iterator
        finally:
            self._release(member_index)

    def select_member(self, prompt: str) -> int:
        """Pick the member for one prompt: prefix affinity, then JSQ.

        The queue key is (router-local in-flight, owner-reported pending
        prefill tokens, index): the local count is instantaneous, the
        remote signal covers load from other clients.

        Args:
            prompt: The incoming prompt text.

        Returns:
            Index into :attr:`members`.
        """
        alive = self._alive_indices()
        pinned, _depth = self.affinity.lookup(prompt)
        if pinned is not None and pinned in alive:
            return pinned
        with self._route_lock:
            inflight = list(self._inflight)
        return min(
            alive,
            key=lambda index: (inflight[index], _member_pending_prefill_tokens(self.members[index]), index),
        )

    # adapter surface
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
        """Aggregate concurrency limit (sum across members)."""
        return sum(int(member.max_num_seqs) for member in self.members)

    @property
    def tokenizer(self):
        """A tokenizer for the served model (first member's)."""
        return self.members[0].tokenizer

    @property
    def num_pending_requests(self) -> int:
        """Aggregate waiting-queue depth across live members."""
        return sum(int(getattr(member, "num_pending_requests", 0)) for member in self.members)

    @property
    def num_running_requests(self) -> int:
        """Aggregate running-set size across live members."""
        return sum(int(getattr(member, "num_running_requests", 0)) for member in self.members)

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
        if request_id is None:
            request_ids: list[str | None] = [None] * len(prompts)
        elif isinstance(request_id, str):
            request_ids = [request_id]
        else:
            request_ids = list(request_id)

        # Validate the explicit/auto id mix up front, on the caller's thread.
        # A mixed batch used to raise inside a daemon `_run` thread before its
        # try/finally, so the error was swallowed (caller saw `None` outputs)
        # and the acquired JSQ slots leaked. Fail cleanly before acquiring any.
        provided = [rid for rid in request_ids if rid is not None]
        if provided and len(provided) != len(request_ids):
            raise ValueError("Mix of explicit and auto request ids in one routed batch is unsupported")

        by_member: dict[int, list[int]] = {}
        for position, prompt in enumerate(prompts):
            member_index = self.select_member(prompt)
            by_member.setdefault(member_index, []).append(position)
            self.affinity.record(prompt, member_index)
            self._acquire(member_index)

        results: list[RequestOutput | None] = [None] * len(prompts)
        errors: list[Exception] = []
        threads: list[threading.Thread] = []

        def _run(member_index: int, positions: list[int]) -> None:
            # The whole body runs under the release-protecting try/finally so
            # the acquired in-flight slots free on every exit path.
            try:
                member = self.members[member_index]
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
                    self._release(member_index)

        for member_index, positions in by_member.items():
            if len(by_member) == 1:
                _run(member_index, positions)
            else:
                thread = threading.Thread(target=_run, args=(member_index, positions), daemon=True)
                thread.start()
                threads.append(thread)
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
        prompt = prompts[0] if isinstance(prompts, list) else prompts
        member_index = self.select_member(prompt)
        self.affinity.record(prompt, member_index)
        self._acquire(member_index)
        member = self.members[member_index]
        yield from self._released_iterator(
            member.stream(
                prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                tool_parser_request=tool_parser_request,
            ),
            member_index,
        )

    def _chat_routing_key(self, messages: list[dict]) -> str:
        """A stable prompt-shaped key for affinity routing of chat calls."""
        try:
            return str(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        except Exception:
            return repr(messages)

    def _route_for_key(self, routing_key: str) -> int:
        member_index = self.select_member(routing_key)
        self.affinity.record(routing_key, member_index)
        self._acquire(member_index)
        return member_index

    def chat(self, messages: list[dict], *args, **kwargs):
        """Route a chat call to one member (affinity by templated prompt)."""
        member_index = self._route_for_key(self._chat_routing_key(messages))
        try:
            result = self.members[member_index].chat(messages, *args, **kwargs)
        except Exception:
            self._release(member_index)
            raise
        if hasattr(result, "__next__"):
            # Streaming chat: the slot frees when the iterator ends.
            return self._released_iterator(result, member_index)
        self._release(member_index)
        return result

    def iter_chat_completion_stream(self, *, messages: list[dict], **kwargs):
        """Route an OpenAI chat-completion stream to one member."""
        member_index = self._route_for_key(self._chat_routing_key(messages))
        yield from self._released_iterator(
            self.members[member_index].iter_chat_completion_stream(messages=messages, **kwargs),
            member_index,
        )

    def iter_responses_stream(self, *, messages: list[dict], **kwargs):
        """Route an OpenAI Responses-API stream to one member."""
        member_index = self._route_for_key(self._chat_routing_key(messages))
        yield from self._released_iterator(
            self.members[member_index].iter_responses_stream(messages=messages, **kwargs),
            member_index,
        )

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
