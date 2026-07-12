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

"""Lightweight (no-engine) unit tests for :class:`RoutedEngine`.

These duck-type the member surface so the router's request-lifecycle
bookkeeping can be checked on CPU without building real eSurge replicas:
in-flight JSQ slots must be acquired and released on every exit path, and a
mixed explicit/auto id batch must fail cleanly on the caller's thread
instead of leaking acquired slots and swallowing the error on a daemon
thread (returning ``None`` outputs to the caller).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed.routed_engine import RoutedEngine


class _FakeMember:
    """Minimal engine-shaped member: records generate calls, returns stubs."""

    def __init__(self, *, name: str = "fake", generate_error: Exception | None = None) -> None:
        self.esurge_name = name
        self.max_model_len = 128
        self.max_num_seqs = 4
        self.page_size = 16
        self._generation_alive = True
        self.calls: list[tuple[list[str], dict]] = []
        self._generate_error = generate_error

    def generate(self, prompts, sampling_params=None, use_tqdm=False, tool_parser_request=None, **kwargs):
        self.calls.append((list(prompts), dict(kwargs)))
        if self._generate_error is not None:
            raise self._generate_error
        return [SimpleNamespace(prompt=p) for p in prompts]


def test_routed_generate_rejects_mixed_ids_without_leaking_slots():
    # Two fresh members: the batch would fan prompt 0 -> member 0 and
    # prompt 1 -> member 1 (JSQ by index), so the OLD per-member check never
    # fires (each member sees a homogeneous slice) and the mix used to slip
    # through silently. The up-front validation must reject it regardless of
    # routing, on the caller's thread, before any slot is acquired.
    member_a, member_b = _FakeMember(name="a"), _FakeMember(name="b")
    router = RoutedEngine([member_a, member_b])

    with pytest.raises(ValueError, match="Mix of explicit and auto"):
        router.generate(["p0", "p1"], request_id=["explicit-0", None])

    # Validation ran before any acquire/dispatch, so nothing leaked...
    assert router._inflight == [0, 0]
    assert member_a.calls == [] and member_b.calls == [], "no member.generate should fire for the invalid batch"

    # ...and a subsequent clean batch still works and rebalances to zero.
    out = router.generate(["p0", "p1"])
    assert len(out) == 2
    assert router._inflight == [0, 0]


def test_routed_generate_all_explicit_ids_is_accepted():
    member = _FakeMember()
    router = RoutedEngine([member])
    out = router.generate(["p0", "p1"], request_id=["id-0", "id-1"])
    assert len(out) == 2
    # Both ids forwarded to the (single) member; slot released afterwards.
    assert member.calls[0][1]["request_id"] == ["id-0", "id-1"]
    assert router._inflight == [0]


def test_routed_generate_surfaces_member_errors_and_releases_slots():
    member = _FakeMember(generate_error=RuntimeError("member exploded"))
    router = RoutedEngine([member])

    with pytest.raises(RuntimeError, match="member exploded"):
        router.generate(["p0"])

    assert router._inflight == [0], "the in-flight slot must release even when the member raises"
