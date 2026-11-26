# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A gang's chorus renders once, with its headcount, in the blocking wait."""

from __future__ import annotations

import io
import random
import typing as tp

import pytest

from eyvan.sdk.client.callables import _CallableProgressRenderer
from eyvan.sdk.job_receipt import JobTerminalOutput


class _OnePageClient:
    """Answer one canned log page, then nothing more."""

    def __init__(self, records: tuple[dict[str, tp.Any], ...]) -> None:
        self._records = records

    def workload_log_page(self, workload_id: str, *, after: str = "", limit: int = 200) -> dict[str, tp.Any]:
        if after:
            return {"records": [], "next_cursor": ""}
        return {"records": list(self._records), "next_cursor": "cursor-1"}


def _render(records: tuple[dict[str, tp.Any], ...]) -> list[str]:
    """Drain one canned page through the real renderer with a plain writer."""

    emitted: list[str] = []
    output = JobTerminalOutput(writer=emitted.append)
    renderer = _CallableProgressRenderer(client=_OnePageClient(records), output=output, queue_admission_recorder=None)
    renderer._drain_output("wl-1")
    return [line for line in emitted if line.strip()]


def _render_ansi(records: tuple[dict[str, tp.Any], ...]) -> str:
    """Drain one canned page into a recording terminal, escapes and all."""

    buffer = io.StringIO()
    output = JobTerminalOutput(
        stream=buffer,
        force_terminal=True,
        environment={"TERM": "xterm-256color", "COLUMNS": "200"},
    )
    renderer = _CallableProgressRenderer(client=_OnePageClient(records), output=output, queue_admission_recorder=None)
    renderer._drain_output("wl-1")
    return buffer.getvalue()


BANNER = "(20:21:39 EasyDeL) please install `torch` (cpu) if you want to use `AutoEasyDeLModel*.from_torch_pretrained`"


def test_four_ranks_saying_the_same_line_render_once_with_the_headcount() -> None:
    """Collapse identical worker lines into one line carrying the contributing worker count."""

    lines = _render(
        tuple(
            {"rank": rank, "pid": 8660 + rank, "host": f"10.130.0.{80 + rank}", "message": BANNER} for rank in range(4)
        )
    )

    assert lines == [f"(4x Worker) {BANNER}"]


def test_rank_specific_lines_keep_their_own_worker_attribution() -> None:
    """A line only one host said is still that host's line.

    Collapsing is for choruses; a rank-specific value ("rank 0 / 4", a local
    device count) must keep naming the worker that produced it.
    """

    lines = _render(
        (
            {"rank": 0, "pid": 100, "host": "10.0.0.1", "message": "8 local devices | 32 total | rank 0 / 4"},
            {"rank": 1, "pid": 101, "host": "10.0.0.2", "message": "8 local devices | 32 total | rank 1 / 4"},
        )
    )

    assert lines == [
        "(Worker pid=100, 10.0.0.1) 8 local devices | 32 total | rank 0 / 4",
        "(Worker pid=101, 10.0.0.2) 8 local devices | 32 total | rank 1 / 4",
    ]


def test_one_rank_repeating_itself_is_not_a_chorus() -> None:
    """A rank printing its own line twice is a repeat, and repeats are signal.

    A retry loop, a stall, a wedged step -- the second print is the fact.
    Collapsing it would hide exactly the failure a person is reading for.
    """

    message = "step 100 complete"
    lines = _render(
        (
            {"rank": 2, "pid": 102, "host": "10.0.0.3", "message": message},
            {"rank": 2, "pid": 102, "host": "10.0.0.3", "message": message},
        )
    )

    assert lines == [
        "(Worker pid=102, 10.0.0.3) step 100 complete",
        "(Worker pid=102, 10.0.0.3) step 100 complete",
    ]


def test_a_page_without_rank_data_never_collapses() -> None:
    """Distinct voices must be proven, not assumed.

    A page that reports no rank field defaults every line to one rank; a
    collapse there would merge lines that may come from anywhere.
    """

    lines = _render(
        (
            {"pid": 100, "host": "10.0.0.1", "message": BANNER},
            {"pid": 101, "host": "10.0.0.2", "message": BANNER},
        )
    )

    assert len(lines) == 2, "rank-less pages degrade to one line per record"


def test_a_straggler_keeps_its_own_order_and_the_chorus_stays_whole() -> None:
    """A rank saying something else first does not split everyone else's chorus.

    Preserve each rank's order without requiring identical lines to be adjacent:
    rank 3's earlier ``compiling`` line comes first, and the shared banner stays
    one line.
    """

    lines = _render(
        (
            {"rank": 0, "pid": 100, "host": "10.0.0.1", "message": BANNER},
            {"rank": 1, "pid": 101, "host": "10.0.0.2", "message": BANNER},
            {"rank": 3, "pid": 103, "host": "10.0.0.4", "message": "compiling"},
            {"rank": 2, "pid": 102, "host": "10.0.0.3", "message": BANNER},
            {"rank": 3, "pid": 103, "host": "10.0.0.4", "message": BANNER},
        )
    )

    assert lines == [
        "(Worker pid=103, 10.0.0.4) compiling",
        "(4x Worker) " + BANNER,
    ]


@pytest.mark.parametrize("ranks", [2, 16, 64])
@pytest.mark.parametrize("seed", range(5))
def test_interleaved_ranks_printing_the_same_program_render_one_line_per_print(ranks: int, seed: int) -> None:
    """The live shape: every rank prints ``start`` then its device list, all interleaved.

    The interleaving is random per seed and the width is a parameter, so this
    pins the rule rather than one page that happened to be observed.
    """

    devices = "[TpuDevice(id=0), TpuDevice(id=1)]"
    shuffled = random.Random(seed)
    pending = {rank: ["start", devices] for rank in range(ranks)}
    records = []
    while pending:
        rank = shuffled.choice(sorted(pending))
        records.append({"rank": rank, "pid": 1000 + rank, "host": f"10.0.0.{rank}", "message": pending[rank].pop(0)})
        if not pending[rank]:
            del pending[rank]

    assert _render(tuple(records)) == [f"({ranks}x Worker) start", f"({ranks}x Worker) {devices}"]


def test_ranks_that_disagree_on_order_never_have_their_own_lines_reordered() -> None:
    """Rank 0 says X then Y; rank 1 says Y then X. No merge may swap either rank's lines."""

    lines = _render(
        (
            {"rank": 0, "pid": 100, "host": "10.0.0.1", "message": "X"},
            {"rank": 1, "pid": 101, "host": "10.0.0.2", "message": "Y"},
            {"rank": 0, "pid": 100, "host": "10.0.0.1", "message": "Y"},
            {"rank": 1, "pid": 101, "host": "10.0.0.2", "message": "X"},
        )
    )

    rank0 = [line.split(") ", 1)[1] for line in lines if "pid=100" in line or "x Worker" in line]
    rank1 = [line.split(") ", 1)[1] for line in lines if "pid=101" in line or "x Worker" in line]
    assert rank0 == ["X", "Y"]
    assert rank1 == ["Y", "X"]
    # The shared Y still merges: it is next for both ranks once rank 0's X is out,
    # and merging it moves neither rank's lines. Only X has to split.
    assert lines == [
        "(Worker pid=100, 10.0.0.1) X",
        "(2x Worker) Y",
        "(Worker pid=101, 10.0.0.2) X",
    ]


def test_only_the_attribution_is_tinted_not_the_jobs_words() -> None:
    """The prefix is ours to colour; the line after it is the job's.

    ``Text(prefix, style="cyan").append(line)`` tints the whole construct,
    because Text's own style argument covers everything appended after it --
    so every worker line read as if the job had printed it in Eyvan's cyan.
    The tint has to ride on a span that ends where the prefix does.
    """

    rendered = _render_ansi(
        (
            {
                "rank": 1,
                "pid": 100,
                "host": "10.0.0.1",
                "message": "8 local devices | 32 total | rank 1 / 4",
            },
        )
    )

    assert "\x1b[36m(Worker pid=100, 10.0.0.1) \x1b[0m8 local devices | 32 total | rank 1 / 4" in rendered
    # The bug's shape, pinned so it cannot return unnoticed: one tint that
    # closes only at the end of the line has swallowed the job's words.
    assert "\x1b[36m(Worker pid=100, 10.0.0.1) 8 local devices" not in rendered


def test_the_chorus_headcount_tints_alone() -> None:
    """The headcount is Eyvan's summary; the line it counts is still the job's."""

    rendered = _render_ansi(
        (
            {"rank": 0, "pid": 100, "host": "10.0.0.1", "message": BANNER},
            {"rank": 1, "pid": 101, "host": "10.0.0.2", "message": BANNER},
        )
    )

    assert f"\x1b[36m(2x Worker) \x1b[0m{BANNER}" in rendered
    assert f"\x1b[36m(2x Worker) {BANNER}" not in rendered


def test_the_colours_the_job_printed_survive_the_trip() -> None:
    """A script that colours its own output is not decoration Eyvan may drop.

    The bytes came through journald like any other stdout; reading them back
    as styles is what keeps a red error red on the way to the terminal, while
    the redirected path still gets clean text rather than raw escapes.
    """

    rendered = _render_ansi(
        (
            {"rank": 0, "pid": 100, "host": "10.0.0.1", "message": "\x1b[31mred\x1b[0m result"},
            {"rank": 1, "pid": 101, "host": "10.0.0.2", "message": "\x1b[31mred\x1b[0m result"},
        )
    )

    assert "\x1b[36m(2x Worker) \x1b[0m\x1b[31mred\x1b[0m result" in rendered

    redirected = _render(({"rank": 0, "pid": 100, "host": "10.0.0.1", "message": "\x1b[31mred\x1b[0m result"},))
    assert redirected == ["(Worker pid=100, 10.0.0.1) red result"]
