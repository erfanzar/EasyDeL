# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the scheduled-MPMD tracer, dispatch watchdog and schedule grid renderer."""

from __future__ import annotations

import json
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from jax.sharding import Mesh
from spectrax import nn
from spectrax.runtime import (
    DualPipeV,
    GPipe,
    InterleavedH1,
    Std1F1B,
    ZeroBubbleH1,
    dispatch_watchdog,
    format_schedule_grid,
    sxjit,
    sxstage_iter,
    sxvalue_and_grad,
    trace_schedule,
)
from spectrax.runtime.mpmd import instrumentation as instr
from spectrax.runtime.mpmd.watchdog import DispatchWatchdog
from spectrax.runtime.schedules import Action, FusedTask, Phase
from spectrax.runtime.types import MpMdMesh


class _TwoLayer(spx.Module):
    """Two linear layers split across two pipeline stages."""

    def __init__(self, rngs):
        super().__init__()
        self.l1 = nn.Linear(8, 8, rngs=rngs)
        self.l2 = nn.Linear(8, 4, rngs=rngs)


def _mean_sq(model, x, y):
    return jnp.mean((model.l2(model.l1(x)) - y) ** 2)


@pytest.fixture
def scheduled_step():
    """Return ``(run, reference)`` for a 2-stage Std1F1B step with 4 microbatches."""
    if len(jax.devices()) < 2:
        pytest.skip("needs at least 2 devices")
    mesh = MpMdMesh(Mesh(jax.devices()[:2], ("pp",)), "pp")
    model = _TwoLayer(spx.Rngs(0))
    x = jnp.linspace(-1.0, 1.0, 64).reshape(8, 8)
    y = jnp.zeros((8, 4))

    @sxjit(mesh=mesh, schedule=Std1F1B(microbatches=4), static_argnums=(), batch_argnums=(1, 2))
    def loss_fn(model, x, y):
        h = sxstage_iter(model.l1(x))
        return jnp.mean((model.l2(h) - y) ** 2)

    def run():
        loss, _ = sxvalue_and_grad(loss_fn, argnums=(0,))(model, x, y)
        return loss

    return run, lambda: _mean_sq(model, x, y)


@pytest.mark.parametrize("block", [False, True], ids=["host", "device"])
def test_trace_records_every_unit_and_dependency(scheduled_step, tmp_path, block):
    run, reference = scheduled_step
    out = tmp_path / "trace.json"
    with trace_schedule(out, block_until_ready=block) as tracer:
        loss = run()
    np.testing.assert_allclose(float(loss), float(reference()), rtol=1e-6)
    assert not instr.active()

    (summary,) = tracer.summary()
    assert summary["dispatcher"] != "fused_serial_units"
    assert summary["units"] == summary["planned_units"] > 0
    assert summary["unit_errors"] == 0
    assert 0.0 <= summary["bubble_fraction"] <= 1.0
    assert summary["transfers"]["count"] > 0
    assert set(summary["per_rank_busy_ms"]) == {0, 1}

    indices = sorted(s.meta["unit_index"] for s in tracer.spans if s.kind == "unit")
    assert indices == list(range(summary["planned_units"]))

    trace = json.loads(out.read_text())
    dispatch = next(s for s in tracer.spans if s.kind == "dispatch")
    dep_edges = sum(len(v) for v in dispatch.meta["deps"].values())
    starts = [e for e in trace["traceEvents"] if e.get("ph") == "s"]
    finishes = [e for e in trace["traceEvents"] if e.get("ph") == "f"]
    assert trace["otherData"]["dependency_flows"] == dep_edges == len(starts) == len(finishes)
    assert {e["id"] for e in starts} == {e["id"] for e in finishes}


@pytest.mark.cpu_ok
def test_hooks_are_inert_without_observer():
    assert not instr.active()
    assert instr.begin("unit", "x") is None
    instr.end(None)
    instr.annotate("transfer", nbytes=1)
    assert instr.open_spans() == []


@pytest.mark.cpu_ok
def test_trace_is_written_when_body_raises(tmp_path):
    out = tmp_path / "fail.json"
    with pytest.raises(RuntimeError), trace_schedule(out):
        span = instr.begin("dispatch", "boom", units=[], deps={})
        instr.end(span, error=RuntimeError("boom"))
        raise RuntimeError("boom")
    (event,) = [e for e in json.loads(out.read_text())["traceEvents"] if e.get("cat") == "dispatch"]
    assert "boom" in event["args"]["error"]


def _count_actions(grid) -> dict[Phase, int]:
    counts: dict[Phase, int] = {}
    for row in grid:
        for cell in row:
            for action in (cell.fwd, cell.bwd) if isinstance(cell, FusedTask) else (cell,):
                if isinstance(action, Action):
                    counts[action.phase] = counts.get(action.phase, 0) + 1
    return counts


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "schedule,n",
    [
        (GPipe(microbatches=4), 4),
        (Std1F1B(microbatches=8), 4),
        (ZeroBubbleH1(microbatches=4), 4),
        (InterleavedH1(microbatches=4, virtual_stages=2), 2),
        (DualPipeV(microbatches=8), 4),
    ],
    ids=["gpipe", "1f1b", "zb", "interleaved", "dualpipev"],
)
def test_grid_render_covers_every_action(schedule, n):
    text = format_schedule_grid(schedule, n)
    grid = schedule.build(n)
    rows = [ln for ln in text.splitlines() if ln.lstrip()[:1] == "t" and ln.lstrip()[1:2].isdigit()]
    assert len(rows) == len(grid)
    letters = {"F": Phase.FWD, "B": Phase.BWD, "I": Phase.BWD_I, "W": Phase.BWD_W}
    rendered: dict[Phase, int] = {}
    for row in rows:
        for cell in row.split()[1:]:
            for part in cell.split("|"):
                if part[:1] in letters:
                    rendered[letters[part[0]]] = rendered.get(letters[part[0]], 0) + 1
    assert rendered == _count_actions(grid)
    assert f"bubble_ratio={schedule.bubble_ratio(n):.3f}" in text


def _held_transfer(release: threading.Event, started: threading.Event) -> None:
    span = instr.begin("transfer", "transfer_bwd_stage1_to_stage0_mb0", rank=0, src_rank=1, dst_rank=0)
    started.set()
    release.wait(30)
    instr.end(span)


@pytest.mark.cpu_ok
def test_watchdog_reports_stalled_transfer_with_stack():
    reports = []
    release, started = threading.Event(), threading.Event()
    with dispatch_watchdog(stall_after_s=0.2, poll_interval_s=0.02, on_stall=reports.append, log=False):
        thread = threading.Thread(target=_held_transfer, args=(release, started), name="stuck-transfer")
        thread.start()
        assert started.wait(5)
        deadline = time.monotonic() + 10
        while not reports and time.monotonic() < deadline:
            time.sleep(0.02)
        release.set()
        thread.join(10)
    assert reports
    report = reports[0]
    assert report.reason == "span_age"
    (stalled,) = report.spans
    assert (stalled.kind, stalled.rank, stalled.thread_name) == ("transfer", 0, "stuck-transfer")
    assert stalled.stack is not None and "_held_transfer" in stalled.stack
    assert not instr.active()


@pytest.mark.cpu_ok
def test_watchdog_no_progress_rule_is_throttled():
    wd = DispatchWatchdog(stall_after_s=1.0, poll_interval_s=1.0, log=False, include_stacks=False)
    instr.attach(wd)
    try:
        span = instr.begin("dispatch", "Std1F1B m=4")
        now = time.perf_counter_ns()
        assert wd.check(now) is None
        report = wd.check(now + int(2e9))
        assert report is not None and report.reason == "no_progress"
        assert wd.check(now + int(2.1e9)) is None
        instr.end(span)
        assert wd.check(now + int(7e9)) is None
    finally:
        instr.detach(wd)


def test_watchdog_quiet_on_healthy_step(scheduled_step):
    run, _ = scheduled_step
    with dispatch_watchdog(stall_after_s=60, poll_interval_s=0.05, log=False) as wd:
        run()
        assert wd.snapshot() == []
    assert wd.reports == []
