# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for MPMD validation and legacy compiler helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.runtime.mpmd.compiler import compile_ranked_executables, run_ranked_pipeline
from spectrax.runtime.mpmd.pscan_compiler import PscanPlan, _pack_grad_tree
from spectrax.runtime.mpmd.runtime import (
    _build_schedule_unit_dependencies,
    _build_schedule_units_from_plan,
    _dependency_topological_schedule_units,
    _infer_schedule_static_argnums,
    _normalize_argnums,
    _resolve_explicit_shardings,
    sxcall,
)
from spectrax.runtime.schedules import (
    Action,
    DualPipeV,
    Eager1F1B,
    FusedTask,
    GPipe,
    Interleaved1F1BPlusOne,
    InterleavedGPipe,
    InterleavedH1,
    KimiK2,
    Phase,
    Schedule,
    Std1F1B,
    ZeroBubbleH1,
)


@dataclass
class _FwdOnly(Schedule):
    """Tiny forward-only schedule for compiler shape tests."""

    def build(self, n_stages: int):
        """Build helper."""
        return [[Action(Phase.FWD, microbatch=mb)] for mb in range(self.microbatches)]

    def peak_activations(self, n_stages: int) -> int:
        """Peak activation helper."""
        return self.microbatches


def _runtime_source_ast() -> ast.Module:
    """Parse the local MPMD runtime source without importing private helpers."""
    root = Path(__file__).resolve().parents[2]
    return ast.parse((root / "spectrax/runtime/mpmd/runtime.py").read_text())


def _find_function(root: ast.AST, name: str) -> ast.FunctionDef:
    """Find a function definition by name in an AST tree."""
    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} was not found")


def _default_name_bindings(fn: ast.FunctionDef) -> dict[str, str]:
    """Return ``argument_name -> default_name`` for simple name defaults."""
    args = fn.args.posonlyargs + fn.args.args
    defaults = fn.args.defaults
    defaulted_args = args[len(args) - len(defaults) :]
    return {
        arg.arg: default.id
        for arg, default in zip(defaulted_args, defaults, strict=True)
        if isinstance(default, ast.Name)
    }


def _loaded_names_in_body(fn: ast.FunctionDef) -> set[str]:
    """Collect names loaded by a function body, excluding argument defaults."""
    names: set[str] = set()
    for statement in fn.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
    return names


def test_routed_transport_closures_bind_current_hop_values():
    """Async routed transfers must not late-bind loop values from a later hop/source."""
    module = _runtime_source_ast()
    routed = _find_function(module, "_routed_stage_edge_transport")
    move_one_hop = _find_function(routed, "move_one_hop")

    assert {
        "transfer_value": "current",
        "transfer_target": "target",
        "transfer_task_name": "hop_task_name",
        "transfer_src_rank": "current_rank",
        "transfer_dst_rank": "hop_rank",
    }.items() <= _default_name_bindings(move_one_hop).items()
    assert not ({"current", "target", "hop_task_name", "current_rank", "hop_rank"} & _loaded_names_in_body(move_one_hop))

    collect_fwd_invars = _find_function(module, "_collect_fwd_invars")
    do_transfer = next(
        fn
        for fn in ast.walk(collect_fwd_invars)
        if isinstance(fn, ast.FunctionDef)
        and fn.name == "do_transfer"
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_routed_stage_edge_transport"
            for call in ast.walk(fn)
        )
    )
    assert _default_name_bindings(do_transfer)["producer_key_for_transfer"] == "producer_key"
    assert "producer_key" not in _loaded_names_in_body(do_transfer)


def test_infer_schedule_static_argnums_keeps_plain_arrays_dynamic():
    """Array batches stay dynamic even when no Module argument is present."""
    args = (jnp.ones((2, 3)), {"x": jnp.ones((1,))}, "metadata")

    assert _infer_schedule_static_argnums(args) == (2,)


def test_normalize_argnums_accepts_jax_style_negative_indices():
    """``sxgrad`` validation should match JAX's negative argnum convention."""
    assert _normalize_argnums((-1, 0), 2) == (1, 0)


def test_resolve_explicit_shardings_preserves_none_placeholders():
    """``None`` entries in ``in_shardings`` still consume positional slots."""
    sharding = object()

    assert _resolve_explicit_shardings((None, sharding, None), [1, 2, 3]) == {1: sharding}


def test_ranked_compiler_forward_outputs_can_change_shape():
    """Legacy rank programs must not assume stage output shape equals input shape."""

    def expand(x):
        """Expand helper."""
        return jnp.stack([x, x + 1], axis=0)

    cluster = jax.make_jaxpr(expand)(jnp.asarray(1.0))
    program = compile_ranked_executables([cluster], _FwdOnly(microbatches=2), n_stages=1)[0]

    _grads, outgoing_acts, _cots, _loss = program([()], jnp.asarray([1.0, 2.0]), jnp.zeros((2,)))

    assert outgoing_acts.shape == (2, 2)
    assert jnp.allclose(outgoing_acts, jnp.asarray([[1.0, 2.0], [2.0, 3.0]]))


def test_ranked_compiler_bwd_uses_incoming_cotangent_for_param_grads():
    """BWD must seed param gradients with ``g_y``, not JAX's implicit scalar one."""

    def stage(w, x):
        """Stage helper."""
        return w * x

    cluster = jax.make_jaxpr(stage)(jnp.asarray(2.0), jnp.asarray(3.0))
    program = compile_ranked_executables([cluster], GPipe(microbatches=1), n_stages=1)[0]

    grads, _outgoing_acts, outgoing_cots, _loss = program(
        [(jnp.asarray(2.0),)],
        jnp.asarray([3.0]),
        jnp.asarray([7.0]),
    )

    assert jnp.allclose(grads[0][0], 21.0)
    assert jnp.allclose(outgoing_cots[0], 14.0)


def test_run_ranked_pipeline_returns_mean_loss_and_mean_grads():
    """Legacy ranked pipeline helper should be numerically correct if used."""

    def stage(w, x):
        """Stage helper."""
        return w * x

    def loss_fn(y, target):
        """Compute the loss."""
        diff = y - target
        return 0.5 * jnp.sum(diff * diff)

    w0 = jnp.asarray(2.0)
    w1 = jnp.asarray(3.0)
    m = 4
    xs = jnp.arange(m * 2, dtype=jnp.float32).reshape(m, 2) + 1.0
    target = jnp.ones((m, 2), dtype=jnp.float32)
    cluster = jax.make_jaxpr(stage)(jnp.asarray(1.0), jnp.ones((2,), dtype=jnp.float32))

    loss, grads = run_ranked_pipeline(
        [cluster, cluster],
        [(w0,), (w1,)],
        GPipe(microbatches=m),
        n_stages=2,
        microbatches=m,
        xs=xs,
        target_args=(target,),
        loss_fn=loss_fn,
    )

    y0 = xs * w0
    y1 = y0 * w1
    diff = y1 - target
    ref_loss = 0.5 * jnp.sum(diff * diff) / m
    ref_g1 = (y0 * diff).sum() / m
    ref_g0 = (xs * diff * w1).sum() / m

    assert jnp.allclose(loss, ref_loss)
    assert jnp.allclose(grads[0][0], ref_g0)
    assert jnp.allclose(grads[1][0], ref_g1)


def test_pack_grad_tree_zeros_missing_const_grads():
    """No-producing-rank cases should return zeros, not index into an empty tuple."""
    plan = PscanPlan(
        n=1,
        v=1,
        n_logical=1,
        m=1,
        schedule=_FwdOnly(microbatches=1),
        ops=(),
        n_outs=0,
        n_outer_consts=1,
        body_mode="train",
        stage_shardings=[],
        rank_submeshes=[],
        mpmd_mesh=None,
        loc_for_logical=((0, 0),),
        logical_for_loc={(0, 0): 0},
        terminal_loc=(0, 0),
        per_loc_consts={},
        const_indices_per_loc={},
        n_invars_per_loc={},
        fwd_jits={},
        bwd_jits={},
        terminal_jit=lambda *args: args,
        init_state_template=[],
        grad_tree=jax.tree.structure({"w": jnp.ones((2,), dtype=jnp.float32)}),
        grad_const_indices=(0,),
        grad_template_leaves=(jnp.ones((2,), dtype=jnp.float32),),
        grad_output_sharding=jax.devices()[0],
    )

    out = _pack_grad_tree(plan, None)

    assert jnp.array_equal(out["w"], jnp.zeros((2,), dtype=jnp.float32))


def test_sxcall_rejects_invalid_mode_before_setup():
    """Unknown modes must not silently enter the train path."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",), mpmd_axis="pp")

    with pytest.raises(ValueError, match="mode"):
        sxcall(object(), (jnp.ones((1,)),), mesh=mesh, schedule=_FwdOnly(microbatches=1), mode="bogus")


def test_sxgrad_argnums_validation_happens_at_call_time():
    """Out-of-range ``argnums`` should raise a friendly ``ValueError``."""

    def plain(x):
        """Plain reference implementation."""
        return x.sum()

    plain._mpmd_state = {"schedule_requested": True}

    from spectrax.runtime.mpmd.runtime import sxgrad

    with pytest.raises(ValueError, match="argnum"):
        sxgrad(plain, argnums=1)(jnp.ones((2,)))


def test_sxvalue_and_grad_argnums_validation_happens_at_call_time():
    """``sxvalue_and_grad`` uses the same friendly bounds validation."""

    def plain(x):
        """Plain reference implementation."""
        return x.sum()

    plain._mpmd_state = {"schedule_requested": True}

    from spectrax.runtime.mpmd.runtime import sxvalue_and_grad

    with pytest.raises(ValueError, match="argnum"):
        sxvalue_and_grad(plain, argnums=2)(jnp.ones((2,)))


def test_dualpipev_build_units_preserves_mixed_fused_logicals():
    """Mixed DualPipeV FWD+BWD cells stay fused and route each half to its own logical stage."""
    grid = DualPipeV(microbatches=4).build(n_stages=2)
    plan = {
        "logical_for_loc": {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3},
        "n_logical": 4,
        "schedule_n_logical": 4,
        "terminal_logical": 3,
        "grid": grid,
    }

    units = _build_schedule_units_from_plan(plan)
    fused_units = [unit for unit in units if unit.kind == "fused"]

    assert fused_units
    assert any(
        isinstance(unit.payload, FusedTask)
        and unit.fwd_logical != unit.bwd_logical
        and unit.payload.fwd.phase is Phase.FWD
        and unit.payload.bwd.phase is Phase.BWD
        for unit in fused_units
    )
    assert not any(unit.bwd_logical == 3 for unit in units)


@pytest.mark.parametrize(
    ("schedule", "n_stages"),
    [
        (GPipe(microbatches=8), 4),
        (Std1F1B(microbatches=8), 4),
        (Eager1F1B(microbatches=8), 4),
        (ZeroBubbleH1(microbatches=8), 4),
        (InterleavedH1(microbatches=8, virtual_stages=2), 4),
        (Interleaved1F1BPlusOne(microbatches=8, virtual_stages=2), 4),
        (InterleavedGPipe(microbatches=8, virtual_stages=2), 4),
        (KimiK2(microbatches=8, virtual_stages=2, extra_warmup=1), 4),
        (DualPipeV(microbatches=8), 4),
    ],
)
def test_schedule_unit_lowering_preserves_scheduler_work(schedule, n_stages):
    """Runtime unit lowering must keep each scheduler's non-terminal work assigned correctly."""
    grid = schedule.build(n_stages)
    v = schedule.virtual_stages_per_rank()
    logical_for_loc = {
        (rank, virt): schedule.logical_at(rank, virt, n_stages) for rank in range(n_stages) for virt in range(v)
    }
    n_logical = n_stages * v
    terminal_rank, terminal_virt = schedule.terminal_loc(n_stages)
    terminal_logical = schedule.logical_at(terminal_rank, terminal_virt, n_stages)
    plan = {
        "logical_for_loc": logical_for_loc,
        "n_logical": n_logical,
        "schedule_n_logical": n_logical,
        "terminal_logical": terminal_logical,
        "grid": grid,
    }

    expected: dict[tuple[int, Phase, int], int] = {}
    for row in grid:
        for rank, cell in enumerate(row):
            actions = cell.split() if isinstance(cell, FusedTask) else (() if cell is None else (cell,))
            for action in actions:
                logical = logical_for_loc[(rank, action.virtual_stage)]
                if logical == terminal_logical and action.phase in (Phase.BWD, Phase.BWD_I, Phase.BWD_W):
                    continue
                key = (logical, action.phase, action.microbatch)
                expected[key] = expected.get(key, 0) + 1

    observed: dict[tuple[int, Phase, int], int] = {}
    for unit in _build_schedule_units_from_plan(plan):
        if unit.fwd_logical is not None:
            key = (unit.fwd_logical, Phase.FWD, unit.fwd_mb)
            observed[key] = observed.get(key, 0) + 1
        if unit.bwd_logical is not None:
            key = (unit.bwd_logical, unit.bwd_phase, unit.bwd_mb)
            observed[key] = observed.get(key, 0) + 1

    assert observed == expected


@pytest.mark.parametrize(
    ("schedule", "n_stages"),
    [
        (GPipe(microbatches=8), 4),
        (Std1F1B(microbatches=8), 4),
        (Eager1F1B(microbatches=8), 4),
        (ZeroBubbleH1(microbatches=8), 4),
        (InterleavedH1(microbatches=8, virtual_stages=2), 4),
        (Interleaved1F1BPlusOne(microbatches=8, virtual_stages=2), 4),
        (InterleavedGPipe(microbatches=8, virtual_stages=2), 4),
        (KimiK2(microbatches=8, virtual_stages=2, extra_warmup=1), 4),
        (DualPipeV(microbatches=8), 4),
        (DualPipeV(microbatches=8, zero_bubble=False), 4),
    ],
)
def test_schedule_unit_dependencies_make_scheduler_work_executable(schedule, n_stages):
    """The async runtime DAG must make every scheduler's grid dependency-legal."""
    grid = schedule.build(n_stages)
    v = schedule.virtual_stages_per_rank()
    logical_for_loc = {
        (rank, virt): schedule.logical_at(rank, virt, n_stages) for rank in range(n_stages) for virt in range(v)
    }
    n_logical = n_stages * v
    terminal_rank, terminal_virt = schedule.terminal_loc(n_stages)
    terminal_logical = schedule.logical_at(terminal_rank, terminal_virt, n_stages)
    plan = {
        "logical_for_loc": logical_for_loc,
        "n_logical": n_logical,
        "schedule_n_logical": n_logical,
        "terminal_logical": terminal_logical,
        "grid": grid,
        "m": schedule.microbatches,
        "invar_sources": [() if logical == 0 else (("cluster_out", logical - 1, 0),) for logical in range(n_logical)],
    }

    units = _build_schedule_units_from_plan(plan)
    deps = _build_schedule_unit_dependencies(plan, units)
    ordered = _dependency_topological_schedule_units(units, deps)
    order = {unit.index: pos for pos, unit in enumerate(ordered)}
    fwd_pos: dict[tuple[int, int], int] = {}
    bwd_cot_pos: dict[tuple[int, int], int] = {}
    bwd_w_pos: dict[tuple[int, int], int] = {}

    for unit in ordered:
        pos = order[unit.index]
        if unit.fwd_logical is not None and unit.fwd_mb is not None:
            fwd_pos[(unit.fwd_logical, unit.fwd_mb)] = pos
            if unit.fwd_logical == terminal_logical:
                bwd_cot_pos[(unit.fwd_logical, unit.fwd_mb)] = pos
        if unit.bwd_logical is not None and unit.bwd_mb is not None:
            key = (unit.bwd_logical, unit.bwd_mb)
            if unit.bwd_phase is Phase.BWD_W:
                bwd_w_pos[key] = pos
            else:
                bwd_cot_pos[key] = pos

    for logical in range(n_logical):
        for mb in range(schedule.microbatches):
            if logical > 0 and (logical, mb) in fwd_pos:
                assert fwd_pos[(logical, mb)] > fwd_pos[(logical - 1, mb)]
            if logical != terminal_logical and (logical, mb) in bwd_cot_pos:
                assert bwd_cot_pos[(logical, mb)] > fwd_pos[(logical, mb)]
                if logical + 1 < n_logical:
                    assert bwd_cot_pos[(logical, mb)] > bwd_cot_pos[(logical + 1, mb)]
            if (logical, mb) in bwd_w_pos:
                assert bwd_w_pos[(logical, mb)] > bwd_cot_pos[(logical, mb)]


@pytest.mark.parametrize(
    ("schedule", "n_stages", "expects_fwd_bwd_overlap"),
    [
        (GPipe(microbatches=8), 4, False),
        (Std1F1B(microbatches=8), 4, True),
        (Eager1F1B(microbatches=8), 4, True),
        (ZeroBubbleH1(microbatches=8), 4, True),
        (InterleavedH1(microbatches=8, virtual_stages=2), 4, True),
        (Interleaved1F1BPlusOne(microbatches=8, virtual_stages=2), 4, True),
        (InterleavedGPipe(microbatches=8, virtual_stages=2), 4, False),
        (KimiK2(microbatches=8, virtual_stages=2, extra_warmup=1), 4, True),
        (DualPipeV(microbatches=8), 4, True),
    ],
)
def test_schedulers_expose_parallel_runtime_work(schedule, n_stages, expects_fwd_bwd_overlap):
    """Every scheduler must expose multi-rank work; 1F1B-like schedulers also overlap FWD/BWD-family phases."""
    grid = schedule.build(n_stages)
    v = schedule.virtual_stages_per_rank()
    logical_for_loc = {
        (rank, virt): schedule.logical_at(rank, virt, n_stages) for rank in range(n_stages) for virt in range(v)
    }
    n_logical = n_stages * v
    terminal_rank, terminal_virt = schedule.terminal_loc(n_stages)
    terminal_logical = schedule.logical_at(terminal_rank, terminal_virt, n_stages)
    plan = {
        "logical_for_loc": logical_for_loc,
        "n_logical": n_logical,
        "schedule_n_logical": n_logical,
        "terminal_logical": terminal_logical,
        "grid": grid,
        "m": schedule.microbatches,
        "invar_sources": [() if logical == 0 else (("cluster_out", logical - 1, 0),) for logical in range(n_logical)],
    }

    def actions(cell):
        if cell is None:
            return ()
        return cell.split() if isinstance(cell, FusedTask) else (cell,)

    row_widths = [sum(cell is not None for cell in row) for row in grid]
    fwd_bwd_rows = 0
    for row in grid:
        phases = {action.phase for cell in row for action in actions(cell)}
        if Phase.FWD in phases and phases.intersection({Phase.BWD, Phase.BWD_I, Phase.BWD_W}):
            fwd_bwd_rows += 1

    units = _build_schedule_units_from_plan(plan)
    deps = _build_schedule_unit_dependencies(plan, units)
    dependents: dict[int, set[int]] = {unit.index: set() for unit in units}
    remaining = {idx: set(unit_deps) for idx, unit_deps in deps.items()}
    unit_by_index = {unit.index: unit for unit in units}
    for idx, unit_deps in deps.items():
        for dep in unit_deps:
            dependents.setdefault(dep, set()).add(idx)
    ready = sorted(
        (idx for idx, unit_deps in remaining.items() if not unit_deps), key=lambda idx: (unit_by_index[idx].row, idx)
    )
    max_runtime_width = 0
    emitted: set[int] = set()
    while ready:
        wave: list[int] = []
        deferred: list[int] = []
        used_ranks: set[int] = set()
        for idx in ready:
            rank = unit_by_index[idx].rank
            if rank in used_ranks:
                deferred.append(idx)
                continue
            wave.append(idx)
            used_ranks.add(rank)
        assert wave
        max_runtime_width = max(max_runtime_width, len(wave))
        for idx in wave:
            emitted.add(idx)
            for dependent in dependents.get(idx, ()):
                remaining[dependent].discard(idx)
                if not remaining[dependent] and dependent not in emitted and dependent not in deferred:
                    deferred.append(dependent)
        ready = sorted(deferred, key=lambda idx: (unit_by_index[idx].row, idx))

    assert max(row_widths) > 1
    assert sum(width > 1 for width in row_widths) > 0
    assert max_runtime_width > 1
    assert (fwd_bwd_rows > 0) is expects_fwd_bwd_overlap
