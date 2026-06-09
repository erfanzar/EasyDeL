# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz-style ABI regressions for SpectraX MPMD stage boundaries.

These tests intentionally check structural contracts instead of only numerical
equality. The eSurge KV-copy regression was caused by a valid-looking program
whose stage output tuple was permuted: values still existed, but the executable
ABI no longer matched donation/carry slot order. The cases below stress that
class of bug with many small, synthetic pytrees so the suite stays fast enough
to run locally.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.core import Jaxpr, Var
from jax.sharding import PartitionSpec, SingleDeviceSharding

import spectrax as spx
from spectrax.runtime.mpmd import MpmdPipelineExecutor, cluster_jaxpr_by_markers, sxstage_iter
from spectrax.runtime.mpmd.markers import (
    has_stage_regions,
    marker_edge_shardings,
    stage_region_cluster_boundaries,
    stage_region_specs,
)
from spectrax.runtime.mpmd.runtime import (
    _assemble_invars_from_plan,
    _build_outvar_map,
    _marker_alias_resolver,
    _prepare_invar_assembly_plan,
)

ABI_CASES = tuple(range(20))


class _FakeSubmesh:
    """Small mesh-like context object for executor-only tests."""

    def __init__(self) -> None:
        self.devices = np.asarray(jax.devices()[:1], dtype=object)

    def __enter__(self) -> "_FakeSubmesh":
        return self

    def __exit__(self, exc_type: tp.Any, exc: BaseException | None, tb: tp.Any) -> bool:
        del exc_type, exc, tb
        return False


def _float_args(count: int, *, width: int = 3) -> tuple[jax.Array, ...]:
    """Return distinct same-shaped float arrays for jaxpr fuzz cases."""
    return tuple(jnp.full((width,), float(i + 1), dtype=jnp.float32) for i in range(count))


def _permutation(count: int, case: int) -> list[int]:
    """Return a deterministic nontrivial permutation for ``case``."""
    order = list(range(count))
    shift = case % max(1, count)
    order = order[shift:] + order[:shift]
    if case % 2:
        order = list(reversed(order))
    if case % 3 == 1:
        order = order[::2] + order[1::2]
    if case % 5 == 2:
        order = order[1:] + order[:1]
    return order


def _defined_vars_before_first_stage_marker(jaxpr: Jaxpr) -> list[Var]:
    """Collect vars written before the first ``sxstage_iter`` in definition order."""
    defined: list[Var] = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "sxstage_iter":
            break
        defined.extend(outvar for outvar in eqn.outvars if isinstance(outvar, Var))
    return defined


def _array_scalar(value: tp.Any) -> int:
    """Convert a rank-1 test array to an integer scalar."""
    return int(np.asarray(value).reshape(-1)[0])


@pytest.mark.parametrize("case", ABI_CASES)
def test_cluster_outvars_keep_definition_order_for_many_live_carries(case: int) -> None:
    """Stage outputs that feed later stages must follow producer definition order."""
    carry_count = 8 + (case % 5)
    args = _float_args(carry_count + 1)

    def model(*xs):
        """Build many live pre-marker carry leaves and consume them after the cut."""
        carries = []
        for idx, leaf in enumerate(xs[:-1]):
            value = leaf + jnp.asarray(idx + 1, dtype=leaf.dtype)
            if (idx + case) % 3 == 0:
                value = value * jnp.asarray(1.125, dtype=leaf.dtype)
            elif (idx + case) % 3 == 1:
                value = jnp.sin(value) + value
            else:
                value = jnp.cos(value) - value
            carries.append(value)
        marker_input = xs[-1] * 2.0 + carries[case % carry_count] * 0.0
        h = sxstage_iter(marker_input, stage=0)
        order = _permutation(carry_count, case)
        return (*(carries[idx] + h * 0.0 for idx in order), h)

    jaxpr = jax.make_jaxpr(model)(*args).jaxpr
    clusters = cluster_jaxpr_by_markers(jaxpr)

    stage0_out_ids = {id(outvar) for outvar in clusters[0].outvars}
    expected = [var for var in _defined_vars_before_first_stage_marker(jaxpr) if id(var) in stage0_out_ids]
    actual = clusters[0].outvars[: len(expected)]

    assert [id(var) for var in actual] == [id(var) for var in expected]


@pytest.mark.parametrize("case", ABI_CASES)
def test_outvar_map_preserves_user_return_leaf_order(case: int) -> None:
    """The final output map must point to return leaves in user-visible order."""
    args = _float_args(9)

    def model(*xs):
        """Return a shuffled mix of original, pre-stage, and post-stage leaves."""
        pre = []
        for idx, leaf in enumerate(xs[:5]):
            value = leaf * jnp.asarray(idx + 2, dtype=leaf.dtype)
            pre.append(jnp.tanh(value) if (idx + case) % 2 else value + 0.25)
        h = sxstage_iter(xs[-1] + pre[0] * 0.0, stage=0)
        post = [jnp.sin(h + pre[idx] * 0.01 + idx) for idx in range(len(pre))]

        leaves = [*pre, *post, xs[case % len(xs)]]
        order = _permutation(len(leaves), case)[:8]
        return tuple(leaves[idx] for idx in order)

    closed = jax.make_jaxpr(model)(*args)
    jaxpr = closed.jaxpr
    clusters = cluster_jaxpr_by_markers(jaxpr)
    original_id_to_idx = {id(var): idx for idx, var in enumerate(jaxpr.invars)}
    fn_outvar_map = _build_outvar_map(closed, clusters, original_id_to_idx)
    resolve_alias = _marker_alias_resolver(jaxpr)

    mapped_vars: list[Var] = []
    for mapping in fn_outvar_map:
        owner = mapping[0]
        if isinstance(owner, int):
            mapped_vars.append(resolve_alias(clusters[int(owner)].outvars[int(mapping[1])]))
        elif owner == "orig_passthrough":
            mapped_vars.append(resolve_alias(jaxpr.invars[int(mapping[1])]))
        else:
            raise AssertionError(f"unexpected output mapping in fuzz case {case}: {mapping!r}")

    assert [id(resolve_alias(var)) for var in jaxpr.outvars] == [id(var) for var in mapped_vars]


@pytest.mark.parametrize("case", ABI_CASES)
def test_invar_assembly_plan_preserves_source_order_and_runtime_static_cache(case: int) -> None:
    """Prepared stage-input plans must assemble each source into the same slot."""
    sharding = SingleDeviceSharding(jax.devices()[0])
    rank_devices = set(sharding.device_set)
    rank_submeshes = [_FakeSubmesh(), _FakeSubmesh(), _FakeSubmesh()]
    ri = case % 3
    orig_count = 18
    dynamic = {idx for idx in range(orig_count) if (idx + case) % 4 != 0}
    runtime_static = {idx for idx in dynamic if (idx + case) % 5 == 0}

    stage_outputs = [
        tuple(jnp.asarray([3000 + rank * 100 + pos], dtype=jnp.int32) for pos in range(8)) for rank in range(3)
    ]
    prev_outputs = tuple(jnp.asarray([4000 + pos], dtype=jnp.int32) for pos in range(8))
    flat_args_first = [jnp.asarray([1000 + idx], dtype=jnp.int32) for idx in range(orig_count)]
    flat_args_second = [jnp.asarray([2000 + idx], dtype=jnp.int32) for idx in range(orig_count)]
    placed = {(ri, idx): jnp.asarray([5000 + idx], dtype=jnp.int32) for idx in range(orig_count) if idx not in dynamic}

    source_cycle: list[tuple] = []
    for slot in range(14):
        selector = (slot + case) % 5
        if selector in (0, 1):
            source_cycle.append(("orig", (slot * 3 + case) % orig_count))
        elif selector == 2:
            source_cycle.append(("stage", (slot + case) % 3, (slot * 2 + case) % 8, None))
        else:
            source_cycle.append(("prev", (slot + case) % 8))

    plan = _prepare_invar_assembly_plan(source_cycle, placed, dynamic, ri)
    runtime_static_cache: dict[tuple[int, int], tp.Any] = {}
    _ = _assemble_invars_from_plan(
        plan,
        flat_args_first,
        {},
        prev_outputs,
        stage_outputs,
        ri,
        sharding,
        rank_devices,
        rank_submeshes,
        None,
        runtime_static_flat_indices=runtime_static,
        runtime_static_cache=runtime_static_cache,
    )
    assembled = _assemble_invars_from_plan(
        plan,
        flat_args_second,
        {},
        prev_outputs,
        stage_outputs,
        ri,
        sharding,
        rank_devices,
        rank_submeshes,
        None,
        runtime_static_flat_indices=runtime_static,
        runtime_static_cache=runtime_static_cache,
    )

    expected: list[int] = []
    for source in source_cycle:
        if source[0] == "orig":
            orig_idx = int(source[1])
            if orig_idx in dynamic:
                expected.append(1000 + orig_idx if orig_idx in runtime_static else 2000 + orig_idx)
            else:
                expected.append(5000 + orig_idx)
        elif source[0] == "stage":
            expected.append(3000 + int(source[1]) * 100 + int(source[2]))
        else:
            expected.append(4000 + int(source[1]))

    assert [_array_scalar(value) for value in assembled] == expected


@pytest.mark.parametrize("case", ABI_CASES)
def test_pipeline_executor_stage_local_carry_map_never_swaps_leaves(case: int) -> None:
    """Wavefront stage-local carry leaves must return to the matching input slot."""
    leaf_count = 2 + (case % 5)
    sharding = SingleDeviceSharding(jax.devices()[0])
    submesh = _FakeSubmesh()

    def stage0(*invars):
        """Update stage-0 state leaves and produce an activation."""
        states = invars[:leaf_count]
        x = invars[leaf_count]
        updated = tuple(state + x + idx for idx, state in enumerate(states))
        return (*updated, x + 1)

    def stage1(*invars):
        """Update stage-1 state leaves and produce an activation."""
        states = invars[:leaf_count]
        activation = invars[leaf_count]
        updated = tuple(state + activation + 10 * (idx + 1) for idx, state in enumerate(states))
        return (*updated, activation + 1)

    def stage2(*invars):
        """Update stage-2 state leaves and produce the final value."""
        states = invars[:leaf_count]
        activation = invars[leaf_count]
        updated = tuple(state + activation + 100 * (idx + 1) for idx, state in enumerate(states))
        return (*updated, activation * 2)

    x_flat_idx = 3 * leaf_count
    state = {
        "compiled": [
            (stage0, submesh, sharding, None, [*(("orig", idx) for idx in range(leaf_count)), ("orig", x_flat_idx)]),
            (
                stage1,
                submesh,
                sharding,
                None,
                [*(("orig", leaf_count + idx) for idx in range(leaf_count)), ("stage", 0, leaf_count)],
            ),
            (
                stage2,
                submesh,
                sharding,
                None,
                [*(("orig", 2 * leaf_count + idx) for idx in range(leaf_count)), ("stage", 1, leaf_count)],
            ),
        ],
        "placed": {},
        "dynamic": set(range(3 * leaf_count + 1)),
        "explicit_in_sh": {},
        "fn_outvar_map": [
            *((0, idx) for idx in range(leaf_count)),
            *((1, idx) for idx in range(leaf_count)),
            *((2, idx) for idx in range(leaf_count)),
            (2, leaf_count),
        ],
        "mpmd_mesh": None,
        "out_shardings": None,
        "result_treedef": None,
    }

    def fn(*args):
        del args
        raise AssertionError("direct fake fn path should not run")

    def prepare(*args, **kwargs):
        del args, kwargs
        return state

    fn._mpmd_prepare = prepare  # type: ignore[attr-defined]
    zero_states = tuple(jnp.asarray([0], dtype=jnp.int32) for _ in range(3 * leaf_count))
    batches = [(*zero_states, jnp.asarray([step + 1], dtype=jnp.int32)) for step in range(4)]
    carry_map = {
        0: {idx: idx for idx in range(leaf_count)},
        1: {leaf_count + idx: idx for idx in range(leaf_count)},
        2: {2 * leaf_count + idx: idx for idx in range(leaf_count)},
    }

    executor = MpmdPipelineExecutor(use_workers=bool(case % 2))
    outputs = executor.dispatch_many(fn, batches, carry_input_output_map=carry_map)
    executor.shutdown()

    got_stage0 = [[_array_scalar(output[idx]) for idx in range(leaf_count)] for output in outputs]
    got_stage1 = [[_array_scalar(output[leaf_count + idx]) for idx in range(leaf_count)] for output in outputs]
    got_stage2 = [[_array_scalar(output[2 * leaf_count + idx]) for idx in range(leaf_count)] for output in outputs]

    expected_stage0 = []
    expected_stage1 = []
    expected_stage2 = []
    s0 = [0] * leaf_count
    s1 = [0] * leaf_count
    s2 = [0] * leaf_count
    for step in range(4):
        x = step + 1
        s0 = [value + x + idx for idx, value in enumerate(s0)]
        activation1 = x + 1
        s1 = [value + activation1 + 10 * (idx + 1) for idx, value in enumerate(s1)]
        activation2 = activation1 + 1
        s2 = [value + activation2 + 100 * (idx + 1) for idx, value in enumerate(s2)]
        expected_stage0.append(list(s0))
        expected_stage1.append(list(s1))
        expected_stage2.append(list(s2))

    assert got_stage0 == expected_stage0
    assert got_stage1 == expected_stage1
    assert got_stage2 == expected_stage2


@pytest.mark.parametrize("case", ABI_CASES)
def test_stage_region_metadata_and_parent_edge_order_survive_mixed_pytrees(case: int) -> None:
    """Region-local markers must not scramble parent stage boundaries or metadata."""
    local_spec = PartitionSpec("tp") if case % 2 else PartitionSpec(("fsdp", "sp"), "tp")
    parent_spec = PartitionSpec("dp", "tp") if case % 3 else PartitionSpec(None, "tp")
    region = spx.sxstage_region(
        f"tower_{case}",
        schedule=spx.GPipe(microbatches=1 + (case % 4)),
        batch_argnums=(0,),
        static_argnums=(2,),
        donate_argnums=(1,),
    )

    def body(x, y, metadata_scale):
        """Wrap a local staged tower and then emit one parent stage boundary."""

        def tower(payload):
            left, right = payload
            hidden = left + right * jnp.asarray(metadata_scale, dtype=left.dtype)
            hidden = sxstage_iter(hidden, stage=0, sharding=local_spec)
            return {"hidden": jnp.tanh(hidden), "skip": left * 0.25}

        out = region(tower)((x, y))
        parent = sxstage_iter(out["hidden"] + out["skip"], stage=0, sharding=parent_spec)
        return parent.sum()

    jaxpr = jax.make_jaxpr(body)(
        jnp.ones((2, 3), dtype=jnp.float32),
        jnp.full((2, 3), 2.0, dtype=jnp.float32),
        3,
    ).jaxpr

    specs = stage_region_specs(jaxpr)
    clusters = cluster_jaxpr_by_markers(
        jaxpr,
        ignore_region_local_markers=True,
        extra_boundary_positions=stage_region_cluster_boundaries(jaxpr),
    )

    assert has_stage_regions(jaxpr)
    assert {spec.name for spec in specs} == {f"tower_{case}"}
    assert {spec.batch_argnums for spec in specs} == {(0,)}
    assert {spec.static_argnums for spec in specs} == {(2,)}
    assert {spec.donate_argnums for spec in specs} == {(1,)}
    assert marker_edge_shardings(jaxpr, ignore_region_local_markers=False) == [local_spec, parent_spec]
    assert marker_edge_shardings(jaxpr, ignore_region_local_markers=True) == [parent_spec]
    assert len(clusters) == 2
