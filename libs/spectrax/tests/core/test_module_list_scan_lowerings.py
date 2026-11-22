# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``ModuleList.scan`` lowering options: ``block_scan``, ``barrier``, ``prefetch_axes`` and ``prefetch_defer_dw``.

Every option is checked against the default segmented scan (forward value and gradients, including the gradient of
a value the layer function closes over) and for the compiled-program property it exists for. Runs on a 4-device
``fsdp`` mesh.
"""

from __future__ import annotations

import functools
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from spectrax.core.containers import (
    ModuleList,
    _build_scan_plan_from_modules,
    _group_periodic_segments,
    _scan_state_constraint_specs,
    _ScanBlock,
    _ScanSegment,
)
from spectrax.core.module import Module
from spectrax.core.scan_prefetch import gathered_sharding, plan_position, spec_mentions
from spectrax.core.variable import Buffer, Parameter
from spectrax.sharding.mesh import use_mesh

D = 256
BATCH = 8
SPECS = {"w": P("fsdp", "tp"), "up": P("fsdp", "tp"), "down": P("tp", "fsdp")}
WEIGHT_SHAPES = {(D, D), (D, 2 * D), (2 * D, D)}


class _Dense(Module):
    """Family A: one fsdp-sharded matrix and a replicated bias."""

    def __init__(self, k, key):
        super().__init__()
        self.w = Parameter(jax.random.normal(key, (D, D), jnp.float32) / math.sqrt(D), sharding=("fsdp", "tp"))
        self.b = Parameter(jnp.full((D,), 0.01 * k, jnp.float32))

    def forward(self, x):
        return jnp.tanh(x @ self.w.value + self.b.value)


class _Experts(Module):
    """A child stored under the ``experts`` key (target of ``prefetch_exclude``)."""

    def __init__(self, key):
        super().__init__()
        self.w = Parameter(jax.random.normal(key, (D, D), jnp.float32) / math.sqrt(D), sharding=("fsdp", "tp"))


class _Mlp(Module):
    """Family B: column- and row-sharded matrices plus an ``experts`` child."""

    def __init__(self, key):
        super().__init__()
        k1, k2, k3 = jax.random.split(key, 3)
        self.up = Parameter(jax.random.normal(k1, (D, 2 * D), jnp.float32) / math.sqrt(D), sharding=("fsdp", "tp"))
        self.down = Parameter(jax.random.normal(k2, (2 * D, D), jnp.float32) / math.sqrt(2 * D), sharding=("tp", "fsdp"))
        self.experts = _Experts(k3)

    def forward(self, x):
        return x + jax.nn.gelu(x @ self.up.value) @ self.down.value + 0.1 * jnp.tanh(x @ self.experts.w.value)


@functools.cache
def _layers(n_blocks: int = 3) -> ModuleList:
    """``[dense] + [mlp, dense, dense] * n_blocks + [mlp]``: head segment, periodic run, tail segment."""
    keys = iter(jax.random.split(jax.random.PRNGKey(0), 64))
    mods = [_Dense(0, next(keys))]
    for b in range(n_blocks):
        mods += [_Mlp(next(keys)), _Dense(b + 1, next(keys)), _Dense(b + 2, next(keys))]
    mods += [_Mlp(next(keys))]
    return ModuleList(mods)


def _mesh() -> Mesh:
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices")
    return Mesh(np.asarray(jax.devices()[:4]).reshape(4, 1), ("fsdp", "tp"))


def _leaf_key(path) -> str:
    return jax.tree_util.keystr(path).rsplit("[", 1)[-1].strip("'[]")


def _place(state, mesh):
    return jax.tree_util.tree_map_with_path(
        lambda p, v: jax.device_put(v, NamedSharding(mesh, SPECS.get(_leaf_key(p), P()))), state
    )


@functools.cache
def _run(options: tuple[tuple[str, object], ...], container: str = "list"):
    """Compile ``value_and_grad`` of ``sum(scan(...)**2)`` w.r.t. the layer state and a closed-over scale."""
    kw = dict(options)
    mods = (
        _layers()
        if container == "list"
        else ModuleList([_Dense(i, k) for i, k in enumerate(jax.random.split(jax.random.PRNGKey(3), 5))]).stack()
    )
    gdef, state = spx.export(mods)
    mesh = _mesh()

    def loss(st, x, s):
        y = spx.bind(gdef, st).scan(lambda m, c: m(c) * s, x, **kw)
        return jnp.sum(y**2)

    with use_mesh(mesh), jax.default_matmul_precision("highest"):
        st = _place(state, mesh)
        x = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(1), (BATCH, D), jnp.float32), NamedSharding(mesh, P("fsdp", None))
        )
        vg = jax.value_and_grad(loss, argnums=(0, 2))
        compiled = jax.jit(vg).lower(st, x, 1.0).compile()
        value, (g_state, g_scale) = compiled(st, x, 1.0)
        jaxpr = jax.make_jaxpr(vg)(st, x, 1.0)
        fwd_jaxpr = jax.make_jaxpr(loss)(st, x, 1.0)
    return {
        "value": float(value),
        "grads": [np.asarray(g) for g in jax.tree_util.tree_leaves(g_state)] + [np.asarray(g_scale)],
        "grad_state": g_state,
        "hlo": compiled.as_text(),
        "memory": compiled.memory_analysis(),
        "jaxpr": jaxpr,
        "fwd_jaxpr": fwd_jaxpr,
        "mesh": mesh,
    }


def _opts(**kw) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(kw.items()))


def _assert_matches(ref, got, *, value_rtol, grad_rtol):
    assert abs(ref["value"] - got["value"]) <= value_rtol * abs(ref["value"]), (ref["value"], got["value"])
    assert len(ref["grads"]) == len(got["grads"])
    for a, b in zip(ref["grads"], got["grads"], strict=True):
        scale = max(float(np.max(np.abs(a))), 1e-30)
        assert float(np.max(np.abs(a - b))) <= grad_rtol * scale, float(np.max(np.abs(a - b))) / scale


def _while_bodies(hlo: str) -> set[str]:
    return set(re.findall(r"body=%?([\w.\-]+)", hlo))


def _walk_eqns(jaxpr):
    for eqn in jaxpr.eqns:
        yield eqn
        for sub in eqn.params.values():
            for item in sub if isinstance(sub, tuple | list) else (sub,):
                inner = getattr(item, "jaxpr", None)
                if inner is not None:
                    yield from _walk_eqns(inner if hasattr(inner, "eqns") else item)


def _scan_arity(eqn) -> tuple[int, int]:
    """``(num_consts, num_carry)`` of a scan equation (JAX 0.11 stores them as ``ft_in`` / ``ft_out`` groups)."""
    if "num_consts" in eqn.params:
        return eqn.params["num_consts"], eqn.params["num_carry"]
    consts, carry, _xs = eqn.params["ft_in"].unpack()
    return len(consts), len(carry)


def _forward_scans(jaxpr, min_length=2):
    return [
        e
        for e in _walk_eqns(jaxpr.jaxpr)
        if e.primitive.name == "scan" and not e.params["reverse"] and e.params["length"] >= min_length
    ]


@pytest.mark.cpu_ok
def test_grouping_folds_the_periodic_run_and_keeps_head_and_tail():
    def seg(start, stop, fid):
        return _ScanSegment(start=start, stop=stop, gdef=None, stacked={"w": np.zeros((stop - start, 2))}, family_id=fid)

    segs = (
        seg(0, 1, 0),
        seg(1, 2, 1),
        seg(2, 5, 0),
        seg(5, 6, 1),
        seg(6, 9, 0),
        seg(9, 10, 1),
        seg(10, 13, 0),
        seg(13, 14, 1),
    )
    out = _group_periodic_segments(segs)
    assert [type(o).__name__ for o in out] == ["_ScanSegment", "_ScanBlock", "_ScanSegment"]
    blk = out[1]
    assert isinstance(blk, _ScanBlock) and blk.blocks == 3 and (blk.start, blk.stop) == (1, 13)
    assert [p.length for p in blk.positions] == [1, 3]
    assert blk.positions[1].stacked["w"].shape == (3, 3, 2)
    assert [m.start for m in blk.members[1]] == [2, 6, 10]
    unfoldable = (True, True, False, True, True, True, True, True)
    assert all(isinstance(o, _ScanSegment) for o in _group_periodic_segments(segs, unfoldable)[:3])


def test_default_scan_is_unchanged_by_the_new_arguments():
    plain = _run(_opts())
    explicit = _run(_opts(block_scan=False, barrier=False, prefetch_axes=None, prefetch_defer_dw=False))
    assert plain["value"] == explicit["value"]
    assert all(np.array_equal(a, b) for a, b in zip(plain["grads"], explicit["grads"], strict=True))


def test_block_scan_matches_and_compiles_fewer_loop_bodies():
    plain = _run(_opts())
    block = _run(_opts(block_scan=True))
    _assert_matches(plain, block, value_rtol=1e-6, grad_rtol=1e-5)
    assert len(_while_bodies(block["hlo"])) < len(_while_bodies(plain["hlo"]))


def test_barrier_matches_and_emits_optimization_barriers():
    plain = _run(_opts())
    for kw in ({"barrier": True}, {"barrier": True, "block_scan": True}):
        got = _run(_opts(**kw))
        _assert_matches(plain, got, value_rtol=1e-6, grad_rtol=1e-5)
        assert "optimization_barrier" in str(got["jaxpr"])
    assert "optimization_barrier" not in str(plain["jaxpr"])


@pytest.mark.parametrize("block_scan", [False, True])
@pytest.mark.parametrize("barrier", [False, True])
def test_prefetch_matches_the_plain_scan(block_scan, barrier):
    plain = _run(_opts())
    got = _run(_opts(prefetch_axes=("fsdp",), block_scan=block_scan, barrier=barrier))
    assert "custom_vjp_call" in str(got["fwd_jaxpr"])
    _assert_matches(plain, got, value_rtol=1e-6, grad_rtol=1e-5)


def test_prefetch_on_a_stacked_module_list_matches():
    plain = _run(_opts(), "stacked")
    got = _run(_opts(prefetch_axes="fsdp"), "stacked")
    assert "custom_vjp_call" in str(got["fwd_jaxpr"])
    _assert_matches(plain, got, value_rtol=1e-6, grad_rtol=1e-5)


@pytest.mark.parametrize("block_scan", [False, True])
def test_prefetch_keeps_weights_and_their_cotangents_sharded(block_scan):
    plain = _run(_opts())
    got = _run(_opts(prefetch_axes=("fsdp",), block_scan=block_scan))
    mesh = got["mesh"]
    assert got["memory"].argument_size_in_bytes == plain["memory"].argument_size_in_bytes
    for shape in re.findall(r"= f32\[([\d,]+)\]\S* all-gather(?:-start)?\(", got["hlo"]):
        dims = tuple(int(d) for d in shape.split(","))
        assert not (dims[-2:] in WEIGHT_SHAPES and math.prod(dims[:-2]) > 1), dims
    for path, g in jax.tree_util.tree_leaves_with_path(got["grad_state"]):
        if g.ndim == 2:
            assert g.sharding.is_equivalent_to(NamedSharding(mesh, SPECS[_leaf_key(path)]), 2), (path, g.sharding)


def test_prefetch_carries_the_next_slot_gather_independently_of_the_layer():
    """In every forward loop body the gathered weights of the next slot are a loop-carry output whose value does
    not depend on the current layer's activation carry: the gather is issued a whole layer ahead of its use."""
    got = _run(_opts(prefetch_axes=("fsdp",), block_scan=True))
    scans = _forward_scans(got["jaxpr"])
    assert scans
    for eqn in scans:
        body = eqn.params["jaxpr"].jaxpr
        n_consts, n_carry = _scan_arity(eqn)
        carry_in = body.invars[n_consts : n_consts + n_carry]
        dependent = {v for v in carry_in if tuple(v.aval.shape) == (BATCH, D)}
        assert dependent
        for e in body.eqns:
            if any(not hasattr(v, "val") and v in dependent for v in e.invars):
                dependent.update(e.outvars)
        carry_out = set(body.outvars[:n_carry])
        prefetched = [
            e
            for e in body.eqns
            if e.primitive.name == "sharding_constraint"
            and e.outvars[0] in carry_out
            and "fsdp" not in str(e.params["sharding"].spec)
        ]
        assert prefetched, "no gathered weight is carried to the next slot"
        assert not any(e.outvars[0] in dependent for e in prefetched)


def test_prefetch_saves_only_activations_as_residuals():
    got = _run(_opts(prefetch_axes=("fsdp",), block_scan=True, barrier=True))
    ys = []
    for eqn in _forward_scans(got["jaxpr"]):
        ys += [tuple(v.aval.shape) for v in eqn.outvars[_scan_arity(eqn)[1] :]]
    assert ys
    assert not any(s[-2:] in WEIGHT_SHAPES for s in ys), ys
    assert any(s[-2:] == (BATCH, D) for s in ys), ys


def test_defer_dw_matches_and_merges_the_backward_loop_bodies():
    ref = _run(_opts(prefetch_axes=("fsdp",), block_scan=True, barrier=True))
    got = _run(_opts(prefetch_axes=("fsdp",), block_scan=True, barrier=True, prefetch_defer_dw=True))
    _assert_matches(ref, got, value_rtol=0.0, grad_rtol=1e-5)
    assert len(_while_bodies(got["hlo"])) < len(_while_bodies(ref["hlo"]))


def test_prefetch_plan_selects_2d_sharded_leaves_and_honours_exclude():
    with use_mesh(_mesh()):
        items = _group_periodic_segments(_build_scan_plan_from_modules(list(_layers()), context="t").segments)
        block = next(i for i in items if isinstance(i, _ScanBlock))
        assert block.blocks == 3 and [p.length for p in block.positions] == [1, 2]
        mlp = block.positions[0]
        specs = _scan_state_constraint_specs(mlp.gdef)

        def picked(**kw):
            plan, leaves = plan_position(mlp.gdef, mlp.stacked, specs, 1, ("fsdp",), block_axis=True, **kw)
            assert all(leaf.shape[:2] == (3, 1) for leaf in leaves)
            return plan, {plan.paths[i] for i in plan.pf_idx}

        _, all_picked = picked()
        assert all_picked == {"['parameters']['up']", "['parameters']['down']", "['parameters']['experts']['w']"}
        plan, kept = picked(exclude=("experts",))
        assert kept == {"['parameters']['up']", "['parameters']['down']"}
        assert tuple(plan.gathered[plan.paths.index("['parameters']['down']")].spec) == ("tp", None)
        none_plan, _ = plan_position(mlp.gdef, mlp.stacked, specs, 1, ("sp",), block_axis=True)
        assert none_plan.pf_idx == ()
        mesh = _mesh()
        assert tuple(gathered_sharding(NamedSharding(mesh, P(("fsdp", "tp"), None)), ("fsdp",)).spec) == ("tp", None)
        assert not spec_mentions(NamedSharding(mesh, P(None, "tp")), ("fsdp",))


def test_prefetch_with_an_unknown_axis_falls_back_to_the_plain_lowering():
    got = _run(_opts(prefetch_axes=("sp",), block_scan=True))
    assert "custom_vjp_call" not in str(got["fwd_jaxpr"])
    _assert_matches(_run(_opts(block_scan=True)), got, value_rtol=0.0, grad_rtol=0.0)


class _CountA(Module):
    def __init__(self, k):
        super().__init__()
        self.w = Parameter(jnp.full((D, D), 0.01 * k, jnp.float32), sharding=("fsdp", "tp"))
        self.count = Buffer(jnp.zeros((), jnp.float32))

    def forward(self, x):
        self.count.value = self.count.value + 1.0
        return jnp.tanh(x @ self.w.value) + self.count.value


class _CountB(_CountA):
    def __init__(self, k):
        super().__init__(k)
        self.b = Parameter(jnp.full((D,), 0.1 * k, jnp.float32))

    def forward(self, x):
        return super().forward(x) + self.b.value


@pytest.mark.parametrize("options", [{"block_scan": True}, {"block_scan": True, "prefetch_axes": ("fsdp",)}])
def test_block_scan_writes_back_layer_buffers(options):
    def build():
        return ModuleList([cls(i) for i, cls in enumerate([_CountA, _CountB, _CountB] * 3 + [_CountA])])

    layers, looped = build(), build()
    items = _group_periodic_segments(_build_scan_plan_from_modules(list(layers), context="t").segments)
    assert any(isinstance(i, _ScanBlock) for i in items)
    x = jnp.ones((BATCH, D), jnp.float32)
    with use_mesh(_mesh()):
        out = layers.scan(lambda m, c: m(c), x, **options)
        ref = looped.scan(lambda m, c: m(c), x, trace=True)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-6, atol=1e-6)
    assert [float(m.count.value) for m in layers] == [1.0] * len(layers)


class _Noisy(Module):
    def __init__(self, rngs):
        super().__init__()
        self.rngs = rngs

    def forward(self, x):
        return x + jax.random.uniform(self.rngs.default, x.shape)


class _NoisyScaled(_Noisy):
    def __init__(self, rngs):
        super().__init__(rngs)
        self.s = Parameter(jnp.full((4,), 0.5, jnp.float32))

    def forward(self, x):
        return super().forward(x) * self.s.value


def test_block_scan_keeps_a_shared_rng_threaded_like_the_python_loop():
    from spectrax.rng.rngs import Rngs

    def build():
        rngs = Rngs(0)
        return ModuleList([cls(rngs) for cls in [_Noisy, _NoisyScaled, _NoisyScaled] * 2]), rngs

    (layers, rngs), (looped, looped_rngs) = build(), build()
    x = jnp.zeros((2, 4), jnp.float32)
    out = layers.scan(lambda m, c: m(c), x, block_scan=True)
    ref = looped.scan(lambda m, c: m(c), x, trace=True)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-6, atol=1e-6)
    assert rngs.stream("default")._unpack()[2] == looped_rngs.stream("default")._unpack()[2]
