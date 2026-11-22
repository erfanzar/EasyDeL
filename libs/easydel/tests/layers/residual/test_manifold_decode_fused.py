# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Opt-in APPROXIMATE decode kernels of :class:`ManifoldHyperConnection` (``decode_fused_kernels``) on TPU."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.sharding import coerce_runtime_sharding_resolver
from easydel.layers.residual import HyperStreamSharding, ManifoldHyperConnection, ManifoldHyperConnectionConfig
from easydel.utils import set_inference_mode
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="the decode kernels run on TPU only")

HC, WIDTH, BATCH = 4, 512, 32


def _module(decode: bool, source=None):
    cfg = ManifoldHyperConnectionConfig(
        hidden_size=WIDTH, hc_mult=HC, iters=20, use_fused_coefficients=True, decode_fused_kernels=decode
    )
    module = ManifoldHyperConnection(
        cfg, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(3), mesh_source=source
    )
    graphdef, state = spx.export(module)
    rng = np.random.default_rng(7)
    leaves = state.flatten()
    for key, value in leaves.items():
        leaves[key] = jnp.asarray(rng.normal(0.1, 0.2, value.shape), value.dtype)
    return graphdef, spx.State.from_flat(leaves)


def _inputs():
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.standard_normal((BATCH, 1, HC, WIDTH)), jnp.bfloat16)
    out = jnp.asarray(rng.standard_normal((BATCH, 1, WIDTH)), jnp.bfloat16)
    return x, out


def _run(graphdef, state, x, out, inference: bool):
    """``(collapsed, new_streams)`` of one read + write, and whether the decode-kernel path was traced."""
    taken = []

    def step(state, x, out):
        module = spx.bind(graphdef, state)
        post, comb, collapsed = module(x)
        taken.append(comb is None)
        return collapsed, module.residual_write(x, out, post, comb)

    if inference:
        with set_inference_mode():
            got = jax.jit(step)(state, x, out)
    else:
        got = jax.jit(step)(state, x, out)
    return jax.device_get(got), taken[0]


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint16 if a.dtype.itemsize == 2 else np.uint32)


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(_bits(a), _bits(b))


def test_switch_off_and_training_paths_unchanged():
    """Switch off (inference mode) and either switch value outside inference mode run the unchanged XLA path."""
    x, out = _inputs()
    off = _module(False)
    on = _module(True)
    base, taken_base = _run(*off, x, out, inference=False)
    off_inf, taken_off = _run(*off, x, out, inference=True)
    on_train, taken_train = _run(*on, x, out, inference=False)
    assert not (taken_base or taken_off or taken_train)
    for got in (off_inf, on_train):
        assert all(_same(g, w) for g, w in zip(got, base, strict=True))


def test_switch_on_takes_decode_kernels_within_one_ulp():
    """Collapse and write are within one bf16 ulp of the XLA path (measured on v5p: 2 of 16384 collapse and 1 of
    65536 stream elements differ; XLA's in-module reduction order is not reproduced)."""
    x, out = _inputs()
    want, _ = _run(*_module(False), x, out, inference=True)
    got, taken = _run(*_module(True), x, out, inference=True)
    assert taken
    for g, w in zip(got, want, strict=True):
        g, w = np.asarray(g, np.float32), np.asarray(w, np.float32)
        assert (np.abs(g - w) <= 2.0**-7 * np.maximum(np.abs(g), np.abs(w))).all()
        assert np.count_nonzero(g != w) <= max(1, g.size // 1000)


def _source(topology):
    if jax.device_count() < 4:
        pytest.skip("needs four devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(topology), ("data", "tokens", "features"))
    axis = spx.PartitionAxis(
        batch_axis="data",
        query_sequence_axis="tokens",
        hidden_state_axis="features",
        decode_batch_axis="data",
        decode_hidden_state_axis="features",
    )
    return SimpleNamespace(mesh=mesh, runtime_sharding_resolver=coerce_runtime_sharding_resolver(axis, mesh=mesh))


@pytest.mark.parametrize(
    "topology,expect_taken",
    # tokens=4 over a length-1 sequence: streams replicated; data=4: batch split; features=4: hidden split (fallback).
    [((1, 4, 1), True), ((4, 1, 1), True), ((1, 1, 4), False)],
)
def test_four_chip_mesh_matches_one_chip(topology, expect_taken):
    """On a 4-chip mesh the decode path (shard_map) equals 1-chip runs on each device's rows, bit for bit."""
    x, out = _inputs()
    source = _source(topology)
    graphdef, state = _module(True, source)
    with set_inference_mode():
        spec = source.runtime_sharding_resolver.resolve(dynamic_axes=HyperStreamSharding, shape=x.shape)
    xs = jax.device_put(x, NamedSharding(source.mesh, spec))
    got, taken = _run(graphdef, state, xs, out, inference=True)
    assert taken == expect_taken, spec
    if not expect_taken:
        want, _ = _run(*_module(False, source), xs, out, inference=True)
        assert all(_same(g, w) for g, w in zip(got, want, strict=True)), spec
        return
    shards = topology[0]  # the batch axis ("data"); the other two are size 1 or unused by a length-1 sequence
    rows = BATCH // shards
    single = _module(True)
    parts = [
        _run(*single, x[i * rows : (i + 1) * rows], out[i * rows : (i + 1) * rows], inference=True)[0]
        for i in range(shards)
    ]
    want = [np.concatenate([p[k] for p in parts]) for k in range(2)]
    assert all(_same(np.asarray(g), w) for g, w in zip(got, want, strict=True)), spec
