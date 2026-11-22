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

"""Opt-in APPROXIMATE prefill kernels of :class:`ManifoldHyperConnection` (``prefill_fused_kernels``) on TPU."""

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

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="the prefill kernels run on TPU only")

HC, WIDTH, TOKENS = 4, 512, 1024


def _module(prefill: bool, source=None):
    cfg = ManifoldHyperConnectionConfig(
        hidden_size=WIDTH, hc_mult=HC, iters=20, use_fused_coefficients=True, prefill_fused_kernels=prefill
    )
    module = ManifoldHyperConnection(
        cfg, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(3), mesh_source=source
    )
    graphdef, state = spx.export(module)
    rng = np.random.default_rng(7)
    leaves = state.flatten()
    for key, value in leaves.items():
        std = 0.02 if value.ndim == 2 else 0.3
        leaves[key] = jnp.asarray(rng.normal(0.0 if value.ndim == 2 else 0.5, std, value.shape), value.dtype)
    return graphdef, spx.State.from_flat(leaves)


def _inputs():
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.standard_normal((1, TOKENS, HC, WIDTH)), jnp.bfloat16)
    out = jnp.asarray(rng.standard_normal((1, TOKENS, WIDTH)), jnp.bfloat16)
    return x, out


def _run(graphdef, state, x, out, inference: bool):
    """``(collapsed, new_streams)`` of one read + write, and whether the prefill-kernel path was traced."""
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


def _close(got, want):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape
    assert np.linalg.norm(got - want) <= 5e-3 * np.linalg.norm(want)


def test_switch_off_and_training_paths_unchanged():
    """Switch off (inference mode) and either switch value outside inference mode run the unchanged XLA path."""
    x, out = _inputs()
    off, on = _module(False), _module(True)
    base, taken_base = _run(*off, x, out, inference=False)
    off_inf, taken_off = _run(*off, x, out, inference=True)
    on_train, taken_train = _run(*on, x, out, inference=False)
    assert not (taken_base or taken_off or taken_train)
    for got in (off_inf, on_train):
        assert all(_same(g, w) for g, w in zip(got, base, strict=True))


def test_decode_sized_blocks_do_not_take_prefill_kernels():
    """At most 128 tokens per device stays off the prefill kernels (bitwise equal to the switch-off path)."""
    x, out = _inputs()
    x, out = x[:, :128], out[:, :128]
    want, _ = _run(*_module(False), x, out, inference=True)
    got, taken = _run(*_module(True), x, out, inference=True)
    assert not taken
    assert all(_same(g, w) for g, w in zip(got, want, strict=True))


def test_switch_on_takes_prefill_kernels_close_to_xla():
    """Collapse and write stay within bf16 rounding of the XLA path (relative L2 at most 5e-3)."""
    x, out = _inputs()
    want, _ = _run(*_module(False), x, out, inference=True)
    got, taken = _run(*_module(True), x, out, inference=True)
    assert taken
    for g, w in zip(got, want, strict=True):
        _close(g, w)


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


@pytest.mark.parametrize("topology", [(1, 4, 1), (1, 1, 4), (1, 2, 2)])
def test_four_chip_mesh_close_to_xla(topology):
    """On a 4-chip mesh (tokens split, hidden split with a psum of the statistics, both) the prefill path is taken
    and stays within bf16 rounding of the switch-off path on the same mesh."""
    x, out = _inputs()
    source = _source(topology)
    with set_inference_mode():
        spec = source.runtime_sharding_resolver.resolve(dynamic_axes=HyperStreamSharding, shape=x.shape)
    xs = jax.device_put(x, NamedSharding(source.mesh, spec))
    got, taken = _run(*_module(True, source), xs, out, inference=True)
    want, taken_off = _run(*_module(False, source), xs, out, inference=True)
    assert taken and not taken_off, spec
    for g, w in zip(got, want, strict=True):
        _close(g, w)
