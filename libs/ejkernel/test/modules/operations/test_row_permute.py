# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Public ``row_permute`` operation: executor dispatch, config, gradient, and the feature-axis ``shard_map`` path."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules.operations import RowPermute, RowPermuteConfig, row_permute
from jax.sharding import Mesh, PartitionSpec


def _runs(rows: int):
    """A partial permutation with tile-unaligned starts on both ends, a gap and a padding run."""
    runs = [[0, 301, 37], [37, 5, 200], [237, 600, 19], [300, 400, 150], [450, 207, 90], [0, 0, 0]]
    return np.asarray(runs, np.int32), rows


def _numpy_permute(x, runs, out_rows, init=None):
    x = np.asarray(x)
    out = np.zeros((out_rows, *x.shape[1:]), x.dtype) if init is None else np.array(init, copy=True)
    for d, s, n in runs:
        if n > 0:
            out[d : d + n] = x[s : s + n]
    return out


def test_shard_map_wrapper_is_defined_on_the_class():
    assert "create_shard_map_wrapper" in RowPermute.__dict__


def test_config_roundtrip():
    cfg = RowPermuteConfig(chunk_rows=128, platform="pallas", backend="tpu")
    assert RowPermuteConfig.from_dict(cfg.to_dict()) == cfg
    assert RowPermuteConfig.from_json(cfg.to_json()) == cfg


def test_default_config_dispatches_to_pallas_on_tpu():
    """The measured default: 128-row windows, platform ``auto`` (the Pallas entry on TPU)."""
    from ejkernel.kernels._registry import Backend, Platform, kernel_registry

    cfg = RowPermute().heuristic_cfg(None)
    assert cfg.chunk_rows == 128 and cfg.platform == "auto"
    if jax.default_backend() == "tpu":
        expected = kernel_registry.get("row_permute", platform=Platform.PALLAS, backend=Backend.TPU)
        assert RowPermute().get_impl(cfg) is expected


@pytest.mark.parametrize("platform", ["xla", "pallas", None])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_public_op_matches_reference(platform, dtype):
    if platform == "pallas" and jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel")
    runs, rows = _runs(1024)
    x = jnp.asarray(np.random.default_rng(0).standard_normal((rows, 256)), dtype)
    fn = jax.jit(functools.partial(row_permute, platform=platform, cfg=RowPermuteConfig(chunk_rows=64)))
    out = fn(x, jnp.asarray(runs))
    np.testing.assert_array_equal(np.asarray(out), _numpy_permute(x, runs, rows))


@pytest.mark.parametrize("platform", ["xla", "pallas"])
def test_public_op_gradient_equals_gather(platform):
    if platform == "pallas" and jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel")
    runs, rows = _runs(1024)
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((rows, 128)), jnp.float32)
    init = jnp.asarray(rng.standard_normal((rows, 128)), jnp.float32)
    cot = jnp.asarray(rng.standard_normal((rows, 128)), jnp.float32)
    index = np.zeros(rows, np.int32)
    covered = np.zeros(rows, bool)
    for d, s, n in runs:
        index[d : d + n] = np.arange(s, s + n)
        covered[d : d + n] = n > 0
    mask = jnp.asarray(covered)[:, None]

    def loss_op(x, init):
        return jnp.sum(row_permute(x, jnp.asarray(runs), None, init, platform=platform) * cot)

    def loss_ref(x, init):
        return jnp.sum(jnp.where(mask, x[jnp.asarray(index)], init) * cot)

    g_op = jax.jit(jax.grad(loss_op, argnums=(0, 1)))(x, init)
    g_ref = jax.jit(jax.grad(loss_ref, argnums=(0, 1)))(x, init)
    for a, b in zip(g_op, g_ref, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices")
def test_shard_map_feature_axis_matches_unsharded():
    runs, rows = _runs(1024)
    x = jnp.asarray(np.random.default_rng(2).standard_normal((rows, 512)), jnp.bfloat16)
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("tp",))
    spec = PartitionSpec(None, "tp")

    def sharded(x, r):
        return row_permute(x, r, mesh=mesh, in_specs=(spec, PartitionSpec(), None, None), out_specs=spec)

    out = jax.jit(sharded)(x, jnp.asarray(runs))
    np.testing.assert_array_equal(np.asarray(out), _numpy_permute(x, runs, rows))


def test_shard_map_rejects_sharded_rows():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    x = jnp.zeros((64, 128), jnp.float32)
    with pytest.raises(ValueError, match="row axis"):
        RowPermute().create_shard_map_wrapper(
            x,
            jnp.zeros((1, 3), jnp.int32),
            mesh=mesh,
            in_specs=(PartitionSpec("dp", None), PartitionSpec(), None, None),
            out_specs=PartitionSpec(),
        )
