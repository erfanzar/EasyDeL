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

"""TPU parity tests for the one-shot Pallas collectives vs raw lax collectives.

Requires a real TPU with at least 2 devices (single libtpu process per host).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, shard_map
from jax.sharding import Mesh, PartitionSpec

if jax.default_backend() != "tpu" or len(jax.devices()) < 2:
    pytest.skip("one-shot collective kernels need a TPU with >= 2 devices", allow_module_level=True)

from ejkernel.kernels._pallas.tpu.all_gather import all_gather
from ejkernel.kernels._pallas.tpu.all_reduce import all_reduce
from ejkernel.kernels._pallas.tpu.reduce_scatter import reduce_scatter

_TP = len(jax.devices())


def _mesh() -> Mesh:
    return Mesh(np.array(jax.devices()), axis_names=("tp",))


def _stacked_sum(x: jax.Array) -> jax.Array:
    return x.reshape(_TP, x.shape[0] // _TP, *x.shape[1:]).sum(0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(("rows", "cols"), [(16, 2048), (8, 128), (64, 512)])
def test_one_shot_all_reduce_matches_psum(dtype, rows: int, cols: int):
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(0), (rows * _TP, cols), dtype=dtype)

    fn = shard_map(
        lambda v: all_reduce(v, "tp", mode="one_shot"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    ref = shard_map(
        lambda v: lax.psum(v, "tp"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    out = fn(x)
    expected = ref(x)
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    if dtype == jnp.float32:
        assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)
    else:
        assert jnp.allclose(out.astype(jnp.float32), expected.astype(jnp.float32), rtol=2e-2, atol=2e-2), (
            "one-shot accumulates in f32; bf16 psum may round differently but must stay close"
        )


def test_one_shot_all_reduce_grad_identity():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(1), (16 * _TP, 512), dtype=jnp.float32)

    fn = shard_map(
        lambda v: all_reduce(v, "tp", mode="one_shot"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    ref = shard_map(
        lambda v: lax.psum(v, "tp"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    g = jax.grad(lambda v: jnp.sum(fn(v) ** 2))(x)
    g_ref = jax.grad(lambda v: jnp.sum(ref(v) ** 2))(x)
    assert jnp.allclose(g, g_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(("rows", "cols"), [(16, 2048), (8, 128)])
def test_one_shot_all_gather_matches_lax(dtype, rows: int, cols: int):
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(2), (rows * _TP, cols), dtype=dtype)

    fn = shard_map(
        lambda v: all_gather(v, "tp", mode="one_shot"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    out = fn(x)
    assert out.shape == x.shape
    assert jnp.array_equal(out, x), "all_gather is pure data movement and must be bit-exact"


def test_one_shot_all_gather_grad_is_reduce_scatter():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(3), (16 * _TP, 512), dtype=jnp.float32)

    fn = shard_map(
        lambda v: all_gather(v, "tp", mode="one_shot"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    ref = shard_map(
        lambda v: lax.all_gather(v, "tp", axis=0, tiled=True),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    g = jax.grad(lambda v: jnp.sum(fn(v) ** 2))(x)
    g_ref = jax.grad(lambda v: jnp.sum(ref(v) ** 2))(x)
    assert jnp.allclose(g, g_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_one_shot_reduce_scatter_matches_psum_scatter(dtype):
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(4), (8 * _TP * _TP, 512), dtype=dtype)

    fn = shard_map(
        lambda v: reduce_scatter(v, "tp", mode="one_shot"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec("tp"),
        check_vma=False,
    )
    ref = shard_map(
        lambda v: lax.psum_scatter(v, "tp", scatter_dimension=0, tiled=True),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec("tp"),
        check_vma=False,
    )
    out = fn(x)
    expected = ref(x)
    assert out.shape == expected.shape
    if dtype == jnp.float32:
        assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)
    else:
        assert jnp.allclose(out.astype(jnp.float32), expected.astype(jnp.float32), rtol=2e-2, atol=2e-2)


def test_one_shot_reduce_scatter_grad_is_all_gather():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(5), (8 * _TP * _TP, 256), dtype=jnp.float32)

    fn = shard_map(
        lambda v: reduce_scatter(v, "tp", mode="one_shot"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec("tp"),
        check_vma=False,
    )
    ref = shard_map(
        lambda v: lax.psum_scatter(v, "tp", scatter_dimension=0, tiled=True),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec("tp"),
        check_vma=False,
    )
    g = jax.grad(lambda v: jnp.sum(fn(v) ** 2))(x)
    g_ref = jax.grad(lambda v: jnp.sum(ref(v) ** 2))(x)
    assert jnp.allclose(g, g_ref, rtol=1e-5, atol=1e-5)


def test_one_shot_alignment_validation_raises():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(6), (8 * _TP, 100), dtype=jnp.float32)

    fn = shard_map(
        lambda v: all_reduce(v, "tp", mode="one_shot"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    with pytest.raises(ValueError, match="multiple of 128"):
        fn(x)


def test_auto_mode_falls_back_for_unaligned():
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(7), (8 * _TP, 100), dtype=jnp.float32)

    fn = shard_map(
        lambda v: all_reduce(v, "tp", mode="auto"),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    out = fn(x)
    assert jnp.allclose(out, _stacked_sum(x), rtol=1e-5, atol=1e-5)
