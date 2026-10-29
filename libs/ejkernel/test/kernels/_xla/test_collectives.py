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

"""Forward/backward parity of XLA all_reduce / all_gather / reduce_scatter vs raw lax."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.all_gather import all_gather
from ejkernel.kernels._xla.all_reduce import all_reduce
from ejkernel.kernels._xla.all_to_all import all_to_all
from ejkernel.kernels._xla.reduce_scatter import reduce_scatter
from jax import lax, shard_map
from jax.sharding import Mesh, PartitionSpec


def _candidate_tp_sizes() -> tuple[int, ...]:
    n = len(jax.devices())
    if n <= 1:
        return (1,)
    sizes = [2]
    if n not in sizes:
        sizes.append(n)
    return tuple(sorted(set(sizes)))


_TP_SIZES = tuple(tp for tp in _candidate_tp_sizes() if tp > 1)
if not _TP_SIZES:
    pytest.skip("collective tests need at least 2 devices", allow_module_level=True)


def _mesh(tp: int) -> Mesh:
    return Mesh(np.array(jax.devices()[:tp]), axis_names=("tp",))


@pytest.mark.parametrize("tp", _TP_SIZES)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_all_reduce_matches_psum(tp: int, dtype):
    mesh = _mesh(tp)
    x = jax.random.normal(jax.random.PRNGKey(0), (8 * tp, 128), dtype=dtype)

    fn = shard_map(
        lambda v: all_reduce(v, "tp"),
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
    assert jnp.allclose(out, expected)


@pytest.mark.parametrize("tp", _TP_SIZES)
def test_all_reduce_grad_matches_psum(tp: int):
    mesh = _mesh(tp)
    x = jax.random.normal(jax.random.PRNGKey(1), (8 * tp, 128), dtype=jnp.float32)

    fn = shard_map(
        lambda v: all_reduce(v, "tp"),
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


@pytest.mark.parametrize("tp", _TP_SIZES)
@pytest.mark.parametrize("gather_axis", [0, 1])
def test_all_gather_matches_lax(tp: int, gather_axis: int):
    mesh = _mesh(tp)
    x = jax.random.normal(jax.random.PRNGKey(2), (8 * tp, 16 * tp), dtype=jnp.float32)
    spec = PartitionSpec("tp") if gather_axis == 0 else PartitionSpec(None, "tp")

    fn = shard_map(
        lambda v: all_gather(v, "tp", gather_axis=gather_axis),
        mesh=mesh,
        in_specs=(spec,),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    out = fn(x)
    assert out.shape == x.shape
    assert jnp.allclose(out, x)


@pytest.mark.parametrize("tp", _TP_SIZES)
def test_all_gather_not_tiled_stacks(tp: int):
    mesh = _mesh(tp)
    x = jax.random.normal(jax.random.PRNGKey(3), (8 * tp, 32), dtype=jnp.float32)

    fn = shard_map(
        lambda v: all_gather(v, "tp", tiled=False),
        mesh=mesh,
        in_specs=(PartitionSpec("tp"),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    out = fn(x)
    assert out.shape == (tp, 8 * tp // tp, 32)
    assert jnp.allclose(out.reshape(x.shape), x)


@pytest.mark.parametrize("tp", _TP_SIZES)
def test_all_gather_grad_is_reduce_scatter(tp: int):
    mesh = _mesh(tp)
    x = jax.random.normal(jax.random.PRNGKey(4), (8 * tp, 64), dtype=jnp.float32)

    fn = shard_map(
        lambda v: all_gather(v, "tp"),
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


@pytest.mark.parametrize("tp", _TP_SIZES)
def test_reduce_scatter_matches_psum_scatter(tp: int):
    mesh = _mesh(tp)
    x = jax.random.normal(jax.random.PRNGKey(5), (8 * tp * tp, 128), dtype=jnp.float32)

    fn = shard_map(
        lambda v: reduce_scatter(v, "tp"),
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
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tp", _TP_SIZES)
def test_reduce_scatter_grad_is_all_gather(tp: int):
    mesh = _mesh(tp)
    x = jax.random.normal(jax.random.PRNGKey(6), (8 * tp * tp, 64), dtype=jnp.float32)

    fn = shard_map(
        lambda v: reduce_scatter(v, "tp"),
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


@pytest.mark.parametrize(
    ("op", "kwargs"),
    [
        (all_reduce, {}),
        (all_gather, {}),
        (reduce_scatter, {}),
        (all_to_all, {"split_axis": 0, "concat_axis": 1}),
    ],
)
def test_tp_size_validation_raises(op, kwargs):
    x = jnp.ones((8, 128), dtype=jnp.float32)
    with pytest.raises(ValueError, match="tp_size must be >= 1"):
        op(x, "tp", tp_size=0, **kwargs)


@pytest.mark.parametrize("tp", _TP_SIZES)
def test_all_to_all_matches_lax(tp: int):
    mesh = _mesh(tp)
    x = jax.random.normal(jax.random.PRNGKey(7), (8 * tp, 4 * tp), dtype=jnp.float32)

    fn = shard_map(
        lambda v: all_to_all(v, "tp", split_axis=0, concat_axis=1),
        mesh=mesh,
        in_specs=(PartitionSpec(None, "tp"),),
        out_specs=PartitionSpec("tp", None),
        check_vma=False,
    )
    ref = shard_map(
        lambda v: lax.all_to_all(v, "tp", split_axis=0, concat_axis=1, tiled=True),
        mesh=mesh,
        in_specs=(PartitionSpec(None, "tp"),),
        out_specs=PartitionSpec("tp", None),
        check_vma=False,
    )
    out = fn(x)
    expected = ref(x)
    assert out.shape == expected.shape
    assert jnp.array_equal(out, expected)


@pytest.mark.parametrize("tp", _TP_SIZES)
def test_all_to_all_grad_is_inverse_exchange(tp: int):
    mesh = _mesh(tp)
    x = jax.random.normal(jax.random.PRNGKey(8), (8 * tp, 4 * tp), dtype=jnp.float32)

    fn = shard_map(
        lambda v: all_to_all(v, "tp", split_axis=0, concat_axis=1),
        mesh=mesh,
        in_specs=(PartitionSpec(None, "tp"),),
        out_specs=PartitionSpec("tp", None),
        check_vma=False,
    )
    scale = jax.random.uniform(jax.random.PRNGKey(9), (8 * tp, 4 * tp), dtype=jnp.float32)
    g = jax.grad(lambda v: jnp.sum((fn(v) * scale) ** 2))(x)
    assert g.shape == x.shape
    assert jnp.isfinite(g).all()
    assert not jnp.allclose(g, 0.0)


def test_multi_axis_all_reduce_matches_lax():
    if len(jax.devices()) < 4:
        pytest.skip("multi-axis test needs 4 devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), axis_names=("a", "b"))
    x = jax.random.normal(jax.random.PRNGKey(10), (16, 64), dtype=jnp.float32)

    fn = shard_map(
        lambda v: all_reduce(v, ("a", "b")),
        mesh=mesh,
        in_specs=(PartitionSpec(("a", "b")),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    ref = shard_map(
        lambda v: lax.psum(v, ("a", "b")),
        mesh=mesh,
        in_specs=(PartitionSpec(("a", "b")),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    assert jnp.allclose(fn(x), ref(x), rtol=1e-6, atol=1e-6)


def test_multi_axis_reduce_scatter_matches_lax():
    if len(jax.devices()) < 4:
        pytest.skip("multi-axis test needs 4 devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), axis_names=("a", "b"))
    x = jax.random.normal(jax.random.PRNGKey(11), (64, 32), dtype=jnp.float32)

    fn = shard_map(
        lambda v: reduce_scatter(v, ("a", "b")),
        mesh=mesh,
        in_specs=(PartitionSpec(("a", "b")),),
        out_specs=PartitionSpec(("a", "b")),
        check_vma=False,
    )
    ref = shard_map(
        lambda v: lax.psum_scatter(v, ("a", "b"), scatter_dimension=0, tiled=True),
        mesh=mesh,
        in_specs=(PartitionSpec(("a", "b")),),
        out_specs=PartitionSpec(("a", "b")),
        check_vma=False,
    )
    assert jnp.allclose(fn(x), ref(x), rtol=1e-6, atol=1e-6)
