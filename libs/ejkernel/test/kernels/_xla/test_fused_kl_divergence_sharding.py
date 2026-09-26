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
# See the License for the specific language governing permissions and
# limitations under the License.

"""XLA vocab-parallel (shard_map) KL direction parity.

The public ``fused_kl_divergence`` documents that its ``mesh`` / ``in_specs``
path supports ``forward`` / ``reverse`` / ``jsd`` over a sharded vocab axis
(not just forward). EasyDeL's distillation loss relies on this: it routes
``reverse`` / ``jsd`` (and the CAKLD ``gamma*reverse + (1-gamma)*forward``
blend) through the vocab-parallel path under TP. This test pins that contract
so a future kernel change cannot silently drop the per-shard normalizer from
the reverse/JSD VJP (the exact class of bug ``check_vma`` exists to prevent).

CPU-runnable; force devices in the shell before pytest (a conftest may import
jax first, so the ``setdefault`` below is best-effort only)::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
    pytest libs/ejkernel/test/kernels/_xla/test_fused_kl_divergence_sharding.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules.operations import fused_kl_divergence
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

if jax.device_count() < 2:
    pytest.skip(
        "needs >=2 devices for the tp=2 vocab-parallel cases — set "
        "XLA_FLAGS=--xla_force_host_platform_device_count=4 (with JAX_PLATFORMS=cpu) "
        "in the shell before pytest starts",
        allow_module_level=True,
    )

B, S, V, TEMP = 2, 6, 8, 2.5
LOGIT_SPEC = P(None, None, "tp")
TOKEN_SPEC = P(None, None)


def _inputs():
    ks, kt, km = jax.random.split(jax.random.PRNGKey(0), 3)
    student = jax.random.normal(ks, (B, S, V), jnp.float32)
    teacher = jax.random.normal(kt, (B, S, V), jnp.float32)
    mask = jax.random.uniform(km, (B, S), jnp.float32, minval=0.2, maxval=1.0)
    return student, teacher, mask


def _mesh():
    return Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("tp",))


def _dense(student, teacher, mask, direction, beta=0.5):
    return fused_kl_divergence(
        student,
        teacher,
        mask,
        reduction="mean",
        direction=direction,
        temperature=TEMP,
        beta=beta,
        platform="xla",
    ).loss


def _sharded(student, teacher, mask, mesh, direction, beta=0.5):
    return fused_kl_divergence(
        student,
        teacher,
        mask,
        reduction="mean",
        direction=direction,
        temperature=TEMP,
        beta=beta,
        platform="xla",
        mesh=mesh,
        in_specs=(LOGIT_SPEC, LOGIT_SPEC, TOKEN_SPEC),
        out_specs=P(),
    ).loss


@pytest.mark.parametrize("direction", ["forward", "reverse", "jsd"])
def test_xla_vocab_parallel_kl_direction_value_parity(direction):
    """Sharded vocab-parallel KL matches the dense value for every direction."""
    student, teacher, mask = _inputs()
    mesh = _mesh()
    with mesh:
        ref = _dense(student, teacher, mask, direction)
        got = _sharded(student, teacher, mask, mesh, direction)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("direction", ["forward", "reverse", "jsd"])
def test_xla_vocab_parallel_kl_direction_grad_parity(direction):
    """The vocab-parallel reverse/JSD VJP (via forced ``check_vma``) matches dense."""
    student, teacher, mask = _inputs()
    mesh = _mesh()
    grad_ref = jax.grad(lambda s: _dense(s, teacher, mask, direction))(student)
    with mesh:
        grad_got = jax.grad(lambda s: _sharded(s, teacher, mask, mesh, direction))(student)
    np.testing.assert_allclose(np.asarray(grad_got), np.asarray(grad_ref), rtol=0.0, atol=1e-5)


def test_xla_vocab_parallel_cakld_blend_parity():
    """The CAKLD ``gamma*reverse + (1-gamma)*forward`` blend matches dense under TP."""
    student, teacher, mask = _inputs()
    mesh = _mesh()
    gamma = 0.7
    ref = gamma * _dense(student, teacher, mask, "reverse") + (1.0 - gamma) * _dense(student, teacher, mask, "forward")
    with mesh:
        got = gamma * _sharded(student, teacher, mask, mesh, "reverse") + (1.0 - gamma) * _sharded(
            student, teacher, mask, mesh, "forward"
        )
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=0.0, atol=1e-5)
