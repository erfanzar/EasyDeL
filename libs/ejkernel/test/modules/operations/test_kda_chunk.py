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

"""Public ``kda_chunk`` operation: executor dispatch, config, and the batch/head ``shard_map`` path."""

from __future__ import annotations

import functools
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules.operations import KdaChunk, KdaChunkConfig, kda_chunk
from jax.sharding import Mesh, PartitionSpec

_ORACLE_PATH = pathlib.Path(__file__).resolve().parents[2] / "kernels" / "_xla" / "test_kda_chunk.py"
_spec = importlib.util.spec_from_file_location("_kda_chunk_xla_test_mod", _ORACLE_PATH)
_oracle_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle_mod)
recurrent_oracle = _oracle_mod.recurrent_oracle
make_inputs = _oracle_mod.make_inputs
packed_ids = _oracle_mod.packed_ids
rel = _oracle_mod.rel


def test_shard_map_wrapper_is_defined_on_the_class():
    assert "create_shard_map_wrapper" in KdaChunk.__dict__


def test_config_roundtrip():
    cfg = KdaChunkConfig(block_h=2, backward="xla", bwd_reduce="mxu", platform="pallas", backend="tpu")
    assert KdaChunkConfig.from_dict(cfg.to_dict()) == cfg
    assert KdaChunkConfig.from_json(cfg.to_json()) == cfg


@pytest.mark.parametrize("platform", ["xla", "pallas", None])
def test_public_op_matches_oracle(platform):
    if platform == "pallas" and jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel")
    q, k, v, beta, g, s0 = make_inputs(0, L=200)
    seg = packed_ids(2, 200, boundaries=(37, 64, 130), pad_rows=((1, 180, None),))
    out, fin = jax.jit(functools.partial(kda_chunk, chunk_size=64, platform=platform))(q, k, v, beta, g, s0, seg)
    o_ref, f_ref, _ = recurrent_oracle(q, k, v, beta, g, s0, seg)
    assert rel(out, o_ref) < 2e-5 and rel(fin, f_ref) < 2e-5


def test_cfg_selects_backward():
    q, k, v, beta, g, s0 = make_inputs(1, L=128)

    def grads(cfg):
        def loss(*a):
            out, fin = kda_chunk(*a, chunk_size=64, cfg=cfg)
            return jnp.sum(out**2) + jnp.sum(fin)

        return jax.jit(jax.grad(loss, argnums=range(6)))(q, k, v, beta, g, s0)

    g_p = grads(KdaChunkConfig(backward="pallas"))
    g_x = grads(KdaChunkConfig(backward="xla", block_h=2))
    for a, b in zip(g_p, g_x, strict=True):
        assert rel(a, b) < 1e-4


@pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices")
@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
def test_shard_map_four_devices_matches_unsharded(packed):
    """Batch over ``dp`` x heads over ``tp`` on a 2x2 mesh: values and gradients match the single-program run."""
    q, k, v, beta, g, s0 = make_inputs(2, B=2, H=4, L=192)
    seg = packed_ids(2, 192, boundaries=(5, 70, 71, 140), pad_rows=((1, 150, None),), lead_pad=2) if packed else None
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))
    P = PartitionSpec
    bh4, bh3 = P("dp", "tp", None, None), P("dp", "tp", None)
    in_specs = (bh4, bh4, bh4, bh3, bh4, bh4, P("dp", None))
    out_specs = (bh4, bh4)
    w = jnp.asarray(np.random.RandomState(3).randn(2, 4, 192, 128).astype(np.float32))

    def loss(fn, *a):
        out, fin = fn(*a)
        return jnp.sum(out * w) + jnp.sum(fin), (out, fin)

    sharded = functools.partial(kda_chunk, chunk_size=64, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    plain = functools.partial(kda_chunk, chunk_size=64)
    args = (q, k, v, beta, g, s0, seg)
    vg = functools.partial(jax.value_and_grad, has_aux=True, argnums=range(6))
    (_, (o_s, f_s)), g_s = jax.jit(vg(functools.partial(loss, sharded)))(*args)
    (_, (o_p, f_p)), g_p = jax.jit(vg(functools.partial(loss, plain)))(*args)
    o_ref, f_ref, _ = recurrent_oracle(*args)
    assert rel(o_s, o_ref) < 2e-5 and rel(f_s, f_ref) < 2e-5
    assert rel(o_s, o_p) < 2e-5 and rel(f_s, f_p) < 2e-5
    for a, b in zip(g_s, g_p, strict=True):
        assert np.isfinite(np.asarray(a)).all()
        assert rel(a, b) < 1e-4


def test_shard_map_rejects_sequence_sharding():
    q, k, v, beta, g, _ = make_inputs(4, B=1, H=1, L=64)
    mesh = Mesh(np.array(jax.devices()[:1]), ("sp",))
    P = PartitionSpec
    bad = P(None, None, "sp", None)
    with pytest.raises(ValueError, match="unsharded sequence"):
        KdaChunk().create_shard_map_wrapper(
            q, k, v, beta, g, None, None,
            mesh=mesh, in_specs=(bad, bad, bad, P(None, None, "sp"), bad, None, None), out_specs=(bad, bad),
        )  # fmt: skip
