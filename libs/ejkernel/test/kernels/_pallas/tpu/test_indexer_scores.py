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

"""Pallas TPU ``indexer_scores`` (Mosaic-compiled forward + backward) vs the float64 oracle and the XLA reference."""

from __future__ import annotations

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels import Backend, Platform, kernel_registry
from ejkernel.kernels._pallas.tpu.indexer_scores import _pallas_impl_fwd as impl
from ejkernel.modules import IndexerScoresConfig, indexer_scores
from ejkernel.ops import FwdParams
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

_ORACLE_PATH = pathlib.Path(__file__).resolve().parents[2] / "_xla" / "test_indexer_scores.py"
_spec = importlib.util.spec_from_file_location("_indexer_scores_xla_test", _ORACLE_PATH)
_oracle_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle_mod)
make_inputs = _oracle_mod.make_inputs
oracle = _oracle_mod.oracle
rel = _oracle_mod.rel
grads = _oracle_mod.grads

PALLAS = kernel_registry.get("indexer_scores", platform=Platform.PALLAS, backend=Backend.TPU)
XLA = kernel_registry.get("indexer_scores", platform=Platform.XLA, backend=Backend.ANY)


@pytest.mark.parametrize(
    ("batch", "heads", "seq", "entries", "dim", "fwd_params"),
    [
        (2, 4, 64, 256, 32, None),  # one q tile, one entry tile
        (1, 8, 300, 700, 64, None),  # both axes padded
        (2, 8, 512, 1024, 64, FwdParams(q_blocksize=64, kv_blocksize=512)),  # several tiles on both axes
        (1, 2, 20, 130, 128, FwdParams(q_blocksize=16, kv_blocksize=128)),  # tiny rows, d = 128
        (1, 4, 250, 330, 32, FwdParams(q_blocksize=100, kv_blocksize=200)),  # non-multiple tiles get rounded
        (1, 3, 1, 5, 16, None),  # a single query, a ragged handful of entries
    ],
)
def test_fp32_highest_matches_oracle(batch, heads, seq, entries, dim, fwd_params):
    q, k, w, cot = make_inputs(batch, heads, seq, entries, dim, seed=seq + entries)
    kw = {"precision": "highest", "fwd_params": fwd_params}
    out = jax.jit(lambda a, b, c: PALLAS(a, b, c, **kw))(q, k, w)
    want, want_g = oracle(q, k, w, cot)
    assert out.shape == (batch, seq, entries) and out.dtype == jnp.float32
    assert rel(out, want) < 1e-5, rel(out, want)
    got_g = jax.jit(lambda a, b, c: grads(PALLAS, a, b, c, cot, **kw))(q, k, w)
    for name, got, ref, arg in zip(("query", "index_k", "weights"), got_g, want_g, (q, k, w), strict=True):
        assert got.shape == arg.shape and got.dtype == arg.dtype, name
        assert bool(jnp.isfinite(got).all()), name
        assert rel(got, ref) < 1e-5, (name, rel(got, ref))


def test_bf16_default_precision_tracks_the_reference():
    """Default precision with bf16 operands: as accurate as the XLA reference against the fp64 truth."""
    q, k, w, cot = make_inputs(1, 8, 256, 512, 64, dtype=jnp.bfloat16, seed=2)
    truth, truth_g = oracle(q, k, w, cot)
    out_p = jax.jit(PALLAS)(q, k, w)
    out_x = jax.jit(XLA)(q, k, w)
    assert rel(out_p, truth) <= 1.5 * rel(out_x, truth) + 1e-3
    gp = jax.jit(lambda a, b, c: grads(PALLAS, a, b, c, cot))(q, k, w)
    gx = jax.jit(lambda a, b, c: grads(XLA, a, b, c, cot))(q, k, w)
    for name, a, b, ref in zip(("query", "index_k", "weights"), gp, gx, truth_g, strict=True):
        assert a.dtype == b.dtype, name
        assert rel(a, ref) <= 1.5 * rel(b, ref) + 1e-3, (name, rel(a, ref), rel(b, ref))
        assert rel(a, ref) < 5e-2, (name, rel(a, ref))


def test_precision_highest_on_bf16_operands_is_exact():
    q, k, w, _ = make_inputs(1, 4, 128, 256, 64, dtype=jnp.bfloat16, seed=6)
    truth = oracle(q, k, w)
    a = jax.jit(lambda x, y, z: PALLAS(x, y, z, precision="highest"))(q, k, w)
    b = jax.jit(lambda x, y, z: PALLAS(x, y, z, precision=jax.lax.Precision.HIGHEST))(q, k, w)
    assert rel(a, truth) < 1e-5
    assert bool(jnp.array_equal(a, b))
    with pytest.raises(ValueError):
        PALLAS(q, k, w, precision="exact")


def test_mixed_operand_dtypes_follow_the_reference():
    """bf16 query, fp32 key: the key is cast to bf16 for the contraction; dK comes back fp32 (bf16-valued)."""
    q, k, w, cot = make_inputs(1, 4, 64, 128, 32, seed=7)
    q = q.astype(jnp.bfloat16)
    gp = jax.jit(lambda a, b, c: grads(PALLAS, a, b, c, cot))(q, k, w)
    gx = jax.jit(lambda a, b, c: grads(XLA, a, b, c, cot))(q, k, w)
    assert gp[1].dtype == gx[1].dtype == jnp.float32
    assert rel(gp[1], np.asarray(gx[1], np.float64)) < 2e-2


def test_vmem_capacity_comes_from_the_chip_and_tiles_fit():
    assert impl.vmem_capacity() == int(pltpu.get_tpu_info().vmem_capacity_bytes)
    small = 16 * 2**20
    bq, bt, limit = impl.vmem_safe_blocks(128, 256, num_heads=32, head_dim=128, itemsize=2, capacity=small)
    assert impl.program_bytes(bq, bt, 32, 128, 2) <= small // 2 and limit is None
    assert bq % 16 == 0 and bt % 128 == 0
    bq, bt, _ = impl.vmem_safe_blocks(100, 200, num_heads=8, head_dim=64, itemsize=2)
    assert bq % 16 == 0 and bt % 128 == 0 and bq >= 100 and bt >= 200


def test_many_heads_wide_dim_compiles_and_matches():
    """32 heads x 128 (the released indexer width) at the chip's own VMEM fit, bf16, forward and backward."""
    q, k, w, cot = make_inputs(1, 32, 256, 640, 128, dtype=jnp.bfloat16, seed=9)
    q32, k32 = q.astype(jnp.float32), k.astype(jnp.float32)
    out = jax.jit(lambda a, b, c: PALLAS(a, b, c, precision="highest"))(q, k, w)
    assert rel(out, oracle(q, k, w)) < 1e-5
    gp = jax.jit(lambda a, b, c: grads(PALLAS, a, b, c, cot, precision="highest"))(q32, k32, w)
    for got, ref in zip(gp, oracle(q32, k32, w, cot)[1], strict=True):
        assert rel(got, ref) < 1e-5


def test_modules_op_with_config_tiles():
    q, k, w, cot = make_inputs(2, 4, 200, 300, 64, seed=11)
    cfg = IndexerScoresConfig(block_q=64, block_t=128)
    out = jax.jit(lambda a, b, c: indexer_scores(a, b, c, precision="highest", platform="pallas", cfg=cfg))(q, k, w)
    want, want_g = oracle(q, k, w, cot)
    assert rel(out, want) < 1e-5
    got_g = jax.jit(
        lambda a, b, c: grads(indexer_scores, a, b, c, cot, precision="highest", platform="pallas", cfg=cfg)
    )(q, k, w)
    for got, ref in zip(got_g, want_g, strict=True):
        assert rel(got, ref) < 1e-5


@pytest.mark.parametrize(
    ("batch", "ins", "out"),
    [
        (4, (P("dp", None, None, None), P("dp", None, None), P("dp", None, None, None)), P("dp", None, None)),
        (2, (P("dp", None, "sp", None), P("dp", None, None), P("dp", None, "sp", None)), P("dp", "sp", None)),
        (1, (P(None, None, None, None), P(None, None, None), P(None, None, None, None)), P(None, None, None)),
    ],
)
def test_pallas_under_shard_map_on_four_chips(batch, ins, out):
    """A Mosaic call is not GSPMD-partitionable; the executor's shard_map path runs it per device."""
    if jax.device_count() < 4:
        pytest.skip("needs four TPU chips")
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices, ("dp",)) if batch == 4 else Mesh(devices.reshape(2, 2), ("dp", "sp"))
    q, k, w, cot = make_inputs(batch, 8, 128, 384, 64, seed=13 + batch)

    def sharded(a, b, c):
        return indexer_scores(a, b, c, platform="pallas", precision="highest", mesh=mesh, in_specs=ins, out_specs=out)

    want, want_g = oracle(q, k, w, cot)
    assert rel(jax.jit(sharded)(q, k, w), want) < 1e-5
    for got, ref in zip(jax.jit(lambda a, b, c: grads(sharded, a, b, c, cot))(q, k, w), want_g, strict=True):
        assert rel(got, ref) < 1e-5
