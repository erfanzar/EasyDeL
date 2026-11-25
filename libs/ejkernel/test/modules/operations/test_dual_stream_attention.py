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

"""Public module operation tests for Dual-Stream Hybrid Attention (including shard_map)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import DualStreamAttentionConfig, dual_stream_attention
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as Ps


@pytest.mark.parametrize("overlap", [True, False])
def test_module_dual_stream_attention_auto_vs_xla(overlap):
    """Public ejkernel.modules.dual_stream_attention matches explicit XLA platform in fwd and bwd."""
    rng = np.random.default_rng(33 if overlap else 44)
    bsz, nq, nkv, seq_len, dim, win = 2, 4, 2, 256, 128, 64
    q = jnp.asarray(rng.standard_normal((bsz, nq, seq_len, dim)) * 0.2, jnp.float32)
    kg = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.float32)
    vg = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.float32)
    kl = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.float32)
    vl = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.float32)

    seg = np.zeros((bsz, seq_len), dtype=np.int32)
    seg[0, 128:] = 1
    seg[1, 192:] = -1
    seg_j = jnp.asarray(seg)

    cfg = DualStreamAttentionConfig(sliding_window=win, overlap=overlap)
    out_auto = dual_stream_attention(q, kg, vg, kl, vl, seg_j, seg_j, seg_j, seg_j, cfg=cfg)
    out_xla = dual_stream_attention(q, kg, vg, kl, vl, seg_j, seg_j, seg_j, seg_j, cfg=cfg, platform="xla")

    assert out_auto.shape == q.shape
    assert np.isfinite(np.asarray(out_auto)).all()
    assert float(np.max(np.abs(np.asarray(out_auto) - np.asarray(out_xla)))) < 2e-3


def test_module_dual_stream_attention_shard_map_matches_unsharded():
    """Differentiable shard_map over (batch, head) matches unsharded execution."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("shard_map parity test requires >= 2 devices")
    tp = 2 if len(devices) >= 2 else 1
    fsdp = len(devices) // tp
    mesh = Mesh(np.array(devices).reshape(fsdp, tp), ("fsdp", "tp"))

    rng = np.random.default_rng(55)
    bsz, nq, nkv, seq_len, dim, win = max(2, fsdp), 4, 2, 256, 128, 64
    q = jnp.asarray(rng.standard_normal((bsz, nq, seq_len, dim)) * 0.2, jnp.bfloat16)
    kg = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.bfloat16)
    vg = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.bfloat16)
    kl = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.bfloat16)
    vl = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.bfloat16)
    seg = jnp.zeros((bsz, seq_len), dtype=jnp.int32)
    cot = jnp.asarray(rng.standard_normal((bsz, nq, seq_len, dim)) * 0.2, jnp.bfloat16)

    q_sh = jax.device_put(q, NamedSharding(mesh, Ps("fsdp", "tp", None, None)))
    kg_sh = jax.device_put(kg, NamedSharding(mesh, Ps("fsdp", "tp", None, None)))
    vg_sh = jax.device_put(vg, NamedSharding(mesh, Ps("fsdp", "tp", None, None)))
    kl_sh = jax.device_put(kl, NamedSharding(mesh, Ps("fsdp", "tp", None, None)))
    vl_sh = jax.device_put(vl, NamedSharding(mesh, Ps("fsdp", "tp", None, None)))
    seg_sh = jax.device_put(seg, NamedSharding(mesh, Ps("fsdp", None)))
    cot_sh = jax.device_put(cot, NamedSharding(mesh, Ps("fsdp", "tp", None, None)))

    q_spec = Ps("fsdp", "tp", None, None)
    seg_spec = Ps("fsdp", None)
    in_specs = (q_spec, q_spec, q_spec, q_spec, q_spec, seg_spec, seg_spec, seg_spec, seg_spec)

    def _sharded_loss(qa, kga, vga, kla, vla):
        out = dual_stream_attention(
            qa,
            kga,
            vga,
            kla,
            vla,
            seg_sh,
            seg_sh,
            seg_sh,
            seg_sh,
            sliding_window=win,
            overlap=True,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=q_spec,
        )
        return jnp.sum(out.astype(jnp.float32) * cot_sh.astype(jnp.float32)), out

    def _ref_loss(qa, kga, vga, kla, vla):
        out = dual_stream_attention(
            qa, kga, vga, kla, vla, seg, seg, seg, seg, sliding_window=win, overlap=True, platform="xla"
        )
        return jnp.sum(out.astype(jnp.float32) * cot.astype(jnp.float32)), out

    (_, out_sh), grads_sh = jax.jit(jax.value_and_grad(_sharded_loss, argnums=(0, 1, 2, 3, 4), has_aux=True))(
        q_sh, kg_sh, vg_sh, kl_sh, vl_sh
    )
    (_, out_ref), grads_ref = jax.value_and_grad(_ref_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(q, kg, vg, kl, vl)

    assert float(np.max(np.abs(np.asarray(out_sh, np.float32) - np.asarray(out_ref, np.float32)))) < 2e-2
    for gs, gr in zip(grads_sh, grads_ref, strict=True):
        assert float(np.max(np.abs(np.asarray(gs, np.float32) - np.asarray(gr, np.float32)))) < 6e-2
