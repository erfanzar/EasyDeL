# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Fused-layout transforms must preserve leaf shardings (regression)."""

from __future__ import annotations

from types import MappingProxyType

import jax
import numpy as np
import pytest


def test_retp_fused_state_preserves_leaf_sharding():
    """Regression: the retp transform must not downgrade a leaf's NamedSharding.

    The fused re-interleave rebuilds the last axis with eager jnp ops; GSPMD's
    inferred output sharding drops the tp axis of the fused output dim, which
    (combined with from_pretrained skipping shard_model when leaves are already
    NamedSharded) left every cross-tp-loaded fused weight tp-replicated: full
    bf16[5120,34816] forward all-gathers on a v5p-2048 fsdp64/tp4 run loading
    a fused_param_tp=2 checkpoint (2026-07-04).
    """
    from easydel.layers.layouts import retp_fused_optimizer_state, retp_fused_state
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    hidden = 32  # fused gate_up out dim = 2*hidden = 64; divisible by tp 2 and 4

    class _Stub:
        fused_params = MappingProxyType({"mlp.gate_up_proj": (hidden, hidden)})

    if len(jax.devices()) < 8:
        pytest.skip("needs 8 (fake) devices")
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 4), ("fsdp", "tp"))
    spec = PartitionSpec("fsdp", "tp")
    key = ("parameters", "mlp", "gate_up_proj", "weight")
    host = np.arange(16 * 2 * hidden, dtype=np.float32).reshape(16, 2 * hidden)

    # Reference result computed on host arrays (no shardings involved).
    reference = retp_fused_state(_Stub(), {key: host.copy()}, saved_tp=2, target_tp=4)[key]

    sharded = jax.device_put(host, NamedSharding(mesh, spec))
    out = retp_fused_state(_Stub(), {key: sharded}, saved_tp=2, target_tp=4)[key]
    assert out.sharding.is_equivalent_to(NamedSharding(mesh, spec), out.ndim), (
        f"retp dropped the tp axis: {out.sharding}"
    )
    np.testing.assert_array_equal(np.asarray(jax.device_get(out)), np.asarray(reference))

    # Optimizer slots must keep the parameter's sharding too.
    slot_key = ("mu", *key)
    opt_ref = retp_fused_optimizer_state(_Stub(), {slot_key: host.copy()}, saved_tp=2, target_tp=4)[slot_key]
    opt_out = retp_fused_optimizer_state(_Stub(), {slot_key: sharded}, saved_tp=2, target_tp=4)[slot_key]
    assert opt_out.sharding.is_equivalent_to(NamedSharding(mesh, spec), opt_out.ndim), (
        f"optimizer retp dropped the tp axis: {opt_out.sharding}"
    )
    np.testing.assert_array_equal(np.asarray(jax.device_get(opt_out)), np.asarray(opt_ref))
