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

"""Regression tests for the build-time MoE layout planner.

Every case here is a real v5p-2048 configuration measured on 2026-07-16
(qwen3.6-35B-A3B: 256 experts, top-8, H=2048, M=512, bf16):

(a) ``dp1/fsdp64/ep4/tp4`` with ``fsdp_is_ep_bound=True`` and the 8k bucket
    (B=512, S=8192): the token batch is replicated per core, the dispatch
    buffer is ``bf16[33554432, 2048]`` = 137.4 GB/core, and the TPU compiler
    rejected it via the jellyfish int32 bounds check — after burning ~40
    pod-minutes of compile. The planner must raise at trace time instead.
(b) Same mesh, ep-unbound, 131k bucket (B=256, S=131072): 17.2 GB/core
    buffer; the compile reached the HBM budgeter (CompileTimeHbmOom
    112.6G/95.7G) rather than the bounds check, so the planner warns but
    must NOT hard-fail.
(c) ``dp4/fsdp64/ep1/tp4``, 131k bucket: 4.3 GB/core, compiles.
(d) ``moe_chunk_size=65536`` caps (b) and (c) at chunk*k*H*2 = 2.1 GB.
(e) ``dp4/fsdp64/ep1/tp4``, expert weights sharded only over tp: 403 MB per
    layer per device (268 MB gate_up + 134 MB down), 15 GB/model at 40
    layers; optimizer init over that residency failed allocation. The
    planner must report the residency and recommend
    ``moe_fsdp_shard_expert_weights``.

The planner is pure shape math, so these run on any host with no devices.
"""

from __future__ import annotations

import pytest
from easydel.layers.moe import estimate_moe_layout, validate_moe_layout
from easydel.layers.moe._layout_planner import (
    FWD_BWD_LIVE_BUFFER_MULTIPLIER,
    INT32_ALLOCATION_HARD_BOUND_BYTES,
    INT32_ALLOCATION_WARN_BOUND_BYTES,
)

# qwen3.6-35B-A3B constants (per MoE layer).
K = 8
H = 2048
E = 256
M = 512
BF16 = 2
# gate_up [E, H, 2M] + down [E, M, H] in bf16: 1.61 GB / layer.
EXPERT_PARAMS_BYTES_PER_LAYER = E * H * (2 * M) * BF16 + E * M * H * BF16

MESH_FSDP64_EP4_TP4 = {"dp": 1, "fsdp": 64, "ep": 4, "tp": 4, "sp": 1}
MESH_DP4_FSDP64_EP1_TP4 = {"dp": 4, "fsdp": 64, "ep": 1, "tp": 4, "sp": 1}

TOKENS_8K_BUCKET = 512 * 8192  # B=512, S=8192
TOKENS_131K_BUCKET = 256 * 131072  # B=256, S=131072
DEVICES = 1024  # v5p-2048 = 1024 chips


def _estimate(mesh, tokens, **kw):
    """Estimate helper filling in the qwen3.6-35B-A3B per-layer constants."""
    defaults = dict(
        tokens_per_device=tokens // DEVICES,
        num_experts_per_tok=K,
        hidden_size=H,
        num_experts=E,
        expert_params_bytes=EXPERT_PARAMS_BYTES_PER_LAYER,
        dtype_bytes=BF16,
        fsdp_is_ep_bound=False,
        use_ring_of_experts=False,
        chunk_size=None,
    )
    defaults.update(kw)
    return estimate_moe_layout(mesh, **defaults)


def test_case_a_ep_bound_replicates_tokens_and_exceeds_int32_bound():
    """(a) ep-bound fsdp replicates the batch: 137.4 GB/core, provably fatal."""
    est = _estimate(MESH_FSDP64_EP4_TP4, TOKENS_8K_BUCKET, fsdp_is_ep_bound=True)
    assert est.local_tokens == TOKENS_8K_BUCKET, "ep-bound must leave the whole batch resident per core (dp=1)"
    assert est.dispatch_buffer_bytes == TOKENS_8K_BUCKET * K * H * BF16  # 137,438,953,472
    assert est.dispatch_buffer_bytes == 137_438_953_472
    assert est.dispatch_buffer_bytes > INT32_ALLOCATION_HARD_BOUND_BYTES
    assert est.exceeds_int32_allocation_bound
    assert "EXCEEDS" in est.summary


def test_case_a_validate_raises_with_actionable_message():
    """(a) the validation hook must raise before the 40-minute compile does."""
    with pytest.raises(ValueError) as excinfo:
        validate_moe_layout(
            MESH_FSDP64_EP4_TP4,
            tokens_per_device=TOKENS_8K_BUCKET // DEVICES,
            num_experts_per_tok=K,
            hidden_size=H,
            num_experts=E,
            expert_params_bytes=EXPERT_PARAMS_BYTES_PER_LAYER,
            dtype_bytes=BF16,
            fsdp_is_ep_bound=True,
            use_ring_of_experts=False,
            chunk_size=None,
        )
    message = str(excinfo.value)
    # The error must name every lever that shrinks the buffer.
    assert "moe_chunk_size" in message
    assert "fsdp_is_ep_bound" in message
    assert "sharding_axis_dims" in message
    assert "int32" in message


def test_case_b_ep_unbound_131k_bucket_warns_but_does_not_raise():
    """(b) 17.2 GB/core: over the 4-byte-word band, but not provably fatal."""
    est = _estimate(MESH_FSDP64_EP4_TP4, TOKENS_131K_BUCKET, fsdp_is_ep_bound=False)
    assert est.local_tokens == TOKENS_131K_BUCKET // 64, "batch must shard over dp*fsdp = 64 ways"
    assert est.dispatch_buffer_bytes == 17_179_869_184  # 16 GiB measured per core
    assert est.dispatch_buffer_bytes < INT32_ALLOCATION_HARD_BOUND_BYTES
    assert est.dispatch_buffer_bytes > INT32_ALLOCATION_WARN_BOUND_BYTES
    assert not est.exceeds_int32_allocation_bound
    assert est.near_int32_allocation_bound
    # ~3 live copies matched the 48G the compile-time HBM report attributed
    # to dispatch buffers.
    assert est.live_dispatch_bytes == est.dispatch_buffer_bytes * FWD_BWD_LIVE_BUFFER_MULTIPLIER

    # The validation hook logs a warning but returns normally.
    result = validate_moe_layout(
        MESH_FSDP64_EP4_TP4,
        tokens_per_device=TOKENS_131K_BUCKET // DEVICES,
        num_experts_per_tok=K,
        hidden_size=H,
        num_experts=E,
        expert_params_bytes=EXPERT_PARAMS_BYTES_PER_LAYER,
        dtype_bytes=BF16,
        fsdp_is_ep_bound=False,
        use_ring_of_experts=False,
        chunk_size=None,
    )
    assert result.dispatch_buffer_bytes == est.dispatch_buffer_bytes


def test_case_c_dp4_mesh_fits():
    """(c) dp4/fsdp64/ep1/tp4 shards the batch 256 ways: 4.3 GB/core, compiled."""
    est = _estimate(MESH_DP4_FSDP64_EP1_TP4, TOKENS_131K_BUCKET, fsdp_is_ep_bound=False)
    assert est.local_tokens == TOKENS_131K_BUCKET // 256
    assert est.dispatch_buffer_bytes == 4_294_967_296  # 4.3 GB measured per core
    assert not est.exceeds_int32_allocation_bound
    assert not est.near_int32_allocation_bound


def test_case_d_chunking_caps_the_buffer():
    """(d) chunk_size=65536 caps both (b) and (c) at chunk*k*H*dtype."""
    chunk = 65536
    capped = chunk * K * H * BF16  # 2,147,483,648
    for mesh, tokens in ((MESH_FSDP64_EP4_TP4, TOKENS_131K_BUCKET), (MESH_DP4_FSDP64_EP1_TP4, TOKENS_131K_BUCKET)):
        est = _estimate(mesh, tokens, fsdp_is_ep_bound=False, chunk_size=chunk)
        assert est.chunk_tokens == chunk
        assert est.dispatch_buffer_bytes == capped
        assert not est.exceeds_int32_allocation_bound
        assert not est.near_int32_allocation_bound


def test_case_e_expert_weight_residency_and_recommendation():
    """(e) ep1/tp4 leaves 403 MB/layer of expert weights per device; recommend fsdp sharding.

    Measured live-array top at ``EasyDeLState.init_tx`` entry (v5p-2048,
    dp4/fsdp64/ep1/tp4): bf16 (256, 2048, 1024) gate_up shards of 268 MB and
    bf16 (256, 512, 2048) down shards of 134 MB per layer per device — 15
    GB/model over 40 layers, which pushed optimizer-state init into
    ``RuntimeBufferAllocationFailure``.
    """
    est = _estimate(MESH_DP4_FSDP64_EP1_TP4, TOKENS_131K_BUCKET, fsdp_is_ep_bound=False)
    # Only ep(=1) * tp(=4) shard the weights: 1.61 GB / 4 = 402.7 MB per layer.
    assert est.expert_weights_bytes_per_device == EXPERT_PARAMS_BYTES_PER_LAYER // 4
    assert est.expert_weights_bytes_per_device == 402_653_184
    # 40 layers -> 16.1 GB/model resident, matching the ~15 GiB probe.
    assert est.expert_weights_bytes_per_device * 40 == 16_106_127_360
    assert est.weight_gather_bytes_per_layer == 0, "flag off: no gather-at-use traffic"
    assert "moe_fsdp_shard_expert_weights" in est.summary, "planner must recommend the flag"

    est_on = _estimate(
        MESH_DP4_FSDP64_EP1_TP4,
        TOKENS_131K_BUCKET,
        fsdp_is_ep_bound=False,
        fsdp_shard_expert_weights=True,
    )
    # fsdp=64 joins the weight sharding: 402.7 MB / 64 = 6.3 MB per layer
    # (~0.25 GB/model at 40 layers — the intended end state).
    assert est_on.expert_weights_bytes_per_device == EXPERT_PARAMS_BYTES_PER_LAYER // (4 * 64)
    assert "recommendation" not in est_on.summary
    # Gather-at-use traffic: the ep/tp-local slice minus the shard already held.
    expected_gather = (EXPERT_PARAMS_BYTES_PER_LAYER // 4) * 63 // 64
    assert est_on.weight_gather_bytes_per_layer == expected_gather


def test_ring_multiplies_dispatch_tokens_by_ep():
    """Ring-of-experts gathers tokens over ep before dispatch (chunking after)."""
    est = _estimate(MESH_FSDP64_EP4_TP4, TOKENS_131K_BUCKET, use_ring_of_experts=True)
    assert est.dispatch_tokens == est.local_tokens * 4
    capped = _estimate(
        MESH_FSDP64_EP4_TP4,
        TOKENS_131K_BUCKET,
        use_ring_of_experts=True,
        chunk_size=65536,
    )
    assert capped.chunk_tokens == 65536
    assert capped.dispatch_buffer_bytes == 65536 * K * H * BF16


def test_chunk_larger_than_stream_is_clamped():
    """chunk >= dispatch tokens degenerates to the unchunked buffer size."""
    est = _estimate(MESH_DP4_FSDP64_EP1_TP4, TOKENS_131K_BUCKET, chunk_size=10**9)
    unchunked = _estimate(MESH_DP4_FSDP64_EP1_TP4, TOKENS_131K_BUCKET, chunk_size=None)
    assert est.chunk_tokens == unchunked.dispatch_tokens
    assert est.dispatch_buffer_bytes == unchunked.dispatch_buffer_bytes


def test_case_f_ep_carries_batch_131k_production_numbers():
    """(f) ep-carries-batch on fsdp64/ep4/tp4, 131k bucket: 256 batch ways, 4.3 GB/dir a2a.

    Batch shards over dp*fsdp*ep = 256 ways -> 131,072 tokens/core. The
    balanced dispatch all-to-all moves tokens_per_core * k * H * bf16 =
    4.3 GB per direction per layer; the worst-case receive buffer is
    ep * balanced = 4x that (min(k=8, E/ep=64) = k, no top-k cap) — 17.2 GB,
    inside the warn band but not the hard int32 bound.
    """
    est = _estimate(MESH_FSDP64_EP4_TP4, TOKENS_131K_BUCKET, ep_carries_batch=True)
    assert est.local_tokens == TOKENS_131K_BUCKET // 256
    assert est.local_tokens == 131_072
    assert est.all_to_all_bytes_per_layer == 131_072 * K * H * BF16
    assert est.all_to_all_bytes_per_layer == 4_294_967_296  # 4.3 GB / direction / layer
    assert est.dispatch_buffer_bytes == 4 * 131_072 * K * H * BF16
    assert est.dispatch_buffer_bytes == 17_179_869_184  # worst-case receive buffer
    assert not est.exceeds_int32_allocation_bound
    assert est.near_int32_allocation_bound
    assert "ep_carries_batch=True" in est.summary


def test_case_f_ep_carries_batch_8k_bucket_numbers():
    """(f) 8k bucket (B=512, S=8192): 16,384 tokens/core, 537 MB/dir a2a, 2.1 GB buffer."""
    est = _estimate(MESH_FSDP64_EP4_TP4, TOKENS_8K_BUCKET, ep_carries_batch=True)
    assert est.local_tokens == TOKENS_8K_BUCKET // 256
    assert est.local_tokens == 16_384
    assert est.all_to_all_bytes_per_layer == 16_384 * K * H * BF16
    assert est.all_to_all_bytes_per_layer == 536_870_912  # 537 MB / direction / layer
    assert est.dispatch_buffer_bytes == 4 * 16_384 * K * H * BF16
    assert est.dispatch_buffer_bytes == 2_147_483_648
    assert not est.exceeds_int32_allocation_bound
    assert not est.near_int32_allocation_bound


def test_ep_carries_batch_chunk_does_not_cap_receive_buffer():
    """Chunking bounds the gmm buffers but NOT the a2a receive buffer (dispatch precedes the scan)."""
    est = _estimate(
        MESH_FSDP64_EP4_TP4,
        TOKENS_131K_BUCKET,
        ep_carries_batch=True,
        chunk_size=65536,
    )
    assert est.chunk_tokens == 65536
    assert est.dispatch_buffer_bytes == 17_179_869_184, (
        "the worst-case a2a receive buffer must not shrink with moe_chunk_size: the dispatch "
        "all-to-all happens before the chunk scan"
    )


def test_ep_carries_batch_recommendation_logic():
    """Planner recommends the knob exactly where it applies (ep>1, non-ring, unbound, off)."""
    applies = _estimate(MESH_FSDP64_EP4_TP4, TOKENS_131K_BUCKET)
    assert "moe_ep_carries_batch" in applies.summary

    active = _estimate(MESH_FSDP64_EP4_TP4, TOKENS_131K_BUCKET, ep_carries_batch=True)
    assert "set moe_ep_carries_batch=True" not in active.summary

    ep1 = _estimate(MESH_DP4_FSDP64_EP1_TP4, TOKENS_131K_BUCKET)
    assert "moe_ep_carries_batch" not in ep1.summary

    bound = _estimate(MESH_FSDP64_EP4_TP4, TOKENS_8K_BUCKET, fsdp_is_ep_bound=True)
    assert "moe_ep_carries_batch" not in bound.summary

    ring = _estimate(MESH_FSDP64_EP4_TP4, TOKENS_8K_BUCKET, use_ring_of_experts=True)
    assert "moe_ep_carries_batch" not in ring.summary


def test_ep_carries_batch_flag_ignored_when_inapplicable():
    """The flag is inert on ep=1 / ring / ep-bound layouts (no silent batch-way change)."""
    for kw in (
        dict(mesh=MESH_DP4_FSDP64_EP1_TP4, extra={}),
        dict(mesh=MESH_FSDP64_EP4_TP4, extra={"use_ring_of_experts": True}),
        dict(mesh=MESH_FSDP64_EP4_TP4, extra={"fsdp_is_ep_bound": True}),
    ):
        base = _estimate(kw["mesh"], TOKENS_8K_BUCKET, **kw["extra"])
        flagged = _estimate(kw["mesh"], TOKENS_8K_BUCKET, ep_carries_batch=True, **kw["extra"])
        assert flagged.local_tokens == base.local_tokens
        assert flagged.dispatch_buffer_bytes == base.dispatch_buffer_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
