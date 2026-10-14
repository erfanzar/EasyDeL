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

"""Expert-load recording, hooked into the one function every MoE family uses.

``permute`` is the shared fused-MoE entry point, so recording there covers
deepseek-v2/v3/v4, qwen3-moe, mixtral, gpt-oss, glm4-moe and every other
registered family without per-model code. The properties worth pinning:

* it is **inert unless a scope is open** — the hook must not perturb outputs or
  leave anything in the graph, or it could not be shipped enabled-by-default in
  the hot path;
* the histogram must equal the routing that actually happened, in GLOBAL expert
  order even when the ring path rotates expert ids;
* the shard reductions must reflect the runtime's real ownership rule
  (contiguous blocks by integer division).

The placement statistics also encode a measured fact: grouped-matmul cost tracks
NON-EMPTY EXPERTS, not tokens (v5p, int4 expert path: a shard's time is flat
within 0.1% from 0.85x to 2.0x token load at fixed buffer height, but scales
near-linearly with experts touched). So both reductions are reported, and a test
pins that they can disagree — that difference is the whole reason token-balancing
placement was not worth building.
"""

import jax
import numpy as np
import pytest
from easydel.layers.moe import (
    balancedness,
    optimal_shard_loads,
    record_expert_load,
    shard_active_experts,
    shard_loads,
)
from easydel.layers.moe._communication_utils import permute
from jax import numpy as jnp

HIDDEN = 8
N_EXPERTS = 8
TOP_K = 2


def _fixed_select(expert_ids):
    """A select_hook that routes every token to `expert_ids`."""

    def hook(gate_logits, pre_bias_logits, k):
        del pre_bias_logits
        tokens = gate_logits.shape[0]
        idx = jnp.tile(jnp.asarray(expert_ids, dtype=jnp.int32)[:k], (tokens, 1))
        return jnp.ones((tokens, k), dtype=gate_logits.dtype), idx

    return hook


def _run_permute(tokens=4, select_hook=None, roll=None, layer_idx=None, num_experts=N_EXPERTS):
    inputs = jnp.arange(tokens * HIDDEN, dtype=jnp.float32).reshape(1, tokens, HIDDEN)
    gate_logits = jnp.zeros((tokens, num_experts), dtype=jnp.float32)
    return permute(
        inputs=inputs,
        gate_logits=gate_logits,
        num_experts_per_tok=TOP_K,
        num_experts=num_experts,
        dtype=jnp.float32,
        select_hook=select_hook,
        roll_to_expert_id=roll,
        layer_idx=layer_idx,
    )


def test_recorder_is_inert_when_no_scope_is_open():
    """The hook must be free and invisible outside a recording scope.

    This is the property that makes it safe to leave in the fused hot path: no
    records, and byte-identical outputs.
    """
    baseline = _run_permute(select_hook=_fixed_select([1, 2]))
    again = _run_permute(select_hook=_fixed_select([1, 2]))
    for a, b in zip(baseline, again, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    with record_expert_load(ep_size=2) as rec:
        pass
    assert rec.records == []


def test_recorded_histogram_matches_the_routing_that_happened():
    """Counts must equal a hand-computed bincount of the selections."""
    tokens = 5
    with record_expert_load(ep_size=2) as rec:
        _run_permute(tokens=tokens, select_hook=_fixed_select([3, 6]))

    assert rec.records, "no histogram captured"
    counts = rec.records[0].counts
    expected = np.zeros(N_EXPERTS, dtype=np.int64)
    expected[3] = tokens
    expected[6] = tokens
    np.testing.assert_array_equal(counts, expected)
    assert counts.sum() == tokens * TOP_K


def test_histogram_is_reported_in_global_expert_order_under_roll():
    """The ring path rotates expert ids; records must still be global.

    ``roll_to_expert_id`` makes a shard's experts land at local ids
    ``[0, experts_per_shard)``. A record left in rolled order would attribute
    load to the wrong experts, silently, and only on the ring path.
    """
    roll = 4
    with record_expert_load(ep_size=2) as rec:
        _run_permute(select_hook=_fixed_select([5, 6]), roll=roll)

    counts = rec.records[0].counts
    assert int(np.argmax(counts)) in (5, 6), f"expected global ids 5/6, got argmax={np.argmax(counts)}"
    assert counts[5] > 0 and counts[6] > 0


def test_layer_index_and_regime_are_recorded():
    """Per-layer separation, and prefill/decode tagged from the ambient scope."""
    from easydel.infra.sharding import decode_mode_specs

    with record_expert_load(ep_size=2) as rec:
        _run_permute(select_hook=_fixed_select([0, 1]), layer_idx=0)
        _run_permute(select_hook=_fixed_select([2, 3]), layer_idx=7)
        with decode_mode_specs(True):
            _run_permute(select_hook=_fixed_select([0, 1]), layer_idx=0)

    assert {r.layer_idx for r in rec.records} == {0, 7}
    assert {r.regime for r in rec.records} == {"prefill", "decode"}
    per_layer = rec.stack(regime="prefill")
    assert set(per_layer) == {0, 7}
    # layer 0 saw experts 0/1, layer 7 saw 2/3 -- no bleed between them
    assert per_layer[0][0] > 0 and per_layer[0][2] == 0
    assert per_layer[7][2] > 0 and per_layer[7][0] == 0


def test_shard_reductions_follow_the_runtime_ownership_rule():
    """Shard d owns experts [d*E/ep, (d+1)*E/ep) -- integer division, no table."""
    counts = np.array([10, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    np.testing.assert_array_equal(shard_loads(counts, 2), np.array([10.0, 4.0]))
    np.testing.assert_array_equal(shard_active_experts(counts, 2), np.array([1.0, 4.0]))
    assert balancedness(counts, 2) == pytest.approx(7.0 / 10.0, rel=1e-3)


def test_optimal_placement_keeps_shard_cardinality_equal():
    """Balancing may not change how many experts a shard holds.

    Shard shapes are static, so the placement problem is number partitioning
    with an exact-cardinality constraint -- the same constraint SGLang's
    ``balanced_packing`` enforces. A solver that ignores it would report
    unreachable gains.
    """
    counts = np.array([100, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    loads = optimal_shard_loads(counts, 2)
    assert loads.sum() == counts.sum()
    # 4 experts each: the hot one cannot be isolated, so imbalance survives.
    assert loads.max() >= 100
    assert len(loads) == 2


def test_token_balance_and_active_experts_can_disagree():
    """Perfectly balanced tokens, wildly unbalanced active experts.

    This is the shape of the measured DeepSeek-V4 result and the reason
    token-balancing placement was not worth building: cost follows the second
    number, which the first cannot see.
    """
    counts = np.array([8, 8, 8, 8, 32, 0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(shard_loads(counts, 2), np.array([32.0, 32.0]))
    np.testing.assert_array_equal(shard_active_experts(counts, 2), np.array([4.0, 1.0]))
    assert balancedness(counts, 2) == pytest.approx(1.0, rel=1e-3)


def test_summary_reports_both_cost_models():
    with record_expert_load(ep_size=2) as rec:
        _run_permute(select_hook=_fixed_select([0, 5]), layer_idx=0)
    s = rec.summary()
    for key in (
        "per_layer_max_over_mean",
        "moe_time_reduction_pct_if_token_bound",
        "active_experts_per_shard_mean",
        "active_experts_per_shard_max_over_mean",
        "ep_size",
    ):
        assert key in s, f"missing {key}"
    assert s["n_experts"] == N_EXPERTS


def test_export_uses_the_eplb_interchange_keys():
    """Dumps carry SGLang's key names so they feed existing EPLB tooling."""
    with record_expert_load(ep_size=2) as rec:
        _run_permute(select_hook=_fixed_select([1, 2]), layer_idx=0)
        rec.mark_step()
        _run_permute(select_hook=_fixed_select([1, 2]), layer_idx=0)

    arrays = rec.to_arrays()
    assert set(arrays) == {"logical_count", "physical_to_logical_map"}
    steps, layers, experts = arrays["logical_count"].shape
    assert (steps, layers, experts) == (2, 1, N_EXPERTS)  # step axis preserved
    assert arrays["physical_to_logical_map"].shape == (1, N_EXPERTS)


def test_recorder_works_under_jit():
    """The hook rides a debug callback, so it must survive tracing."""

    @jax.jit
    def go(x, logits):
        return permute(
            inputs=x,
            gate_logits=logits,
            num_experts_per_tok=TOP_K,
            num_experts=N_EXPERTS,
            dtype=jnp.float32,
            select_hook=_fixed_select([2, 4]),
            layer_idx=3,
        )[3]

    x = jnp.ones((1, 4, HIDDEN), dtype=jnp.float32)
    logits = jnp.zeros((4, N_EXPERTS), dtype=jnp.float32)
    with record_expert_load(ep_size=2) as rec:
        jax.block_until_ready(go(x, logits))

    assert rec.records, "nothing captured under jit"
    counts = rec.records[0].counts
    assert counts[2] == 4 and counts[4] == 4
