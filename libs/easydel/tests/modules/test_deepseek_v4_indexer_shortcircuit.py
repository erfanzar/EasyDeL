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

"""The DSA indexer's ``top_k`` is a no-op when it cannot exclude anything.

Entries number ``max_length // rate``, so at ``max_model_len=2048`` every
DeepSeek-V4 layer has at most as many compressed entries as ``index_topk``:
compressed_sparse rate 4 gives 512 entries against ``index_topk`` 512, and
heavily_compressed rate 128 gives 16. ``top_k`` then returns *every* slot,
merely reordered by score -- and the consumer ``_indexer_opened_bias`` scatters
those indices into a position-indexed bias, where order cannot matter.

Profiling a cc32 decode step measured ``sort`` at 3.0 ms of 40.3 ms, before
counting the scoring matmul that feeds it. The decode path now short-circuits
to the visibility mask when ``index_topk >= n_slots``.

The claim being pinned here is an equivalence, so it is tested against the real
consumer rather than by asserting the shortcut equals itself -- and the last
test guards against the equivalence being vacuously true.
"""

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest
from easydel.modules.deepseek_v4.modeling_deepseek_v4 import _indexer_opened_bias
from jax import numpy as jnp

BATCH, SEQ = 3, 1


def _topk_path(scores, visible, k):
    """What the indexer computes when it really runs the selection."""
    masked = jnp.where(visible, scores, -jnp.inf)
    idx = jax.lax.top_k(masked, k)[1]
    picked = jnp.take_along_axis(visible, idx, axis=-1)
    return jnp.where(picked, idx, -1)


def _shortcircuit_path(visible, n_slots):
    """What the indexer computes when top_k cannot exclude anything."""
    entry_ids = jnp.broadcast_to(jnp.arange(n_slots, dtype=jnp.int32), visible.shape)
    return jnp.where(visible, entry_ids, -1)


def _bias(indices, n_slots):
    return np.asarray(_indexer_opened_bias(indices, BATCH, SEQ, n_slots), np.float32)


@pytest.mark.parametrize("n_slots", [16, 64, 512])
def test_shortcircuit_bias_is_identical_when_topk_cannot_exclude(n_slots):
    """k == n_slots: the two index sets must produce the same opened bias."""
    rng = np.random.default_rng(n_slots)
    scores = jnp.asarray(rng.standard_normal((BATCH, SEQ, n_slots)), jnp.float32)
    visible = jnp.asarray(rng.random((BATCH, SEQ, n_slots)) > 0.35)

    got = _bias(_shortcircuit_path(visible, n_slots), n_slots)
    want = _bias(_topk_path(scores, visible, n_slots), n_slots)
    np.testing.assert_array_equal(got, want)


def test_shortcircuit_holds_for_every_visibility_pattern():
    """Exhaustive over all 2^8 masks -- no reliance on a lucky random draw."""
    n_slots = 8
    rng = np.random.default_rng(0)
    scores = jnp.asarray(rng.standard_normal((BATCH, SEQ, n_slots)), jnp.float32)
    for pattern in range(1 << n_slots):
        bits = [(pattern >> i) & 1 for i in range(n_slots)]
        visible = jnp.asarray(np.tile(np.array(bits, bool), (BATCH, SEQ, 1)))
        got = _bias(_shortcircuit_path(visible, n_slots), n_slots)
        want = _bias(_topk_path(scores, visible, n_slots), n_slots)
        np.testing.assert_array_equal(got, want, err_msg=f"pattern {pattern:08b}")


def test_all_hidden_entries_stay_masked():
    """Nothing visible -> the bias must open nothing (not everything)."""
    n_slots = 32
    visible = jnp.zeros((BATCH, SEQ, n_slots), bool)
    bias = _bias(_shortcircuit_path(visible, n_slots), n_slots)
    assert np.all(bias < 0), "hidden entries were opened -- attention would see the future"


def test_equivalence_is_not_vacuous():
    """With k < n_slots the paths MUST differ, or the test above proves nothing.

    If the shortcut matched the real selection even when top_k genuinely
    filters, the comparison would be insensitive and could not catch a
    regression.
    """
    n_slots, k = 32, 8
    rng = np.random.default_rng(7)
    scores = jnp.asarray(rng.standard_normal((BATCH, SEQ, n_slots)), jnp.float32)
    visible = jnp.ones((BATCH, SEQ, n_slots), bool)

    filtered = _bias(_topk_path(scores, visible, k), n_slots)
    everything = _bias(_shortcircuit_path(visible, n_slots), n_slots)
    assert not np.array_equal(filtered, everything), "top_k with k<n did not actually exclude anything"
    assert int((filtered[0, 0, 0] == 0.0).sum()) == k, "expected exactly k opened entries"


def test_end_to_end_model_matches_with_and_without_the_shortcut():
    """The real model must produce identical logits either way.

    The unit tests above pin the index sets; this pins the whole model, which is
    what actually ships. The shortcut is selected by a static predicate, so the
    full path is restored by patching that predicate to ``False`` -- otherwise
    there is no way to exercise both branches on one configuration.

    The config is chosen so the shortcut WOULD fire (index_topk >= n_slots):
    max_length 64 with compress rate 4 gives 16 entries, against index_topk 32.
    """
    import easydel as ed
    from easydel.modules.deepseek_v4 import modeling_deepseek_v4 as mod
    from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config

    config = DeepseekV4Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        max_position_embeddings=256,
        sliding_window=16,
        index_topk=32,
        index_n_heads=4,
        index_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        head_dim=32,
        o_groups=2,
    )
    # Pin a single-device mesh: the default fills ep=-1 with every visible
    # device, which is ep=8 under the CPU trio against 4 routed experts, and
    # the fused-MoE shard_map rejects the indivisible expert axis.
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    model = ed.AutoEasyDeLModelForCausalLM.from_config(config, dtype=jnp.float32, param_dtype=jnp.float32)

    ids = jnp.asarray([[3, 9, 5, 1]], dtype=jnp.int32)
    pos = jnp.full((1, 4), 5, dtype=jnp.int32)

    class _Meta:
        query_start_loc = jnp.arange(5, dtype=jnp.int32)
        num_seqs = jnp.asarray([4], dtype=jnp.int32)
        recurrent_state_indices = None

    def run():
        cache = model.init_cache(batch_size=4, max_length=64)
        out = model(input_ids=ids, position_ids=pos, past_key_values=cache, cache_metadata=_Meta())
        return np.asarray(out.logits, np.float32)

    original = mod._indexer_selection_is_vacuous
    assert original(config.index_topk, 16), "config chosen so the shortcut fires; it did not"
    with_shortcut = run()
    try:
        mod._indexer_selection_is_vacuous = lambda *_a, **_k: False  # force the full indexer
        full_path = run()
    finally:
        mod._indexer_selection_is_vacuous = original

    assert np.all(np.isfinite(with_shortcut))
    delta = float(np.max(np.abs(with_shortcut - full_path)))
    assert delta < 1e-4, f"skipping the dead indexer changed model output, max|delta|={delta:.3e}"


# ---------------------------------------------------------------------------
# The same argument, made dynamic.
#
# The shortcut above is a *static* one: it fires only when `index_topk` is at
# least `n_slots`, which at a 262,144 window it never is (65,536 entries against
# `index_topk` 512). But the entries that exist at a given decode step are only
# the ones the sequence has produced so far. While that live prefix is no longer
# than `index_topk`, `top_k` again cannot exclude anything, and the scoring
# matmul -- which reads the whole entry buffer, priced by `cost_analysis` at
# 1.510 GiB per decode step -- is computed only to be discarded.
#
# What is pinned here is that equivalence, against the real consumer, plus its
# non-vacuity. The buffer state update is deliberately *not* skipped along with
# the scoring: entries must still be written or later steps read a stale buffer.
# ---------------------------------------------------------------------------


def _live_prefix_path(visible, k):
    """What the indexer computes while the live prefix fits inside top_k."""
    idx = jnp.broadcast_to(jnp.arange(k, dtype=jnp.int32), (*visible.shape[:-1], k))
    return jnp.where(visible[..., :k], idx, -1)


@pytest.mark.parametrize(("n_slots", "k", "live"), [(512, 64, 1), (512, 64, 63), (512, 64, 64), (4096, 512, 300)])
def test_live_prefix_matches_scored_topk(n_slots, k, live):
    """live <= k: ranking cannot exclude, so the prefix must open the same bias."""
    rng = np.random.default_rng(live)
    scores = jnp.asarray(rng.standard_normal((BATCH, SEQ, n_slots)), jnp.float32)
    # only the first `live` entries exist yet
    visible = jnp.asarray(np.arange(n_slots)[None, None, :] < live).repeat(BATCH, 0).repeat(SEQ, 1)

    got = _bias(_live_prefix_path(visible, k), n_slots)
    want = _bias(_topk_path(scores, visible, k), n_slots)
    np.testing.assert_array_equal(got, want)


def test_live_prefix_equivalence_is_not_vacuous():
    """live > k must genuinely differ, or the test above proves nothing."""
    n_slots, k, live = 512, 64, 200
    rng = np.random.default_rng(7)
    scores = jnp.asarray(rng.standard_normal((BATCH, SEQ, n_slots)), jnp.float32)
    visible = jnp.asarray(np.arange(n_slots)[None, None, :] < live).repeat(BATCH, 0).repeat(SEQ, 1)

    got = _bias(_live_prefix_path(visible, k), n_slots)
    want = _bias(_topk_path(scores, visible, k), n_slots)
    assert not np.array_equal(got, want), "scoring excluded nothing -- the gate would be untested"


def test_indexer_opened_bias_score_proxy_is_primal_exact_and_differentiable():
    indices = jnp.array([[[0, 2]]], jnp.int32)
    plain = _indexer_opened_bias(indices, 1, 1, 4)
    scores = jnp.array([[[0.2, -0.1, 0.7, 0.3]]], jnp.float32)

    def loss(s):
        bias = _indexer_opened_bias(indices, 1, 1, 4, score_proxy=s)
        assert jnp.array_equal(bias, plain)
        probs = jax.nn.softmax(bias[:, 0], axis=-1)
        return jnp.sum(probs * jnp.arange(4, dtype=jnp.float32))

    grad = jax.grad(loss)(scores)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(jnp.abs(grad) > 0)
