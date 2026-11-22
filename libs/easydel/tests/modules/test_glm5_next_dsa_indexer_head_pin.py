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

"""The indexer head pin on the replicated fused top-k call must not change results.

Steps whose token count does not divide the mesh size (every step of this schedule: 25, 11, 14 and 5 tokens on four
devices) run the fused top-k as one replicated ``shard_map``. The serving cache view pins the indexer query and head
weights head-sharded before that call, so XLA all-gathers the small activation instead of the query projection
weight. One GLM-5 DSA layer runs the multi-step ragged serving schedule of ``test_glm5_next_dsa_ep_split_serving`` with
the pin and with it disabled; outputs and caches must be bitwise equal, and the pinned program must really carry the
extra constraints.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from easydel.caching import MLARaggedPagesCacheView

from .test_glm5_next_dsa_ep_split_serving import _layer, _run


@pytest.mark.parametrize("dims", [(1, 1, 1, 2, 2, 1)], ids=["ep2tp2"])
def test_replicated_pool_topk_head_pin_is_bitwise(dims, monkeypatch):
    if jax.default_backend() != "tpu":
        pytest.skip("the fused top-k shard_map dispatch is TPU-only")
    if jax.device_count() < int(np.prod(dims)):
        pytest.skip(f"needs {int(np.prod(dims))} devices")
    config, layer = _layer(dims)
    pinned_counts = _run(config, layer, None, trace_only=True)
    pinned, pinned_cache, _ = _run(config, layer, None)
    monkeypatch.setattr(MLARaggedPagesCacheView, "constrain_indexer_heads", lambda self, x: x)
    plain_counts = _run(config, layer, None, trace_only=True)
    plain, plain_cache, _ = _run(config, layer, None)
    # Guard against a vacuous comparison: the pinned program constrains the query and the head weights.
    assert pinned_counts["sharding_constraint"] > plain_counts["sharding_constraint"], (
        pinned_counts["sharding_constraint"],
        plain_counts["sharding_constraint"],
    )
    for step, (a, b) in enumerate(zip(pinned, plain, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"step {step}")
    for a, b in zip(pinned_cache, plain_cache, strict=True):
        np.testing.assert_array_equal(a, b)
