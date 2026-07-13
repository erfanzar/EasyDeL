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

"""Regression test for DP prefix-cache rank probing (bug S2).

Choosing a DP rank for a new request probes every rank's prefix cache. The
probe went through ``get_computed_pages``, which persists page hashes into
``req_to_page_hashes``. Only the chosen (owning) rank ever frees that entry, so
the other ``dp_size - 1`` ranks leaked one ``req_to_page_hashes`` entry per
request forever. The probe must be read-only: a rank must not retain page-hash
state for requests it never owns.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler.dp_scheduler import DPScheduler
from easydel.inference.sampling_params import SamplingParams


def _kv_config(*, num_pages: int, page_size: int) -> CacheGroupsConfig:
    return CacheGroupsConfig(
        num_pages=num_pages,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=page_size,
                    num_kv_heads=1,
                    head_size=4,
                    dtype=jnp.float32,
                    use_mla=False,
                ),
                layer_names=None,
            )
        ],
    )


def _request(rid: str, prompt_len: int) -> EngineRequest:
    return EngineRequest(
        request_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=8),
        eos_token_id=1,
    )


def test_rank_probe_does_not_leak_page_hashes_on_non_owning_ranks():
    dp_size = 2
    num_pages = 64
    page_size = 4
    scheduler = DPScheduler(
        kv_cache_config=_kv_config(num_pages=num_pages, page_size=page_size),
        dp_size=dp_size,
        max_num_seqs=8,
        max_num_batched_tokens=64,
        max_model_len=256,
        num_pages=num_pages,
        page_size=page_size,
        enable_prefix_caching=True,
        async_scheduling=False,
        use_worker_processes=False,  # in-process so we can inspect rank state
    )
    try:
        # Prompts long enough to hash multiple pages (page_size=4).
        for i in range(6):
            scheduler.add_request(_request(f"r{i}", prompt_len=8))

        for rank, rank_scheduler in enumerate(scheduler.schedulers):
            owned = {rid for rid, owner in scheduler.req_id_to_dp_rank.items() if owner == rank}
            hashed = set(rank_scheduler.kv_cache_manager.req_to_page_hashes.keys())
            leaked = hashed - owned
            assert not leaked, (
                f"rank {rank} retained prefix-hash entries for requests it does not own: {sorted(leaked)}"
            )
    finally:
        scheduler.shutdown()
