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

"""Host-only DP worker RPCs must not mutate the parent process environment."""

import multiprocessing
import os
import threading

import pytest
from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.scheduler.dp_scheduler import DPScheduler

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("start_method", multiprocessing.get_all_start_methods())
def test_worker_rpc_does_not_change_parent_environment(monkeypatch, start_method) -> None:
    monkeypatch.setenv("ENABLE_DISTRIBUTED_INIT", "1")
    observed = set()
    done = threading.Event()

    def observe():
        while not done.wait(0.001):
            observed.add(os.environ.get("ENABLE_DISTRIBUTED_INIT"))

    observer = threading.Thread(target=observe)
    observer.start()
    scheduler = None
    try:
        scheduler = DPScheduler(
            kv_cache_config=CacheGroupsConfig(
                num_pages=16,
                kv_cache_groups=[
                    CacheGroupSpec(
                        kv_cache_spec=FullAttentionSpec(
                            page_size=4, num_kv_heads=1, head_size=4, dtype="float32", use_mla=False
                        ),
                        layer_names=None,
                    )
                ],
            ),
            dp_size=2,
            max_num_seqs=1,
            max_num_batched_tokens=4,
            max_model_len=16,
            num_pages=16,
            page_size=4,
            enable_prefix_caching=False,
            async_scheduling=False,
            worker_start_method=start_method,
        )
        assert scheduler.get_request_counts() == (0, 0)
        assert scheduler.schedule().total_num_scheduled_tokens == 0
        assert os.environ["ENABLE_DISTRIBUTED_INIT"] == "1"
    finally:
        if scheduler is not None:
            scheduler.shutdown()
        done.set()
        observer.join(timeout=2)
    assert observed == {"1"}
