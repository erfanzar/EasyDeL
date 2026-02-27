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

import jax.numpy as jnp
import pytest

from easydel.inference.esurge.config import CacheConfig, Config, SchedulerConfig
from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.evaluations.esurge_eval import eSurgeLMEvalAdapter
from easydel.inference.sampling_params import SamplingParams


class _DummyProcessor:
    pad_token_id = None
    eos_token_id = 1
    padding_side = "right"


class _DummySurge:
    max_num_seqs = 2

    class _Runner:
        model = None

    runner = _Runner()


def test_esurge_eval_rejects_non_iterable_math_hint_input(monkeypatch):
    monkeypatch.setattr(eSurgeLMEvalAdapter, "_setup", lambda self: None)

    with pytest.raises(TypeError, match="math_answer_task_hints"):
        eSurgeLMEvalAdapter(
            _DummySurge(),
            _DummyProcessor(),
            math_answer_task_hints=False,
        )


def test_esurge_eval_accepts_single_string_math_hint(monkeypatch):
    monkeypatch.setattr(eSurgeLMEvalAdapter, "_setup", lambda self: None)

    adapter = eSurgeLMEvalAdapter(
        _DummySurge(),
        _DummyProcessor(),
        math_answer_task_hints="gsm8k",
    )

    assert adapter.math_answer_task_hints == ("gsm8k",)


def test_scheduler_falls_back_to_model_len_when_batch_token_limit_is_none():
    config = Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=None,
            max_model_len=128,
            token_safety_margin=None,
        ),
        cache_config=CacheConfig(num_pages=16, page_size=8, enable_prefix_caching=False),
    )
    kv_cache_config = CacheGroupsConfig(
        num_pages=16,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=8,
                    num_kv_heads=1,
                    head_size=4,
                    dtype=jnp.float32,
                    use_mla=False,
                ),
                layer_names=None,
            )
        ],
    )

    scheduler = Scheduler(config=config, kv_cache_config=kv_cache_config)

    assert scheduler.max_num_scheduled_tokens == 128
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == 0


def test_scheduler_aborts_empty_prompt_request_instead_of_requeueing():
    config = Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_model_len=128,
            token_safety_margin=None,
        ),
        cache_config=CacheConfig(num_pages=16, page_size=8, enable_prefix_caching=False),
    )
    kv_cache_config = CacheGroupsConfig(
        num_pages=16,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=8,
                    num_kv_heads=1,
                    head_size=4,
                    dtype=jnp.float32,
                    use_mla=False,
                ),
                layer_names=None,
            )
        ],
    )

    scheduler = Scheduler(config=config, kv_cache_config=kv_cache_config)
    request = EngineRequest(
        request_id="req-empty-prompt",
        prompt_token_ids=[],
        sampling_params=SamplingParams(max_tokens=8),
        eos_token_id=1,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert "req-empty-prompt" in output.finished_req_ids
    assert request.status == EngineRequestStatus.FINISHED_ABORTED
    assert scheduler.get_num_unfinished_requests() == 0
    assert "req-empty-prompt" not in scheduler.requests
