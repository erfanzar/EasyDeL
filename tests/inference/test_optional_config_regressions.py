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

import jax
import jax.numpy as jnp
import pytest
from flax import nnx as nn

import easydel as ed
from easydel.inference.esurge.config import CacheConfig, Config, SchedulerConfig
from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.evaluations.esurge_eval import eSurgeLMEvalAdapter
from easydel.inference.sampling_params import SamplingParams
from easydel.infra.utils import AttnMaskType


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


def test_linear_attention_mask_type_is_accepted():
    assert AttnMaskType.from_hf("linear_attention") == AttnMaskType.FULL


def test_create_mesh_normalizes_multi_device_axis_dims_on_single_device(monkeypatch):
    captured = {}

    def _fake_create_mesh(*, axis_dims, **kwargs):
        captured["axis_dims"] = axis_dims
        return object()

    monkeypatch.setattr(jax, "device_count", lambda backend=None: 1)
    monkeypatch.setattr("eformer.escale.create_mesh", _fake_create_mesh)

    ed.EasyDeLBaseConfig.create_mesh(sharding_axis_dims=(1, 4, 1, -1, 1))

    assert captured["axis_dims"] == (1, 1, 1, -1, 1)


def _build_tiny_qwen35(attn_mechanism: str):
    config = ed.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        head_dim=16,
        attn_mechanism=attn_mechanism,
    )
    with config.mesh:
        model = ed.Qwen3_5ForCausalLM.lazy_init(
            config=config,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=nn.Rngs(0),
        )
    return model


def _build_tiny_qwen35_vlm(attn_mechanism: str):
    text_config = ed.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        head_dim=16,
        attn_mechanism=attn_mechanism,
        rope_scaling={"rope_type": "default", "mrope_section": [8, 4, 4], "mrope_interleaved": True},
        partial_rotary_factor=0.5,
    )
    vision_config = ed.Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=text_config.hidden_size,
        num_position_embeddings=128,
        deepstack_visual_indexes=[],
    )
    config = ed.Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=text_config.vocab_size - 1,
        video_token_id=text_config.vocab_size - 2,
        vision_start_token_id=text_config.vocab_size - 3,
        vision_end_token_id=text_config.vocab_size - 4,
    )
    with config.mesh:
        model = ed.Qwen3_5ForConditionalGeneration.lazy_init(
            config=config,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=nn.Rngs(0),
        )
    return model


def test_esurge_compatible_model_forces_ragged_on_tpu(monkeypatch):
    model = _build_tiny_qwen35("unified_attention")
    import jax

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    compatible = model.esurge_compatible_model
    assert compatible.config.get_text_config().attn_mechanism == "ragged_page_attention_v3"


def test_esurge_compatible_model_forces_unified_on_gpu(monkeypatch):
    model = _build_tiny_qwen35("sdpa")
    import jax

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    compatible = model.esurge_compatible_model
    assert compatible.config.get_text_config().attn_mechanism == "unified_attention"


def test_esurge_compatible_model_updates_vlm_text_attn_recursively():
    import jax

    backend = jax.default_backend()
    if backend == "gpu":
        source_attn = "sdpa"
        expected_attn = "unified_attention"
    else:
        source_attn = "unified_attention"
        expected_attn = "ragged_page_attention_v3"

    model = _build_tiny_qwen35_vlm(source_attn)
    compatible = model.esurge_compatible_model
    assert compatible.config.get_text_config().attn_mechanism == expected_attn
