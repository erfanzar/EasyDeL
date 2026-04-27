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

"""Comprehensive tests for elarge builders module."""

from easydel.infra.elarge.builders import to_esurge_kwargs, to_from_pretrained_kwargs
from easydel.infra.elarge.processing import normalize


class TestToFromPretrainedKwargs:
    def test_minimal_config(self):
        cfg = {"model": {"name_or_path": "test-model"}}
        result = to_from_pretrained_kwargs(cfg)
        assert result["pretrained_model_name_or_path"] == "test-model"
        assert result["auto_shard_model"] is True

    def test_loader_settings(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "loader": {"verbose": False, "from_torch": True, "trust_remote_code": True},
        }
        result = to_from_pretrained_kwargs(cfg)
        assert result["verbose"] is False
        assert result["from_torch"] is True
        assert result["trust_remote_code"] is True

    def test_sharding_defaults(self):
        cfg = {"model": {"name_or_path": "test-model"}}
        result = to_from_pretrained_kwargs(cfg)
        assert result["sharding_axis_dims"] == (1, 1, 1, 1, -1, 1)
        assert result["sharding_axis_names"] == ("pp", "dp", "fsdp", "ep", "tp", "sp")

    def test_custom_sharding(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "sharding": {"axis_dims": [1, 2, 1, 1, 4, 1], "auto_shard_model": False},
        }
        result = to_from_pretrained_kwargs(cfg)
        assert result["sharding_axis_dims"] == (1, 2, 1, 1, 4, 1)
        assert result["auto_shard_model"] is False

    def test_dcn_axis_dims(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "sharding": {"dcn_axis_dims": [1, 2, 2, 1, 1, 1]},
        }
        result = to_from_pretrained_kwargs(cfg)
        assert result["sharding_dcn_axis_dims"] == (1, 2, 2, 1, 1, 1)

    def test_dcn_axis_dims_none(self):
        cfg = {"model": {"name_or_path": "test-model"}}
        result = to_from_pretrained_kwargs(cfg)
        assert result["sharding_dcn_axis_dims"] is None

    def test_quantization_config(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "quantization": {"model": {"dtype": "int8"}},
        }
        result = to_from_pretrained_kwargs(cfg)
        assert result["quantization_config"] is not None

    def test_no_quantization(self):
        cfg = {"model": {"name_or_path": "test-model"}}
        result = to_from_pretrained_kwargs(cfg)
        assert result["quantization_config"] is None
        assert result["apply_quantization"] is False

    def test_extra_kwargs_merged(self):
        cfg = {
            "model": {"name_or_path": "test-model", "extra_kwargs": {"custom_param": 42}},
        }
        result = to_from_pretrained_kwargs(cfg)
        assert result["custom_param"] == 42

    def test_extra_kwargs_none(self):
        cfg = {"model": {"name_or_path": "test-model", "extra_kwargs": None}}
        result = to_from_pretrained_kwargs(cfg)
        assert "pretrained_model_name_or_path" in result

    def test_platform_settings(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "platform": {"backend": "gpu", "platform": "cuda"},
        }
        result = to_from_pretrained_kwargs(cfg)
        assert result["backend"] == "gpu"
        assert result["platform"] == "cuda"


class TestToEsurgeKwargs:
    def test_minimal_config(self):
        cfg = {"model": {"name_or_path": "test-model"}}
        result = to_esurge_kwargs(cfg)
        assert "max_model_len" in result
        assert "max_num_seqs" in result

    def test_default_values(self):
        cfg = {"model": {"name_or_path": "test-model"}}
        result = to_esurge_kwargs(cfg)
        assert result["max_num_seqs"] == 32
        assert result["hbm_utilization"] == 0.80
        assert result["page_size"] == 128
        assert result["use_aot_forward"] is True
        assert result["enable_prefix_caching"] is True
        assert result["compile_runner"] is True
        assert result["auto_truncate_prompt"] is True

    def test_custom_esurge_settings(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {
                "max_num_seqs": 64,
                "hbm_utilization": 0.9,
                "page_size": 64,
                "max_num_batched_tokens": 8192,
            },
        }
        result = to_esurge_kwargs(cfg)
        assert result["max_num_seqs"] == 64
        assert result["hbm_utilization"] == 0.9
        assert result["page_size"] == 64
        assert result["max_num_batched_tokens"] == 8192

    def test_extra_stops_list(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {"extra_stops": ["<|stop|>", "<|end|>"]},
        }
        result = to_esurge_kwargs(cfg)
        assert result["extra_stops"] == ["<|stop|>", "<|end|>"]

    def test_extra_stops_non_string_coerced(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {"extra_stops": 42},
        }
        result = to_esurge_kwargs(cfg)
        assert result["extra_stops"] == ["42"]

    def test_extra_stops_none(self):
        cfg = {"model": {"name_or_path": "test-model"}}
        result = to_esurge_kwargs(cfg)
        assert result["extra_stops"] is None

    def test_extra_eos_token_ids(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {"extra_eos_token_ids": [50256, 50257]},
        }
        result = to_esurge_kwargs(cfg)
        assert result["extra_eos_token_ids"] == [50256, 50257]

    def test_idle_reset_seconds(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {"idle_reset_seconds": 300},
        }
        result = to_esurge_kwargs(cfg)
        assert result["idle_reset_seconds"] == 300.0

    def test_worker_startup_timeout(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {"worker_startup_timeout": 30},
        }
        result = to_esurge_kwargs(cfg)
        assert result["worker_startup_timeout"] == 30.0

    def test_truncate_mode(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {"truncate_mode": "middle"},
        }
        result = to_esurge_kwargs(cfg)
        assert result["truncate_mode"] == "middle"

    def test_distributed_defaults(self):
        cfg = {"model": {"name_or_path": "test-model"}}
        result = to_esurge_kwargs(cfg)
        assert result["distributed_mode"] is False
        assert result["distributed_role"] == "auto"
        assert result["distributed_control_port"] == 19666

    def test_distributed_settings(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {
                "distributed_mode": True,
                "distributed_role": "leader",
                "distributed_control_port": 20000,
            },
        }
        result = to_esurge_kwargs(cfg)
        assert result["distributed_mode"] is True
        assert result["distributed_role"] == "leader"
        assert result["distributed_control_port"] == 20000

    def test_boolean_flags_false(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {
                "use_aot_forward": False,
                "enable_prefix_caching": False,
                "compile_runner": False,
                "overlap_execution": True,
                "silent_mode": True,
            },
        }
        result = to_esurge_kwargs(cfg)
        assert result["use_aot_forward"] is False
        assert result["enable_prefix_caching"] is False
        assert result["compile_runner"] is False
        assert result["overlap_execution"] is True
        assert result["silent_mode"] is True

    def test_sharding_axis_dims(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {"sharding_axis_dims": [1, 2, 1, 1, 4, 1]},
        }
        result = to_esurge_kwargs(cfg)
        assert result["sharding_axis_dims"] == (1, 2, 1, 1, 4, 1)

    def test_sharding_axis_dims_none(self):
        cfg = {
            "model": {"name_or_path": "test-model"},
            "esurge": {"sharding_axis_dims": None},
        }
        result = to_esurge_kwargs(cfg)
        assert result["sharding_axis_dims"] is None


class TestNormalize:
    def test_preserves_model_name(self):
        cfg = {"model": {"name_or_path": "test"}}
        result = normalize(cfg)
        assert result["model"]["name_or_path"] == "test"

    def test_adds_default_sections(self):
        cfg = {"model": {"name_or_path": "test"}}
        result = normalize(cfg)
        assert "model" in result
        assert "esurge" in result
