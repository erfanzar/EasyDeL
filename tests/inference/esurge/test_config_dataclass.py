from __future__ import annotations

from easydel.inference.esurge.config import (
    CacheConfig,
    Config,
    SchedulerConfig,
    eSurgeCacheRuntimeConfig,
    eSurgeContextConfig,
    eSurgeParsingConfig,
    eSurgeRuntimeConfig,
)
from easydel.modules.auto.auto_modeling import PreTrainedLoading


def test_esurge_config_dataclasses_round_trip_nested_dicts():
    config = Config.from_dict(
        {
            "scheduler_config": {
                "max_num_seqs": 4,
                "max_num_batched_tokens": 128,
                "max_model_len": 256,
                "max_num_seq_buckets": [1, 2, 4],
            },
            "cache_config": {
                "num_pages": None,
                "page_size": 32,
                "enable_prefix_caching": True,
                "max_cache_tokens": 4096,
            },
            "pipeline_inference": "ON",
            "kernel_tile_policy": "B8",
        }
    )

    assert isinstance(config.scheduler_config, SchedulerConfig)
    assert isinstance(config.cache_config, CacheConfig)
    assert config.scheduler_config.max_num_seq_buckets == (1, 2, 4)
    assert config.pipeline_inference == "on"
    assert config.kernel_tile_policy == "b8"
    assert config.to_dict() == {
        "scheduler_config": {
            "max_num_seqs": 4,
            "max_num_batched_tokens": 128,
            "max_model_len": 256,
            "policy": "fcfs",
            "long_prefill_token_threshold": 128,
            "chunked_prefill_enabled": False,
            "token_safety_margin": None,
            "max_num_seq_buckets": (1, 2, 4),
            "async_scheduling": True,
        },
        "cache_config": {
            "num_pages": None,
            "page_size": 32,
            "enable_prefix_caching": True,
            "max_cache_tokens": 4096,
            "cache_capacity_margin": 0.92,
        },
        "speculative_config": None,
        "mpmd_scheduler": None,
        "pipeline_inference": "on",
        "kernel_tile_policy": "b8",
    }


def test_esurge_config_typed_direct_from_dict_kwargs():
    scheduler = SchedulerConfig.from_dict(max_num_seqs=2, max_num_batched_tokens=None, max_model_len=128)
    cache = CacheConfig.from_dict(num_pages=16, page_size=32, enable_prefix_caching=False)
    config = Config.from_dict(scheduler_config=scheduler, cache_config=cache)

    assert scheduler.long_prefill_token_threshold > 0
    assert config.as_dict()["cache_config"]["page_size"] == 32
    assert config.replace(pipeline_inference="on").pipeline_inference == "on"


def test_config_dataclass_rejects_unknown_keys():
    try:
        SchedulerConfig.from_dict(max_num_seqs=2, max_num_batched_tokens=None, max_model_len=128, nope=True)
    except TypeError as exc:
        assert "unknown field" in str(exc)
        assert "nope" in str(exc)
    else:
        raise AssertionError("unknown config keys must fail loudly")


def test_esurge_engine_config_coerces_nested_sections():
    model = PreTrainedLoading.from_dict(pretrained_model_name_or_path="model-id", config_kwargs={"foo": "bar"})
    runtime = eSurgeRuntimeConfig.from_dict(max_model_len=256, max_num_seqs=4, max_num_batched_tokens=None)
    cache = eSurgeCacheRuntimeConfig.from_dict(page_size=32, hbm_utilization=0.5)
    context = eSurgeContextConfig.from_dict(reserve_tokens=4)
    parsing = eSurgeParsingConfig.from_dict()

    assert model.pretrained_model_name_or_path == "model-id"
    assert model.config_kwargs == {"foo": "bar"}
    assert runtime.max_model_len == 256
    assert runtime.max_num_seqs == 4
    assert cache.page_size == 32
    assert context.reserve_tokens == 4
    assert parsing.silent_mode is False


def test_generated_config_mutable_defaults_are_isolated():
    one = PreTrainedLoading.from_dict(pretrained_model_name_or_path="one", config_kwargs={"foo": "bar"})
    two = PreTrainedLoading.from_dict(pretrained_model_name_or_path="two", config_kwargs={"baz": "qux"})

    one.config_kwargs["flag"] = True
    # Mutating one instance's dict must not leak into another.
    assert "flag" not in two.config_kwargs
