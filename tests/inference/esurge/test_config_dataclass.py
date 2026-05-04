from __future__ import annotations

from easydel.inference.esurge.config import (
    eSurgeCacheRuntimeConfig,
    eSurgeContextConfig,
    eSurgeParsingConfig,
    eSurgeRuntimeConfig,
)
from easydel.modules.auto.auto_modeling import PreTrainedLoading


def test_esurge_sectioned_configs_round_trip_dicts():
    runtime = eSurgeRuntimeConfig.from_dict(
        {
            "max_num_seqs": 4,
            "max_num_batched_tokens": 128,
            "max_model_len": 256,
            "max_num_seq_buckets": [4],
            "kernel_tile_policy": "B8",
        }
    )
    cache = eSurgeCacheRuntimeConfig.from_dict(
        {
            "max_cache_tokens": 4096,
            "page_size": 32,
            "enable_prefix_caching": True,
        }
    )

    assert runtime.max_num_seq_buckets == [4]
    assert runtime.kernel_tile_policy == "b8"
    assert runtime.to_dict()["max_num_seqs"] == 4
    assert cache.to_dict()["page_size"] == 32


def test_esurge_config_typed_direct_from_dict_kwargs():
    runtime = eSurgeRuntimeConfig.from_dict(max_num_seqs=2, max_num_batched_tokens=None, max_model_len=128)
    cache = eSurgeCacheRuntimeConfig.from_dict(max_cache_tokens=16, page_size=32, enable_prefix_caching=False)

    assert runtime.long_prefill_token_threshold is None
    assert cache.as_dict()["page_size"] == 32
    assert runtime.replace(kernel_tile_policy="b8").kernel_tile_policy == "b8"


def test_config_dataclass_rejects_unknown_keys():
    try:
        eSurgeRuntimeConfig.from_dict(max_num_seqs=2, max_num_batched_tokens=None, max_model_len=128, nope=True)
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
