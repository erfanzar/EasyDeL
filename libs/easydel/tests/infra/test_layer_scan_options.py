"""``layer_scan_options``: validation, per-scope resolution, config roundtrip and sub-config propagation."""

import pytest
from easydel.infra.layer_scan import (
    auto_layer_scan_kwargs,
    infer_layer_scan_scope,
    normalize_layer_scan_options,
    resolve_layer_scan_kwargs,
)

pytestmark = pytest.mark.cpu_ok


def test_none_and_empty_pass_nothing_to_scan():
    for options in (None, {}):
        for scope in ("text", "vision", "audio"):
            assert resolve_layer_scan_kwargs(options, scope) == {}


@pytest.mark.parametrize(
    ("shape", "scope", "inference", "expected"),
    [
        ({"fsdp": 4}, "text", False, {"prefetch_axes": ("fsdp",)}),
        ({"fsdp": 1}, "text", False, {}),
        ({"fsdp": 4}, "vision", False, {}),
        ({"fsdp": 4}, "audio", False, {}),
        ({"fsdp": 4}, "text", True, {}),
        ({"dp": 4}, "text", False, {}),
    ],
)
def test_auto_default_prefetches_only_fsdp_sharded_training_text_scans(shape, scope, inference, expected):
    from types import SimpleNamespace

    assert auto_layer_scan_kwargs(SimpleNamespace(shape=shape), scope, inference=inference) == expected
    assert auto_layer_scan_kwargs(None, scope, inference=inference) == {}


def test_scope_overrides_merge_over_top_level_defaults():
    options = {"block_scan": True, "barrier": True, "vision": {"block_scan": False, "unroll": 2}}
    assert resolve_layer_scan_kwargs(options, "text") == {"block_scan": True, "barrier": True}
    assert resolve_layer_scan_kwargs(options, "vision") == {"block_scan": False, "barrier": True, "unroll": 2}
    assert resolve_layer_scan_kwargs(options, "audio") == {"block_scan": True, "barrier": True}


def test_scope_only_options_leave_other_towers_untouched():
    options = {"audio": {"prefetch_axes": ["fsdp"], "prefetch_exclude": "experts"}}
    assert resolve_layer_scan_kwargs(options, "text") == {}
    assert resolve_layer_scan_kwargs(options, "audio") == {"prefetch_axes": ("fsdp",), "prefetch_exclude": ("experts",)}


def test_name_options_are_normalized_to_tuples():
    kwargs = resolve_layer_scan_kwargs({"prefetch_axes": "fsdp", "prefetch_exclude": ["experts", "router"]}, "text")
    assert kwargs == {"prefetch_axes": ("fsdp",), "prefetch_exclude": ("experts", "router")}
    assert resolve_layer_scan_kwargs({"prefetch_axes": []}, "text") == {"prefetch_axes": None}


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"blok_scan": True}, KeyError),
        ({"vision": {"nope": 1}}, KeyError),
        ({"block_scan": 1}, TypeError),
        ({"vision": True}, TypeError),
        ({"prefetch_axes": 3}, TypeError),
        ({"unroll": 0}, ValueError),
        ({"unroll": True}, ValueError),
        ("block_scan", TypeError),
    ],
)
def test_invalid_options_are_refused(options, error):
    with pytest.raises(error):
        normalize_layer_scan_options(options)


def test_unknown_scope_is_refused():
    with pytest.raises(KeyError):
        resolve_layer_scan_kwargs({"block_scan": True}, "image")


def test_scope_inference_from_config_class_name():
    names = {"Qwen2VLVisionConfig": "vision", "Qwen3OmniMoeAudioEncoderConfig": "audio", "LlamaConfig": "text"}
    for name, scope in names.items():
        assert infer_layer_scan_scope(type(name, (), {})()) == scope


def test_config_field_roundtrips_and_validates_at_construction():
    import easydel as ed

    options = {"block_scan": True, "vision": {"unroll": 2}}
    cfg = ed.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        layer_scan_options=options,
    )
    assert cfg.layer_scan_options == options
    assert ed.LlamaConfig.from_dict(cfg.to_dict()).layer_scan_options == options
    assert ed.LlamaConfig().layer_scan_options is None
    with pytest.raises(KeyError):
        ed.LlamaConfig(layer_scan_options={"blok": True})


def test_add_basic_configurations_propagates_to_sub_configs():
    from easydel.modules.qwen2_vl import Qwen2VLConfig

    cfg = Qwen2VLConfig()
    options = {"text": {"block_scan": True}, "vision": {"barrier": True}}
    cfg.add_basic_configurations(layer_scan_options=options)
    assert cfg.layer_scan_options == options
    for name in cfg.sub_configs:
        assert getattr(cfg, name).layer_scan_options == options
