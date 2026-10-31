"""Pure-Python public output and Qwen4 cache-layout regression tests."""

import importlib
from dataclasses import fields
from types import SimpleNamespace

import pytest
from easydel.caching import RaggedPagesCacheView, UnifiedAttentionCacheView
from easydel.infra.modeling_outputs import VLMCausalLMOutput
from easydel.modules._base import BaseCausalLMModule, BaseVisionLanguageModule
from easydel.modules.qwen3_vl import Qwen3VLCausalLMOutputWithPast
from easydel.modules.qwen3_vl_moe import Qwen3VLMoeCausalLMOutputWithPast
from easydel.modules.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpOperationsLinearView,
    Qwen4ExpPagedQSAView,
)

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("output_cls", [Qwen3VLCausalLMOutputWithPast, Qwen3VLMoeCausalLMOutputWithPast])
def test_family_outputs_preserve_positional_values_and_unified_fields(output_cls):
    loss, logits, cache, hidden, attention, rope, image, final = [object() for _ in range(8)]
    output = output_cls(loss, logits, cache, hidden, attention, rope, image, last_hidden_state=final)
    assert isinstance(output, VLMCausalLMOutput)
    assert (output.loss, output.logits, output.past_key_values, output.hidden_states) == (loss, logits, cache, hidden)
    assert (output.attentions, output.rope_deltas, output.image_hidden_states) == (attention, rope, image)
    assert output.last_hidden_state is final
    legacy = (loss, logits, cache, hidden, attention, rope, image)
    legacy_keys = (
        "loss",
        "logits",
        "past_key_values",
        "hidden_states",
        "attentions",
        "rope_deltas",
        "image_hidden_states",
    )
    assert output.to_tuple() == (*legacy, final)
    assert tuple(output.keys())[:7] == legacy_keys
    assert output[:7] == legacy
    assert all(output[index] is value for index, value in enumerate(legacy))
    assert {field.name for field in fields(VLMCausalLMOutput)} <= {field.name for field in fields(output)}
    updated = output.replace(aux_loss=3.0, video_hidden_states=final)
    assert type(updated) is output_cls
    assert updated.aux_loss == 3.0
    assert updated.video_hidden_states is final
    assert updated.logits is logits
    assert output.aux_loss is None


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_vl_moe"])
def test_conditional_forward_returns_exported_family_output(monkeypatch, family):
    module = importlib.import_module(f"easydel.modules.{family}.modeling_{family}")
    is_moe = family.endswith("moe")
    prefix = "Qwen3VLMoe" if is_moe else "Qwen3VL"
    wrapper_cls = getattr(module, f"{prefix}ForConditionalGeneration")
    output_cls = getattr(module, f"{prefix}CausalLMOutputWithPast")
    final, cache, rope, router = (object() for _ in range(4))
    base_output = SimpleNamespace(
        last_hidden_state=final,
        past_key_values=cache,
        hidden_states=None,
        attentions=None,
        rope_deltas=rope,
        router_logits=router,
    )
    config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        runtime_sharding_resolver=None,
        get_text_config=lambda: SimpleNamespace(output_router_logits=True),
    )
    wrapper = SimpleNamespace(
        config=config, model=lambda **kwargs: base_output, compute_router_aux_loss=lambda outputs: 0.125
    )
    monkeypatch.setattr(module, "apply_logical_sharding", lambda hidden, **kwargs: hidden)
    result = wrapper_cls.forward(wrapper, input_ids=None, apply_lm_head=False)
    assert isinstance(result, output_cls)
    assert isinstance(result, VLMCausalLMOutput)
    assert result.last_hidden_state is final
    assert result.past_key_values is cache
    assert result.rope_deltas is rope
    assert result.logits is None
    if is_moe:
        assert result.router_logits is router
        assert result.aux_loss == 0.125


def _wrapper(cls, text):
    wrapper = object.__new__(cls)
    object.__setattr__(wrapper, "config", text if cls is Qwen4ExpForCausalLM else SimpleNamespace(text_config=text))
    return wrapper


def _text_config():
    return SimpleNamespace(
        layer_types=["qwen_sparse_attention", "linear_attention"],
        qsa_enabled=True,
        ple_layer_indices_0based={1: 0},
        indexer_head_dim=8,
        indexer_compress_ratio=2,
        hidden_size=16,
        hc_count=2,
        ple_conv_kernel_size=3,
        ngram_size=4,
    )


@pytest.mark.parametrize(
    "cls,base", [(Qwen4ExpForCausalLM, BaseCausalLMModule), (Qwen4ExpForConditionalGeneration, BaseVisionLanguageModule)]
)
def test_public_cache_adapters_keep_qsa_and_ple_layouts(monkeypatch, cls, base):
    monkeypatch.setattr(base, "get_operations_cache_view", lambda self: {0: RaggedPagesCacheView, 1: object})
    monkeypatch.setattr(base, "init_operations_cache_config", lambda self: [SimpleNamespace(), SimpleNamespace()])
    wrapper = _wrapper(cls, _text_config())
    assert wrapper.get_operations_cache_view() == {0: Qwen4ExpPagedQSAView, 1: Qwen4ExpOperationsLinearView}
    configs = wrapper.init_operations_cache_config()
    assert vars(configs[0]) == {"qwen4_indexer_head_dim": 8, "qwen4_indexer_compress_ratio": 2}
    assert vars(configs[1]) == {
        "qwen4_with_ple": True,
        "qwen4_ple_conv_dim": 32,
        "qwen4_ple_conv_state_len": 8,
        "qwen4_ple_context_len": 3,
    }
    monkeypatch.setattr(base, "get_operations_cache_view", lambda self: {0: UnifiedAttentionCacheView})
    with pytest.raises(NotImplementedError):
        wrapper.get_operations_cache_view()


@pytest.mark.parametrize("cls", [Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration])
def test_disabled_mtp_returns_no_predictions(cls):
    wrapper = _wrapper(cls, _text_config())
    object.__setattr__(wrapper, "mtp", None)
    object.__setattr__(wrapper, "model", SimpleNamespace(language_model=None))
    assert wrapper.compute_mtp_outputs(last_stream_state=None, next_token_ids=None) is None
