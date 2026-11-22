"""Pure-Python public Qwen2-VL output-protocol regressions."""

from types import SimpleNamespace

import pytest
from easydel.infra.modeling_outputs import BaseModelOutput, VLMCausalLMOutput
from easydel.modules.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast, Qwen2VLModel


@pytest.mark.cpu_ok
def test_qwen2vl_legacy_output_preserves_shared_vlm_fields():
    output = Qwen2VLCausalLMOutputWithPast(
        logits="logits", last_hidden_state="hidden", rope_deltas="delta", past_key_values="cache"
    )
    assert isinstance(output, VLMCausalLMOutput)
    assert output.logits == "logits"
    assert output.last_hidden_state == "hidden"
    assert output.rope_deltas == "delta"
    assert output.past_key_values == "cache"
    positional = Qwen2VLCausalLMOutputWithPast("loss", "logits", "cache", "hidden", "attention", "delta")
    assert positional.loss == "loss"
    assert positional.logits == "logits"
    assert positional.past_key_values == "cache"
    assert positional.hidden_states == "hidden"
    assert positional.attentions == "attention"
    assert positional.rope_deltas == "delta"


@pytest.mark.cpu_ok
@pytest.mark.parametrize("config_use_cache", [False, True])
@pytest.mark.parametrize("use_cache", [None, False, True])
def test_qwen2vl_base_forward_always_returns_named_output_with_cache(config_use_cache, use_cache):
    hidden, cache = object(), object()
    language_output = BaseModelOutput(last_hidden_state=hidden, past_key_values=cache)
    model = SimpleNamespace(
        config=SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            text_config=SimpleNamespace(use_cache=config_use_cache),
        ),
        language_model=lambda **kwargs: language_output,
    )
    # Supplying embeddings and already-3D positions exercises only output routing,
    # with no array creation, model initialization, or numerical operations.
    kwargs = {} if use_cache is None else {"use_cache": use_cache}
    result = Qwen2VLModel.forward(
        model,
        inputs_embeds=hidden,
        position_ids=SimpleNamespace(ndim=3),
        **kwargs,
    )
    assert isinstance(result, BaseModelOutput)
    assert result.last_hidden_state is hidden
    assert result.past_key_values is cache
