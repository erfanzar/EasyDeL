"""Output collection through the public LM wrappers, without array computation."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.modules._base import BaseCausalLMModule
from easydel.modules.minimax import modeling_minimax as minimax
from easydel.modules.minimax_m3_vl import modeling_minimax_m3_vl as minimax_m3
from easydel.modules.mistral import modeling_mistral as mistral
from easydel.modules.mixtral import modeling_mixtral as mixtral

pytestmark = pytest.mark.cpu_ok


class _Layers:
    def __init__(self, states, attentions):
        self.states = states
        self.attentions = attentions

    def __len__(self):
        return len(self.states)

    def scan(self, function, carry, trace):
        for state, attention in zip(self.states, self.attentions, strict=True):

            def block(state=state, attention=attention, **kwargs):
                return SimpleNamespace(
                    hidden_states=state, attention_weight=attention, cache_view=None, router_logits=None
                )

            carry = function(block, carry)
        return carry


@pytest.mark.parametrize(
    "module,trunk_class,lm_class",
    [
        (mistral, mistral.MistralModel, mistral.MistralForCausalLM),
        (mixtral, mixtral.MixtralModel, mixtral.MixtralForCausalLM),
        (minimax, minimax.MiniMaxModel, minimax.MiniMaxForCausalLM),
        (minimax_m3, minimax_m3.MiniMaxM3VLTextModel, minimax_m3.MiniMaxM3VLForCausalLM),
    ],
)
@pytest.mark.parametrize("override", [None, False, True])
def test_public_lm_preserves_configured_outputs_and_final_hidden_state(
    monkeypatch, module, trunk_class, lm_class, override
):
    embeddings = SimpleNamespace(shape=(1, 3, 4))
    first, second, normalized = (object() for _ in range(3))
    attentions = (object(), object())
    cache = SimpleNamespace(views=(None, None))
    config = SimpleNamespace(
        output_attentions=True,
        output_hidden_states=True,
        output_router_logits=False,
        max_position_embeddings=16,
        runtime_sharding_resolver=None,
        get_basic_frequencies=lambda: None,
    )
    monkeypatch.setattr(module, "MaskInfo", SimpleNamespace(dynamic_init=lambda **kwargs: kwargs["mask_info"]))
    monkeypatch.setattr(module, "apply_logical_sharding", lambda value, **kwargs: value)
    monkeypatch.setattr(module, "checkpoint_name", lambda value, *args, **kwargs: value)
    trunk = SimpleNamespace(
        config=config,
        layers=_Layers((first, second), attentions),
        frequencies=None,
        norm=lambda value: normalized,
        _layer_scan_trace=lambda *args, **kwargs: True,
        _layer_stage_context=lambda *args, **kwargs: nullcontext(),
        _layer_cache_view_at=lambda *args, **kwargs: None,
        _layer_cache_view_update=lambda views, *args, **kwargs: views,
        _mark_layer_stage_boundary=lambda value, *args, **kwargs: value,
    )
    # The real shared scan step, driven by the stubbed stage/cache hooks above.
    trunk._decoder_scan_step = MethodType(EasyDeLBaseModule._decoder_scan_step, trunk)
    lm = SimpleNamespace(
        config=config,
        base_model=lambda **kwargs: trunk_class.forward(trunk, **kwargs),
        compute_router_aux_loss=lambda *args, **kwargs: None,
        compute_aux_loss=lambda *args, **kwargs: None,
    )
    lm.model = lm.base_model
    lm.forward_moe = MethodType(BaseCausalLMModule.forward_moe, lm)
    output = lm_class.forward(
        lm,
        inputs_embeds=embeddings,
        mask_info=SimpleNamespace(q_position_ids=object()),
        past_key_values=cache,
        apply_lm_head=False,
        output_attentions=override,
        output_hidden_states=override,
    )
    assert output.past_key_values is cache
    assert output.last_hidden_state is normalized
    if override is False:
        assert output.hidden_states is None
        assert output.attentions is None
    else:
        assert output.hidden_states == (embeddings, first, normalized)
        assert len(output.hidden_states) == len(trunk.layers) + 1
        assert output.hidden_states[-1] is output.last_hidden_state
        assert output.attentions == attentions
