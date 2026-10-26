"""Non-computation regressions for Qwen family compatibility and forward metadata."""

from types import SimpleNamespace

import pytest
from easydel.modules.qwen3_vl import Qwen3VLCausalLMOutputWithPast
from easydel.modules.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from easydel.modules.qwen3_vl_moe import Qwen3VLMoeCausalLMOutputWithPast
from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModel

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("output_cls", [Qwen3VLCausalLMOutputWithPast, Qwen3VLMoeCausalLMOutputWithPast])
def test_legacy_output_order_omits_none_values_and_survives_replace(output_cls):
    logits, rope, image, router = (object() for _ in range(4))
    kwargs = {"router_logits": router} if output_cls is Qwen3VLMoeCausalLMOutputWithPast else {}
    output = output_cls(None, logits, None, None, None, rope, image, **kwargs)
    expected = (logits, rope, image, router) if kwargs else (logits, rope, image)
    assert output.to_tuple() == expected
    assert output[:] == expected
    assert output[0] is logits
    assert output["rope_deltas"] is rope
    replaced = output.replace(loss=1.0)
    assert replaced.to_tuple() == (1.0, *expected)
    assert replaced[0] == 1.0
    assert replaced[1] is logits
    output["loss"] = 2.0
    assert output.to_tuple() == (2.0, *expected)
    output.loss = None
    assert output.to_tuple() == expected


def _model_with_decoder(cls):
    model = cls.__new__(cls)
    object.__setattr__(model, "config", SimpleNamespace())
    final, cache, hidden, attention, router = (object() for _ in range(5))

    def decoder(**kwargs):
        # Model outputs expose the exact DeepStack metadata the decoder consumed;
        # this avoids any JAX computation while checking the public forward contract.
        return SimpleNamespace(
            last_hidden_state=final,
            past_key_values=cache,
            hidden_states=(
                kwargs["inputs_embeds"],
                kwargs["visual_pos_masks"],
                kwargs["deepstack_visual_embeds"],
                hidden,
            ),
            attentions=attention,
            router_logits=router,
        )

    object.__setattr__(model, "language_model", decoder)
    return model, (final, cache, hidden, attention, router)


@pytest.mark.parametrize("cls", [Qwen3VLModel, Qwen3VLMoeModel])
def test_forward_preserves_precomputed_embeddings_deepstack_and_router_outputs(cls):
    model, (final, cache, hidden, attention, router) = _model_with_decoder(cls)
    embeddings, mask, positions = (object() for _ in range(3))
    residuals = [object()]
    kwargs = {"output_router_logits": True} if cls is Qwen3VLMoeModel else {}
    output = model.forward(
        inputs_embeds=embeddings,
        position_ids=positions,
        visual_pos_masks=mask,
        deepstack_visual_embeds=residuals,
        **kwargs,
    )
    assert output.last_hidden_state is final
    assert output.past_key_values is cache
    assert output.hidden_states == (embeddings, mask, residuals, hidden)
    assert output.attentions is attention
    assert output.rope_deltas is None
    if kwargs:
        assert output.router_logits is router


@pytest.mark.parametrize("cls", [Qwen3VLModel, Qwen3VLMoeModel])
def test_embeddings_only_forward_delegates_automatic_positions_to_text_decoder(cls):
    model, (final, _, _, _, _) = _model_with_decoder(cls)
    # A non-array sentinel proves the VLM does not dereference missing token
    # IDs or manufacture a shape-based position tensor before its decoder.
    embeddings = object()
    output = model(inputs_embeds=embeddings)
    assert output.last_hidden_state is final
    assert output.hidden_states[0] is embeddings
    assert output.rope_deltas is None


@pytest.mark.parametrize("cls", [Qwen3VLModel, Qwen3VLMoeModel])
def test_missing_deepstack_mask_is_rejected_before_decoder_execution(cls):
    model, _ = _model_with_decoder(cls)
    with pytest.raises(ValueError, match="visual_pos_masks"):
        model.forward(inputs_embeds=object(), position_ids=object(), deepstack_visual_embeds=[object()])


@pytest.mark.parametrize("cls", [Qwen3VLModel, Qwen3VLMoeModel])
def test_embedding_preparation_retains_precomputed_text_without_reencoding(cls):
    model = object.__new__(cls)
    embeddings = object()
    assert model.compute_embedding(None, inputs_embeds=embeddings) is embeddings
    with pytest.raises(ValueError, match="input_ids"):
        model.compute_embedding(None)
    with pytest.raises(ValueError, match="input_ids"):
        model.compute_embedding_with_info(None, inputs_embeds=embeddings)
