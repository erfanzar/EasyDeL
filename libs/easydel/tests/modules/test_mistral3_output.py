# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""The documented Mistral3 output is returned even when the LM head is disabled."""

from types import SimpleNamespace

import pytest
from easydel.infra.modeling_outputs import VLMCausalLMOutput
from easydel.modules.mistral3 import modeling_mistral3 as modeling

pytestmark = pytest.mark.cpu_ok


def test_output_preserves_legacy_loss_first_positional_constructor():
    values = [object() for _ in range(6)]
    names = ("loss", "logits", "past_key_values", "hidden_states", "attentions", "image_hidden_states")
    for count in range(1, 7):
        output = modeling.Mistral3CausalLMOutputWithPast(*values[:count])
        assert isinstance(output, VLMCausalLMOutput)
        for index, name in enumerate(names):
            assert getattr(output, name) is (values[index] if index < count else None)
        assert tuple(output.keys()) == names[:count]
        assert output.to_tuple() == tuple(values[:count])
        for index in range(count):
            assert output[index] is values[index]
        assert output[:] == tuple(values[:count])


@pytest.mark.parametrize("missing", ["loss", "logits", "past_key_values", "hidden_states", "attentions"])
def test_output_omits_none_fields_without_changing_legacy_mapping_order(missing):
    names = ("loss", "logits", "past_key_values", "hidden_states", "attentions", "image_hidden_states")
    fields = {name: object() for name in names if name != missing}
    output = modeling.Mistral3CausalLMOutputWithPast(**fields)
    assert tuple(output) == tuple(fields)
    assert output.to_tuple() == tuple(fields.values())
    assert output[0] is next(iter(fields.values()))


def test_legacy_constructor_accepts_shared_fields_by_keyword():
    from dataclasses import replace

    from jax.tree_util import tree_flatten, tree_unflatten

    loss, logits, last_hidden = object(), object(), object()
    output = modeling.Mistral3CausalLMOutputWithPast(loss, logits, last_hidden_state=last_hidden)
    assert output.loss is loss
    assert output.logits is logits
    assert output.last_hidden_state is last_hidden
    leaves, tree = tree_flatten(output)
    restored = tree_unflatten(tree, leaves)
    copied = replace(output)
    for other in (restored, copied):
        assert type(other) is type(output)
        assert other.loss is loss
        assert other.logits is logits
        assert other.last_hidden_state is last_hidden
        assert tuple(other) == ("loss", "logits", "last_hidden_state")
        assert other.to_tuple() == (loss, logits, last_hidden)
        assert other[0] is loss


@pytest.mark.parametrize("container", [dict, list, tuple, iter])
def test_output_preserves_legacy_key_value_construction(container):
    logits, cache, loss = object(), object(), object()
    # Pair construction preserves input order, not dataclass field order.
    pairs = [("past_key_values", cache), ("logits", logits), ("loss", loss), ("attentions", None)]
    payload = container(pairs)
    output = modeling.Mistral3CausalLMOutputWithPast(payload)
    assert isinstance(output, VLMCausalLMOutput)
    assert output.past_key_values is cache
    assert output.logits is logits
    assert output.loss is loss
    assert output.attentions is None
    assert tuple(output) == ("past_key_values", "logits", "loss")
    assert output.to_tuple() == (cache, logits, loss)
    assert output[0] is cache
    assert output[:] == (cache, logits, loss)


@pytest.mark.parametrize("container", [dict, list, tuple, iter])
def test_output_constructed_from_logits_pairs_has_no_loss_mapping_entry(container):
    logits = object()
    payload = container([("logits", logits)])
    output = modeling.Mistral3CausalLMOutputWithPast(payload)
    assert output.logits is logits
    # ModelOutput retains the first-field attribute itself, but only yields
    # the keys explicitly supplied by the pair iterator.
    assert output.loss is payload
    assert tuple(output) == ("logits",)
    assert output.to_tuple() == (logits,)
    assert output[0] is logits


@pytest.mark.parametrize("payload", [{}, [], ()])
def test_output_preserves_empty_legacy_key_value_construction(payload):
    output = modeling.Mistral3CausalLMOutputWithPast(payload)
    assert tuple(output) == ()
    assert output.to_tuple() == ()
    assert output.loss is payload
    assert output.logits is None


def test_output_preserves_legacy_iterable_validation():
    with pytest.raises(ValueError, match="Cannot set key/value"):
        modeling.Mistral3CausalLMOutputWithPast([("logits", object()), "invalid"])
    # A non-pair first element means this is a loss value, not key/value data.
    loss = [1, 2]
    output = modeling.Mistral3CausalLMOutputWithPast(loss)
    assert output.loss is loss
    assert tuple(output) == ("loss",)
    assert output[0] is loss


def test_output_dataclass_fields_retain_legacy_order():
    from dataclasses import fields

    names = tuple(field.name for field in fields(modeling.Mistral3CausalLMOutputWithPast))
    assert names[:6] == ("loss", "logits", "past_key_values", "hidden_states", "attentions", "image_hidden_states")
    assert "last_hidden_state" in names[6:]


def test_forward_returns_documented_output_and_preserves_backbone_states(monkeypatch):
    # Replace the numerical backbone and sharding boundary, not the output assembly.
    hidden, cache, image = object(), object(), object()
    backbone_output = SimpleNamespace(
        last_hidden_state=hidden,
        past_key_values=cache,
        hidden_states=(hidden,),
        attentions=None,
        image_hidden_states=image,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(output_attentions=False, output_hidden_states=True, runtime_sharding_resolver=None),
        base_model=lambda **kwargs: backbone_output,
    )
    monkeypatch.setattr(modeling, "apply_logical_sharding", lambda value, **kwargs: value)
    output = modeling.Mistral3ForConditionalGeneration.forward(
        model, input_ids=[[1]], pixel_values=object(), apply_lm_head=False
    )
    assert isinstance(output, modeling.Mistral3CausalLMOutputWithPast)
    assert isinstance(output, VLMCausalLMOutput)
    assert output.last_hidden_state is hidden
    assert output.past_key_values is cache
    assert output.hidden_states == (hidden,)
    assert output.image_hidden_states is image
    assert output.logits is None
    assert output.loss is None
