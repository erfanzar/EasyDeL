"""Mapping, attribute, tuple, pickle and pytree output views stay coherent."""

import pickle

import jax
import pytest
from easydel.infra.modeling_outputs import CausalLMOutput


@pytest.mark.cpu_ok
def test_output_field_mutation_inserts_replaces_and_removes_in_field_order():
    output = CausalLMOutput(logits="scores", hidden_states="states")
    output.loss = "loss"
    assert list(output)[:3] == ["logits", "hidden_states", "loss"]
    assert output[-1] == output["loss"] == output.loss == "loss"
    output.logits = "new scores"
    assert output["logits"] == "new scores"
    output.loss = None
    assert "loss" not in output
    output["hidden_states"] = None
    assert output.hidden_states is None
    assert "hidden_states" not in output
    assert output.to_tuple() == ("new scores",)
    output["loss"] = "new loss"
    assert output.loss == "new loss"
    assert output.to_tuple() == ("new scores", "new loss")
    output.logits = None
    output.logits = "new scores"
    assert list(output) == ["logits", "loss"]
    for restored in (pickle.loads(pickle.dumps(output)), jax.tree_util.tree_map(lambda x: x, output)):
        assert dict(restored) == dict(output)
        restored.loss = None
        assert "loss" not in restored
        restored.loss = "restored loss"
        assert restored[-1] == "restored loss"


@pytest.mark.cpu_ok
def test_pair_iterator_constructor_and_non_field_attributes_remain_supported():
    output = CausalLMOutput({"logits": "scores", "hidden_states": "states"})
    assert output["logits"] == output.logits == "scores"
    assert output["hidden_states"] == output.hidden_states == "states"
    output.annotation = "not an output field"
    assert "annotation" not in output
    output.logits = None
    assert "logits" not in output


@pytest.mark.cpu_ok
@pytest.mark.parametrize("container", [dict, list, tuple, iter])
def test_pair_constructor_preserves_supplied_order_and_later_field_synchronization(container):
    cache, logits, loss, extra = object(), object(), object(), object()
    pairs = [
        ("extra", extra),
        ("past_key_values", cache),
        ("loss", loss),
        ("logits", logits),
        ("attentions", None),
    ]
    output = CausalLMOutput(container(pairs))
    assert tuple(output) == ("extra", "past_key_values", "loss", "logits")
    assert output.to_tuple() == (extra, cache, loss, logits)
    assert output[0] is extra
    assert output[1:] == (cache, loss, logits)
    for key, value in pairs:
        assert getattr(output, key) is value
        if value is not None:
            assert output[key] is value
    assert "attentions" not in output

    # Once initialized, mutation still orders declared fields and omits None.
    output.logits = None
    assert "logits" not in output
    output["logits"] = logits
    assert output.logits is logits
    assert tuple(output) == ("logits", "past_key_values", "loss", "extra")
    assert output.to_tuple() == (logits, cache, loss, extra)
    output["loss"] = None
    assert output.loss is None
    assert "loss" not in output
    assert output.to_tuple() == (logits, cache, extra)
