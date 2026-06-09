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

"""Tests for ``easydel.infra.modeling_outputs``.

The module defines ``ModelOutput`` (an OrderedDict subclass) plus dozens of
``@auto_pytree`` dataclass subclasses for model return types. The tests lock
in:

* dual access (``out.field`` / ``out["field"]`` / ``out[0]``)
* ``to_tuple`` excludes None fields
* mutation methods are disabled (delitem, setdefault, pop, update)
* ``ModelOutput`` rejects subclasses that aren't dataclass-decorated
* dict-iterable single-arg construction works
* pickling round-trips
* JAX pytree flatten/unflatten preserves field order
* representative dataclass subclasses across tasks (Causal LM, MoE, VLM, Seq2Seq, Embedding)
"""

from __future__ import annotations

import collections
import pickle

import jax
import jax.numpy as jnp
import pytest
from eformer.pytree import auto_pytree

from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutput,
    DecoderLayerOutput,
    EmbeddingOutput,
    EncoderLayerOutput,
    ImageClassifierOutputWithNoAttention,
    MaskedLMOutput,
    ModelOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    QuestionAnsweringModelOutput,
    Seq2SeqLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    VLMCausalLMOutput,
)


def test_modeloutput_is_ordered_dict_subclass():
    assert issubclass(ModelOutput, collections.OrderedDict)


def test_modeloutput_rejects_undecorated_subclass():
    """A subclass that isn't @auto_pytree-decorated must raise TypeError on construction."""

    class BadOutput(ModelOutput):
        pass

    with pytest.raises(TypeError, match="not a dataclasss"):
        BadOutput()


def test_decorated_subclass_constructs_with_required_field():
    arr = jnp.zeros((2, 3))
    out = BaseModelOutput(last_hidden_state=arr)

    assert out.last_hidden_state is arr
    assert out["last_hidden_state"] is arr
    assert list(out.keys()) == ["last_hidden_state"]


def test_decorated_subclass_filters_none_fields_from_dict():
    arr = jnp.zeros((1, 1))
    out = BaseModelOutput(last_hidden_state=arr, hidden_states=None, attentions=None)
    assert "hidden_states" not in out
    assert "attentions" not in out
    assert list(out.keys()) == ["last_hidden_state"]


def test_decorated_subclass_to_tuple_returns_only_present_fields():
    arr = jnp.zeros((1, 1))
    attn = jnp.zeros((1, 2, 1, 1))
    out = BaseModelOutput(last_hidden_state=arr, attentions=(attn,))
    t = out.to_tuple()
    assert len(t) == 2
    assert t[0] is arr

    assert t[1] == (attn,)


def test_decorated_subclass_supports_integer_indexing():
    arr = jnp.zeros((1, 1))
    out = BaseModelOutput(last_hidden_state=arr)
    assert out[0] is arr


def test_decorated_subclass_disables_delitem():
    out = BaseModelOutput(last_hidden_state=jnp.zeros((1,)))
    with pytest.raises(Exception, match="__delitem__"):
        del out["last_hidden_state"]


def test_decorated_subclass_disables_setdefault():
    out = BaseModelOutput(last_hidden_state=jnp.zeros((1,)))
    with pytest.raises(Exception, match="setdefault"):
        out.setdefault("hidden_states", "x")


def test_decorated_subclass_disables_pop():
    out = BaseModelOutput(last_hidden_state=jnp.zeros((1,)))
    with pytest.raises(Exception, match="pop"):
        out.pop("last_hidden_state")


def test_decorated_subclass_disables_update():
    out = BaseModelOutput(last_hidden_state=jnp.zeros((1,)))
    with pytest.raises(Exception, match="update"):
        out.update({"last_hidden_state": jnp.zeros((2,))})


def test_setitem_syncs_to_attribute():
    """Per ``ModelOutput.__setitem__``, dict assignment also updates the attribute."""
    out = BaseModelOutput(last_hidden_state=jnp.zeros((1,)))
    new_arr = jnp.ones((3,))
    out["last_hidden_state"] = new_arr
    assert out.last_hidden_state is new_arr


def test_setattr_syncs_existing_key_to_dict():
    """Setting an attribute that's already a key updates the dict too."""
    out = BaseModelOutput(last_hidden_state=jnp.zeros((1,)))
    new_arr = jnp.ones((3,))
    out.last_hidden_state = new_arr
    assert out["last_hidden_state"] is new_arr


def test_modeloutput_accepts_dict_iterable_construction():
    """The ``__post_init__`` path that processes a dict in the first field."""
    arr1 = jnp.ones((2,))
    arr2 = jnp.ones((3,))
    out = BaseModelOutput({"last_hidden_state": arr1, "hidden_states": (arr2,)})
    assert out.last_hidden_state is arr1
    assert out.hidden_states == (arr2,)


def test_modeloutput_pickle_round_trip_preserves_fields():
    arr = jnp.zeros((2, 3))
    out = CausalLMOutput(logits=arr)
    revived = pickle.loads(pickle.dumps(out))
    assert isinstance(revived, CausalLMOutput)
    assert jnp.array_equal(revived.logits, arr)


def test_modeloutput_pickle_preserves_optional_fields():
    arr = jnp.zeros((2, 3, 5))
    hs = (jnp.zeros((2, 3, 4)),)
    out = MaskedLMOutput(logits=arr, hidden_states=hs)
    revived = pickle.loads(pickle.dumps(out))
    assert jnp.array_equal(revived.logits, arr)
    assert len(revived.hidden_states) == 1


def test_modeloutput_is_a_jax_pytree():
    """``@auto_pytree`` makes the class a registered pytree -- jax.tree_util can flatten it."""
    arr = jnp.array([1.0, 2.0, 3.0])
    out = BaseModelOutput(last_hidden_state=arr)
    leaves, treedef = jax.tree_util.tree_flatten(out)
    assert len(leaves) >= 1
    revived = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(revived, BaseModelOutput)
    assert jnp.array_equal(revived.last_hidden_state, arr)


def test_modeloutput_pytree_propagates_through_jit():
    """Pytree structure survives jax.jit -- common case for trainer step functions."""

    @jax.jit
    def double(out: BaseModelOutput) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=out.last_hidden_state * 2)

    arr = jnp.array([1.0, 2.0])
    result = double(BaseModelOutput(last_hidden_state=arr))
    assert isinstance(result, BaseModelOutput)
    assert jnp.allclose(result.last_hidden_state, arr * 2)


def test_attention_layer_output_first_field_is_required():
    arr = jnp.zeros((1, 2, 3))
    out = AttentionLayerOutput(attention_output=arr)
    assert out.attention_output is arr
    assert out.attention_weight is None
    assert out.cache_view is None
    assert list(out.keys()) == ["attention_output"]


def test_encoder_layer_output_with_attention_weights():
    arr = jnp.zeros((1, 2, 3))
    weights = jnp.zeros((1, 4, 2, 2))
    out = EncoderLayerOutput(hidden_states=arr, attention_weight=weights)
    assert out.attention_weight.shape == (1, 4, 2, 2)
    assert "residual_states" not in out


def test_decoder_layer_output_with_router_logits():
    """DecoderLayerOutput supports MoE router logits and gate loss."""
    hs = jnp.zeros((1, 2, 3))
    rl = jnp.zeros((1, 2, 8))
    gl = jnp.array(0.1)
    out = DecoderLayerOutput(hidden_states=hs, router_logits=rl, gate_loss=gl)
    assert out.router_logits.shape == (1, 2, 8)
    assert float(out.gate_loss) == pytest.approx(0.1)

    present = list(out.keys())
    assert "hidden_states" in present
    assert "router_logits" in present
    assert "gate_loss" in present


@pytest.mark.parametrize(
    "cls,first_field,extra_kwargs",
    [
        (BaseModelOutputWithPast, "last_hidden_state", {}),
        (CausalLMOutput, "logits", {"loss": jnp.array(1.0)}),
        (MaskedLMOutput, "logits", {}),
        (MoeModelOutput, "last_hidden_state", {}),
        (
            MoeCausalLMOutput,
            "logits",
            {"aux_loss": jnp.array(0.05)},
        ),
        (Seq2SeqLMOutput, "logits", {}),
        (SequenceClassifierOutput, "logits", {}),
        (TokenClassifierOutput, "logits", {}),
        (QuestionAnsweringModelOutput, "start_logits", {"end_logits": jnp.zeros((2, 4))}),
        (
            ImageClassifierOutputWithNoAttention,
            "logits",
            {},
        ),
        (EmbeddingOutput, "embeddings", {}),
        (VLMCausalLMOutput, "logits", {}),
    ],
)
def test_task_specific_output_round_trip(cls, first_field, extra_kwargs):
    """Each output class accepts its first field and returns it via attr/dict/tuple access."""
    arr = jnp.zeros((2, 4))
    init_kwargs = {first_field: arr, **extra_kwargs}
    out = cls(**init_kwargs)
    assert getattr(out, first_field) is arr
    assert out[first_field] is arr

    assert out[0] is arr

    t = out.to_tuple()
    assert t[0] is arr
    assert len(t) == 1 + sum(1 for v in extra_kwargs.values() if v is not None)


def test_modeloutput_rejects_multiple_required_fields():
    """``__post_init__`` raises ValueError if more than one field lacks a None default."""

    @auto_pytree
    class TwoRequired(ModelOutput):
        a: object
        b: object

    with pytest.raises(ValueError, match="more than one required field"):
        TwoRequired(a="x", b="y")


def test_modeloutput_rejects_no_fields():
    """``__post_init__`` raises ValueError on an empty dataclass."""

    @auto_pytree
    class Empty(ModelOutput):
        pass

    with pytest.raises(ValueError, match="no fields"):
        Empty()


def test_modeloutput_keys_view_iterates_in_insertion_order():
    """``ModelOutput`` is OrderedDict -- field order is the dataclass declaration order."""
    arr = jnp.zeros((1,))
    weights = jnp.zeros((1, 2, 1, 1))
    out = AttentionLayerOutput(attention_output=arr, attention_weight=weights)
    assert list(out.keys()) == ["attention_output", "attention_weight"]


def test_modeloutput_indexing_by_unknown_string_key_raises_keyerror():
    out = BaseModelOutput(last_hidden_state=jnp.zeros((1,)))
    with pytest.raises(KeyError):
        _ = out["nonexistent_field"]


def test_modeloutput_integer_indexing_out_of_range_raises():
    out = BaseModelOutput(last_hidden_state=jnp.zeros((1,)))
    with pytest.raises(IndexError):
        _ = out[5]
