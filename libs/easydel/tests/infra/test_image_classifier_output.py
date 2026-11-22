"""Image classification output contracts without numerical array execution."""

import pickle

import pytest
from easydel.infra.modeling_outputs import ImageClassifierOutput
from jax import tree_util


@pytest.mark.cpu_ok
@pytest.mark.parametrize("with_loss", [False, True])
def test_classification_scores_and_loss_support_output_protocol(with_loss):
    logits = object()
    loss = object() if with_loss else None
    output = ImageClassifierOutput(logits=logits, loss=loss)

    assert output.logits is logits
    assert output["logits"] is logits
    assert output.loss is loss
    assert list(output) == (["logits", "loss"] if with_loss else ["logits"])
    assert output.to_tuple() == ((logits, loss) if with_loss else (logits,))
    assert output[0] is logits
    if with_loss:
        assert output["loss"] is loss
        assert output[1] is loss


@pytest.mark.cpu_ok
def test_legacy_positional_embedding_fields_are_unchanged():
    embeds, last_hidden, hidden, attention = (object() for _ in range(4))
    output = ImageClassifierOutput(embeds, last_hidden, hidden, attention)

    assert output.text_embeds is embeds
    assert output.last_hidden_state is last_hidden
    assert output.hidden_states is hidden
    assert output.attentions is attention
    assert output.logits is None
    assert output.loss is None
    assert output.to_tuple() == (embeds, last_hidden, hidden, attention)


@pytest.mark.cpu_ok
def test_new_and_legacy_fields_survive_pytree_and_pickle_round_trips():
    output = ImageClassifierOutput(
        text_embeds="image embeddings",
        last_hidden_state="final features",
        hidden_states=("intermediate features",),
        attentions=("attention weights",),
        logits="classification scores",
        loss="supervised loss",
    )
    leaves, structure = tree_util.tree_flatten(output)
    restored = tree_util.tree_unflatten(structure, leaves)
    unpickled = pickle.loads(pickle.dumps(output))

    for round_tripped in (restored, unpickled):
        assert isinstance(round_tripped, ImageClassifierOutput)
        assert dict(round_tripped) == dict(output)
        assert round_tripped.to_tuple() == output.to_tuple()
        assert round_tripped.logits == "classification scores"
        assert round_tripped.loss == "supervised loss"
