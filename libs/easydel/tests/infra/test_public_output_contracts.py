"""Public output contracts are useful independently of built-in model callers."""

import pytest
from easydel.infra import modeling_outputs


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("output_name", "field_name"),
    [
        ("MambaOutput", "cache_params"),
        ("MambaCausalLMOutput", "cache_params"),
        ("NextSentencePredictorOutput", "logits"),
        ("Seq2SeqSequenceClassifierOutput", "logits"),
        ("Seq2SeqQuestionAnsweringModelOutput", "start_logits"),
        ("BaseModelOutputWithNoAttention", "last_hidden_state"),
        ("BaseModelOutputWithPoolingAndNoAttention", "pooler_output"),
        ("ImageClassifierOutputWithNoAttention", "logits"),
    ],
)
def test_documented_output_types_retain_mapping_and_tuple_access(output_name, field_name):
    # Python objects suffice to test the output protocol without array computation.
    value = object()
    output = getattr(modeling_outputs, output_name)(**{field_name: value})

    assert getattr(output, field_name) is value
    assert output[field_name] is value
    assert value in output.to_tuple()
