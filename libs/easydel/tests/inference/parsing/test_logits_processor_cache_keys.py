"""Pure-Python cache-key contracts for callable logits processor configurations."""

import pytest
from easydel.inference.logits_process import (
    EmptyProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    FrequencyPenaltyLogitsProcessor,
    MinLengthLogitsProcessor,
    MinPLogitsWarper,
    NoRepeatNGramLogitsProcessor,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    ("processor_type", "configuration"),
    [
        (EmptyProcessor, {}),
        (TemperatureLogitsWarper, {"temperature": 0.7}),
        (TopPLogitsWarper, {"top_p": 0.9}),
        (TopKLogitsWarper, {"top_k": 5}),
        (ForcedBOSTokenLogitsProcessor, {"bos_token_id": 1}),
        (ForcedEOSTokenLogitsProcessor, {"max_length": 8, "eos_token_id": 2}),
        (MinLengthLogitsProcessor, {"min_length": 3, "eos_token_id": 2}),
        (SuppressTokensAtBeginLogitsProcessor, {"begin_suppress_tokens": [1, 2], "begin_index": 3}),
        (SuppressTokensLogitsProcessor, {"suppress_tokens": [1, 2]}),
        (PresencePenaltyLogitsProcessor, {"presence_penalty": 0.5}),
        (FrequencyPenaltyLogitsProcessor, {"frequency_penalty": 0.5}),
        (RepetitionPenaltyLogitsProcessor, {"repetition_penalty": 2.0}),
        (MinPLogitsWarper, {"min_p": 0.1}),
        (NoRepeatNGramLogitsProcessor, {"ngram_size": 2}),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else None,
)
def test_processor_configuration_is_a_cache_key(processor_type, configuration):
    processor = processor_type(**configuration)
    restored = processor_type.from_dict(processor.to_dict())
    cache = {processor: "compiled processor"}
    assert restored == processor
    assert cache[restored] == "compiled processor"
    assert cache[processor.replace(**configuration)] == "compiled processor"


@pytest.mark.parametrize(
    ("processor_type", "configuration", "equivalent_configuration"),
    [
        (TemperatureLogitsWarper, {"temperature": 1}, {"temperature": 1.0}),
        (TopPLogitsWarper, {"top_p": 1}, {"top_p": 1.0}),
        (TopKLogitsWarper, {"top_k": 5}, {"top_k": 5.0}),
        (ForcedBOSTokenLogitsProcessor, {"bos_token_id": 1}, {"bos_token_id": 1.0}),
        (
            ForcedEOSTokenLogitsProcessor,
            {"max_length": 8, "eos_token_id": 2},
            {"max_length": 8.0, "eos_token_id": 2.0},
        ),
        (
            MinLengthLogitsProcessor,
            {"min_length": 0, "eos_token_id": 1},
            {"min_length": False, "eos_token_id": True},
        ),
        (
            SuppressTokensAtBeginLogitsProcessor,
            {"begin_suppress_tokens": [1, 2], "begin_index": 3},
            {"begin_suppress_tokens": [1.0, 2.0], "begin_index": 3.0},
        ),
        (SuppressTokensLogitsProcessor, {"suppress_tokens": [1, 2]}, {"suppress_tokens": [1.0, 2.0]}),
        (PresencePenaltyLogitsProcessor, {"presence_penalty": 0.0}, {"presence_penalty": -0.0}),
        (FrequencyPenaltyLogitsProcessor, {"frequency_penalty": 0.0}, {"frequency_penalty": -0.0}),
        (RepetitionPenaltyLogitsProcessor, {"repetition_penalty": 2}, {"repetition_penalty": 2.0}),
        (MinPLogitsWarper, {"min_p": 0.0}, {"min_p": -0.0}),
        (NoRepeatNGramLogitsProcessor, {"ngram_size": 2}, {"ngram_size": 2.0}),
        (TopPLogitsWarper, {"top_p": 1, "filter_value": 0.0}, {"top_p": 1.0, "filter_value": -0.0}),
        (TopKLogitsWarper, {"top_k": 5, "min_tokens_to_keep": 1}, {"top_k": 5, "min_tokens_to_keep": 1.0}),
        (MinPLogitsWarper, {"min_p": 0.1, "filter_value": 0.0}, {"min_p": 0.1, "filter_value": -0.0}),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else None,
)
def test_equal_numeric_configurations_share_cache_keys(processor_type, configuration, equivalent_configuration):
    processor = processor_type(**configuration)
    equivalent = processor_type(**equivalent_configuration)
    assert processor == equivalent
    assert hash(processor) == hash(equivalent)
    cache = {processor: "compiled processor"}
    assert cache[equivalent] == "compiled processor"
    cache[equivalent] = "reused processor"
    assert len(cache) == 1
    assert cache[processor] == "reused processor"


def test_distinct_processor_configurations_remain_distinct_cache_keys():
    short = MinLengthLogitsProcessor(min_length=3, eos_token_id=2)
    long = short.replace(min_length=5)
    cache = {short: "short", long: "long"}
    assert len(cache) == 2
    assert cache[MinLengthLogitsProcessor(min_length=3, eos_token_id=2)] == "short"
    assert cache[MinLengthLogitsProcessor(min_length=5, eos_token_id=2)] == "long"
