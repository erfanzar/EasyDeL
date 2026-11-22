"""Pure-Python regression coverage for Whisper generation option preparation."""

import copy
from types import SimpleNamespace

import pytest
from easydel.inference.logits_process import LogitsProcessorList, WhisperTimeStampLogitsProcessor
from easydel.modules._base import BaseConditionalGenerationModule
from easydel.modules.whisper.modeling_whisper import WhisperForConditionalGeneration
from easydel.modules.xerxes2.xerxes2_configuration import Xerxes2Config

pytestmark = pytest.mark.cpu_ok


@pytest.fixture
def model(monkeypatch):
    """Exercise public generation preparation without allocating parameters or arrays."""

    def generate(self, inputs, generation_config, logits_processor=None, **kwargs):
        return SimpleNamespace(config=generation_config, processors=logits_processor, kwargs=kwargs)

    monkeypatch.setattr(BaseConditionalGenerationModule, "generate", generate)
    model = object.__new__(WhisperForConditionalGeneration)
    object.__setattr__(model, "config", SimpleNamespace(vocab_size=64))
    object.__setattr__(
        model,
        "generation_config",
        SimpleNamespace(
            return_timestamps=False,
            is_multilingual=True,
            language="en",
            task="transcribe",
            lang_to_id={"en": 10, "fr": 11},
            task_to_id={"transcribe": 12, "translate": 13},
            no_timestamps_token_id=20,
            eos_token_id=2,
            forced_decoder_ids=None,
        ),
    )
    return model


def test_generation_options_do_not_leak_between_calls(model):
    before = copy.deepcopy(vars(model.generation_config))
    first = model.generate(None, return_timestamps=True, task="translate", language="fr")
    assert first.config.language == "fr"
    assert first.config.task == "translate"
    assert first.config.return_timestamps is True
    assert vars(model.generation_config) == before
    second = model.generate(None)
    assert second.config.language == "en"
    assert second.config.task == "transcribe"
    assert second.config.return_timestamps is False
    assert vars(model.generation_config) == before


def test_explicit_generation_config_is_not_mutated(model):
    config = copy.deepcopy(model.generation_config)
    before = copy.deepcopy(vars(config))
    result = model.generate(None, generation_config=config, is_multilingual=False, return_timestamps=True)
    assert result.config.is_multilingual is False
    assert vars(config) == before
    assert result.config is not config


def test_timestamp_processors_keep_caller_processors_and_prompt_length(model):
    user_processor = object()
    processors = LogitsProcessorList([user_processor])
    # Only shape metadata is consumed; intentionally not an array computation.
    prompt = SimpleNamespace(shape=(3, 7))
    result = model.generate(None, return_timestamps=True, logits_processor=processors, decoder_input_ids=prompt)
    assert isinstance(result.processors, LogitsProcessorList)
    assert result.processors[0] is user_processor
    assert len(processors) == 1
    timestamp = result.processors[-1]
    assert isinstance(timestamp, WhisperTimeStampLogitsProcessor)
    assert timestamp.begin_index == 7 + 1 + 2


def test_no_timestamps_token_is_forced_once(model):
    result = model.generate(None, return_timestamps=False)
    assert result.config.forced_decoder_ids == [(1, 10), (2, 12), (3, 20)]
    result = model.generate(None, return_timestamps=False, is_multilingual=False)
    assert result.config.forced_decoder_ids == [(1, 20)]


@pytest.mark.parametrize("multilingual", [False, True])
@pytest.mark.parametrize("has_no_timestamps", [False, True])
def test_explicit_forced_tokens_are_preserved(model, multilingual, has_no_timestamps):
    config = copy.deepcopy(model.generation_config)
    config.is_multilingual = multilingual
    configured = [(1, 7), (2, 8)]
    if has_no_timestamps:
        configured.append((3, 20))
    config.forced_decoder_ids = configured
    original = copy.deepcopy(vars(config))
    result = model.generate(None, generation_config=config)
    assert result.config.forced_decoder_ids == [(1, 7), (2, 8), (3, 20)]
    assert vars(config) == original


def test_explicit_language_and_task_override_only_prefix_positions(model):
    config = copy.deepcopy(model.generation_config)
    config.forced_decoder_ids = [(1, 7), (2, 8), (5, 9)]
    result = model.generate(None, generation_config=config, language="fr", task="translate")
    assert result.config.forced_decoder_ids == [(1, 11), (2, 13), (5, 9), (6, 20)]
    assert config.forced_decoder_ids == [(1, 7), (2, 8), (5, 9)]


def test_forced_tokens_keyword_is_prepared_without_mutating_caller(model):
    configured = [(1, 7), (2, 8)]
    result = model.generate(None, is_multilingual=False, forced_decoder_ids=configured)
    assert result.config.forced_decoder_ids == [(1, 7), (2, 8), (3, 20)]
    assert configured == [(1, 7), (2, 8)]
    assert "forced_decoder_ids" not in result.kwargs


def test_timestamps_preserve_explicit_forced_tokens(model):
    config = copy.deepcopy(model.generation_config)
    config.forced_decoder_ids = [(1, 7), (2, 8)]
    result = model.generate(None, generation_config=config, return_timestamps=True)
    assert result.config.forced_decoder_ids == [(1, 7), (2, 8)]
    assert isinstance(result.processors[-1], WhisperTimeStampLogitsProcessor)


def test_no_timestamps_token_is_not_duplicated_when_it_is_not_last(model):
    config = copy.deepcopy(model.generation_config)
    config.forced_decoder_ids = [(1, 20), (2, 8)]
    result = model.generate(None, generation_config=config, is_multilingual=False)
    assert result.config.forced_decoder_ids == [(1, 20), (2, 8)]


def test_generic_generation_config_without_timestamp_vocabulary_is_supported(model):
    object.__setattr__(model, "generation_config", SimpleNamespace(forced_decoder_ids=None))
    result = model.generate(None)
    assert result.config.forced_decoder_ids is None


def test_xerxes2_default_rope_is_documented_yarn():
    config = Xerxes2Config()
    rope = config._get_rope_config()
    assert rope.rope_type == "yarn"
    assert rope.factor == 1.0
    assert rope.original_max_position_embeddings == 4096
    restored = Xerxes2Config.from_dict(config.to_dict())
    assert restored._get_rope_config().rope_type == "yarn"


def test_xerxes2_explicit_rope_and_mla_aliases_roundtrip():
    config = Xerxes2Config(rope_scaling={"type": "linear", "factor": 2.0}, q_lora_dim=16, kv_lora_dim=8, vhead_dim=4)
    assert config._get_rope_config().rope_type == "linear"
    assert config._get_rope_config().factor == 2.0
    assert (config.q_lora_rank, config.kv_lora_rank, config.v_head_dim) == (16, 8, 4)
    config.q_lora_rank = 12
    config.kv_lora_rank = 6
    config.v_head_dim = 2
    restored = Xerxes2Config.from_dict(config.to_dict())
    assert (restored.q_lora_dim, restored.kv_lora_dim, restored.vhead_dim) == (12, 6, 2)
    assert restored.attention_bias is False
