"""Host-only caller regressions for shared tokenizer initialization."""

from types import SimpleNamespace

import pytest
from easydel.trainers import model_loading
from easydel.trainers.reward_trainer import RewardConfig, RewardTrainer
from easydel.trainers.supervised_fine_tuning_trainer import SFTConfig, SFTTrainer
from easydel.trainers.trainer import Trainer

pytestmark = pytest.mark.cpu_ok

TRAINERS = [(RewardTrainer, RewardConfig), (SFTTrainer, SFTConfig)]


def _config_without_runtime_dropout(config_cls, **kwargs):
    if config_cls is RewardConfig:
        kwargs["disable_dropout"] = False
    return config_cls(**kwargs)


@pytest.mark.parametrize("trainer_cls, config_cls", TRAINERS)
@pytest.mark.parametrize("nested_tokenizer", [False, True])
@pytest.mark.parametrize("template_source", ["file", "pretrained"])
def test_constructor_applies_configured_tokens_and_template(
    monkeypatch, tmp_path, trainer_cls, config_cls, nested_tokenizer, template_source
):
    template = "{{ messages[0]['content'] }}"
    template_path = tmp_path / "chat_template.jinja"
    if template_source == "file":
        template_path.write_text(template, encoding="utf-8")
    else:
        monkeypatch.setattr(
            model_loading.AutoTokenizer,
            "from_pretrained",
            lambda source: SimpleNamespace(chat_template=template),
        )

    def initialize(self, **kwargs):
        self.constructor_inputs = kwargs

    monkeypatch.setattr(Trainer, "__init__", initialize)
    tokenizer = SimpleNamespace(eos_token="old-eos", pad_token=None, chat_template="old-template")
    processing_class = SimpleNamespace(tokenizer=tokenizer) if nested_tokenizer else tokenizer
    config = _config_without_runtime_dropout(
        config_cls, eos_token="<eos>", pad_token="<pad>", chat_template_path=str(template_path)
    )
    model = SimpleNamespace(to_state=lambda **kwargs: object())
    trainer = trainer_cls(arguments=config, processing_class=processing_class, model=model, data_collator=lambda x: x)
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.chat_template == template
    assert trainer.constructor_inputs["processing_class"] is processing_class


@pytest.mark.parametrize("trainer_cls, config_cls", TRAINERS)
def test_constructor_retains_pad_fallback_when_no_override(monkeypatch, trainer_cls, config_cls):
    monkeypatch.setattr(Trainer, "__init__", lambda self, **kwargs: None)
    tokenizer = SimpleNamespace(eos_token="<existing-eos>", pad_token=None, chat_template="keep")
    trainer_cls(
        arguments=_config_without_runtime_dropout(config_cls),
        processing_class=tokenizer,
        model=SimpleNamespace(to_state=lambda **kwargs: object()),
        data_collator=lambda x: x,
    )
    assert tokenizer.pad_token == "<existing-eos>"
    assert tokenizer.eos_token == "<existing-eos>"
    assert tokenizer.chat_template == "keep"


@pytest.mark.parametrize("trainer_cls, config_cls", TRAINERS)
def test_constructor_rejects_source_without_template(monkeypatch, tmp_path, trainer_cls, config_cls):
    monkeypatch.setattr(
        model_loading.AutoTokenizer, "from_pretrained", lambda source: SimpleNamespace(chat_template=None)
    )
    tokenizer = SimpleNamespace(eos_token="<eos>", pad_token=None, chat_template="keep")
    with pytest.raises(ValueError, match="No chat_template found"):
        trainer_cls(
            arguments=_config_without_runtime_dropout(config_cls, chat_template_path=str(tmp_path / "missing-template")),
            processing_class=tokenizer,
            model=None,
        )
    assert tokenizer.chat_template == "keep"


@pytest.mark.parametrize("trainer_cls, config_cls", TRAINERS)
def test_legacy_tokenizer_methods_preserve_updates_and_none_semantics(tmp_path, trainer_cls, config_cls):
    tokenizer = SimpleNamespace(eos_token="old-eos", pad_token="old-pad", chat_template=None)
    trainer_cls._apply_tokenizer_overrides(tokenizer, eos_token=None, pad_token=None)
    assert (tokenizer.eos_token, tokenizer.pad_token) == ("old-eos", "old-pad")
    trainer_cls._apply_tokenizer_overrides(tokenizer, eos_token="<eos>", pad_token="<pad>")
    assert (tokenizer.eos_token, tokenizer.pad_token) == ("<eos>", "<pad>")
    template_path = tmp_path / "template.jinja"
    template_path.write_text("template body", encoding="utf-8")
    trainer_cls._apply_chat_template_path(tokenizer, str(template_path))
    assert tokenizer.chat_template == "template body"
