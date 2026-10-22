"""Host-only GRPO tokenizer chat-template regressions; no array creation or numerical execution."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from easydel.trainers import model_loading
from easydel.trainers.group_relative_policy_optimization import grpo_trainer

pytestmark = pytest.mark.cpu_ok


def test_shared_tokenizer_overrides_preserve_unspecified_tokens():
    tokenizer = SimpleNamespace(eos_token="old-eos", pad_token="old-pad", chat_template="template")
    model_loading.apply_tokenizer_overrides(tokenizer, eos_token="new-eos")
    assert vars(tokenizer) == {"eos_token": "new-eos", "pad_token": "old-pad", "chat_template": "template"}
    model_loading.apply_tokenizer_overrides(tokenizer, pad_token="new-pad")
    assert tokenizer.pad_token == "new-pad"
    assert tokenizer.eos_token == "new-eos"


def test_shared_chat_template_loads_utf8_file(tmp_path):
    path = tmp_path / "template.jinja"
    path.write_text("{{ messages }} — hello", encoding="utf-8")
    tokenizer = SimpleNamespace(chat_template="old")
    model_loading.apply_chat_template_path(tokenizer, str(path))
    assert tokenizer.chat_template == "{{ messages }} — hello"


@pytest.mark.parametrize("template", ["{{ messages }}", None])
def test_shared_chat_template_from_tokenizer_source(monkeypatch, template):
    monkeypatch.setattr(
        model_loading.AutoTokenizer,
        "from_pretrained",
        lambda source: SimpleNamespace(chat_template=template),
    )
    tokenizer = SimpleNamespace(chat_template="unchanged")
    if template is None:
        with pytest.raises(ValueError, match="No chat_template found"):
            model_loading.apply_chat_template_path(tokenizer, "review/nonexistent-tokenizer")
        assert tokenizer.chat_template == "unchanged"
    else:
        model_loading.apply_chat_template_path(tokenizer, "review/nonexistent-tokenizer")
        assert tokenizer.chat_template == template


def test_shared_chat_template_uses_cloud_path_backend(monkeypatch):
    monkeypatch.setattr(
        model_loading,
        "ePath",
        lambda path: SimpleNamespace(is_file=lambda: True, read_text=lambda **kwargs: "cloud-template"),
    )
    tokenizer = SimpleNamespace(chat_template=None)
    model_loading.apply_chat_template_path(tokenizer, "gs://example/template.jinja")
    assert tokenizer.chat_template == "cloud-template"


@dataclass(frozen=True)
class HostState:
    """State interface double containing only Python data, not JAX arrays."""

    model: object
    apply_fn: object = None
    graphdef: str = "definition"
    graphstate: str = "parameters"
    graphother: str = "buffers"

    def replace(self, **kwargs):
        return replace(self, **kwargs)


@pytest.mark.parametrize("bound_graphdef", [False, True])
def test_grpo_preserves_custom_reward_outputs_and_caller_state_list(monkeypatch, bound_graphdef):
    # Isolate constructor dispatch from training setup and device array allocation.
    monkeypatch.setattr(grpo_trainer, "EasyDeLState", HostState)
    monkeypatch.setattr(grpo_trainer, "jnp", SimpleNamespace(asarray=lambda values, dtype: values))
    monkeypatch.setattr(grpo_trainer, "replicated_named_sharding", lambda mesh: None)
    monkeypatch.setattr(grpo_trainer.Trainer, "__init__", lambda self, **kwargs: None)
    config = grpo_trainer.GRPOConfig(beta=0.0, disable_dropout=False, log_completions=False)
    policy = HostState(model=SimpleNamespace(mesh=None))
    tokenizer = SimpleNamespace(pad_token_id=0)

    def forward(graphdef, graphstate, graphother, batch):
        return {"score": f"{graphdef}/{graphstate}/{graphother}/{batch['text']}"}

    def bound_forward(graphstate, graphother, batch):
        return {"score": f"bound/{graphstate}/{graphother}/{batch['text']}"}

    original_forward = bound_forward if bound_graphdef else forward
    reward = HostState(model=SimpleNamespace(config=SimpleNamespace()), apply_fn=original_forward)
    caller_rewards = [reward]
    trainer = grpo_trainer.GRPOTrainer(
        config,
        model=policy,
        reward_funcs=caller_rewards,
        processing_class=tokenizer,
        reward_processing_classes=[tokenizer],
    )
    prepared = trainer.reward_funcs[0]
    output = prepared.apply_fn(prepared.graphdef, prepared.graphstate, prepared.graphother, {"text": "completion"})
    assert output == {"score": f"{'bound' if bound_graphdef else 'definition'}/parameters/buffers/completion"}
    assert caller_rewards[0] is reward
    assert reward.apply_fn is original_forward
    if not bound_graphdef:
        assert prepared.apply_fn is original_forward
