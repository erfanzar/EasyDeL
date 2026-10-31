"""Array-free configuration and forward-planning tests for joint-vocabulary loss."""

from types import SimpleNamespace

import pytest
from easydel.infra.loss_utils import CausalLMLossStrategy, LossConfig, _resolve_module_vocab_size
from easydel.modules.llama.llama_configuration import VisionLlamaConfig

pytestmark = pytest.mark.cpu_ok


class _ProjectionProtocol:
    """Expose only the chunked-forward protocol, without constructing any arrays."""

    def __init__(self, config, vocab_size=None):
        self.config = config
        self.vocab_size = vocab_size

    def forward(self, input_ids, apply_lm_head=True):
        raise AssertionError("Forward planning must not execute the model")

    def compute_lm_logits(self, hidden_states):
        raise AssertionError("Forward planning must not execute the projection")


@pytest.mark.parametrize(
    ("config_fields", "module_vocab", "expected_vocab"),
    [
        ({"vocab_size": 16, "total_vocab_size": 24}, 16, 24),
        ({"vocab_size": 16}, 24, 16),
        ({"vocab_size": 16, "total_vocab_size": None}, None, 16),
        ({"vocab_size": 16, "total_vocab_size": "unavailable"}, None, 16),
        ({"vocab_size": "invalid"}, 16, 16),
        ({}, 16, 16),
        ({"text_config": SimpleNamespace(vocab_size=16)}, None, 16),
        ({"get_text_config": lambda: SimpleNamespace(vocab_size=16)}, None, 16),
    ],
)
def test_loss_vocabulary_resolution_preserves_legacy_fallbacks(config_fields, module_vocab, expected_vocab):
    module = _ProjectionProtocol(SimpleNamespace(lmhead_chunksize=None, **config_fields), module_vocab)
    assert _resolve_module_vocab_size(module) == expected_vocab
    # Metadata only: with this shape the joint vocabulary needs chunking, but the
    # text vocabulary does not. No token or logit array is allocated.
    labels = SimpleNamespace(ndim=2, shape=(1024, 12289))
    plan = CausalLMLossStrategy().plan_forward(
        module=module, labels=labels, loss_config=LossConfig(), batch={}, loss_kwargs={}
    )
    assert plan.forward_kwargs == ({"apply_lm_head": False} if expected_vocab == 24 else {})


def test_vision_llama_joint_vocab_drives_loss_planning_without_changing_text_shapes():
    config = VisionLlamaConfig(vocab_size=16, vision_vocab_size=8)
    module = _ProjectionProtocol(config, vocab_size=16)
    assert config.vocab_size == config.vision_token_offset == 16
    assert config.total_vocab_size == _resolve_module_vocab_size(module) == 24
    plan = CausalLMLossStrategy().plan_forward(
        module=module,
        labels=SimpleNamespace(ndim=2, shape=(1024, 12289)),
        loss_config=LossConfig(),
        batch={},
        loss_kwargs={},
    )
    assert plan.forward_kwargs == {"apply_lm_head": False}


def test_missing_vocabulary_still_raises():
    with pytest.raises(AttributeError, match="Could not resolve vocabulary size"):
        _resolve_module_vocab_size(SimpleNamespace(config=SimpleNamespace()))
