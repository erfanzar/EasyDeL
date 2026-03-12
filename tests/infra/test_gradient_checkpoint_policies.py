import jax

from easydel.infra.etils import GRADIENT_CHECKPOINT_TARGETS, EasyDeLGradientCheckPointers
from easydel.infra.utils import get_gradient_checkpoint_policy


def test_mlp_notsaveable_uses_non_mlp_targets(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_save_only_these_names(*names):
        captured["names"] = names
        return sentinel

    monkeypatch.setattr(jax.checkpoint_policies, "save_only_these_names", fake_save_only_these_names)

    policy = get_gradient_checkpoint_policy(EasyDeLGradientCheckPointers.MLP_NOTSAVEABLE)

    assert policy is sentinel
    assert set(captured["names"]) == {name for name in GRADIENT_CHECKPOINT_TARGETS if not name.startswith("mlp_")}


def test_attn_notsaveable_uses_non_attn_targets(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_save_only_these_names(*names):
        captured["names"] = names
        return sentinel

    monkeypatch.setattr(jax.checkpoint_policies, "save_only_these_names", fake_save_only_these_names)

    policy = get_gradient_checkpoint_policy(EasyDeLGradientCheckPointers.ATTN_NOTSAVEABLE)

    assert policy is sentinel
    assert set(captured["names"]) == {name for name in GRADIENT_CHECKPOINT_TARGETS if not name.startswith("attn_")}


def test_mlp_attn_notsaveable_uses_non_mlp_non_attn_targets(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_save_only_these_names(*names):
        captured["names"] = names
        return sentinel

    monkeypatch.setattr(jax.checkpoint_policies, "save_only_these_names", fake_save_only_these_names)

    policy = get_gradient_checkpoint_policy(EasyDeLGradientCheckPointers.MLP_ATTN_NOTSAVEABLE)

    assert policy is sentinel
    assert set(captured["names"]) == {
        name for name in GRADIENT_CHECKPOINT_TARGETS if not name.startswith("mlp_") and not name.startswith("attn_")
    }
