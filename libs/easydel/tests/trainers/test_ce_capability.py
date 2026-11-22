"""Custom causal-LM heads can declare CE-only eligibility without a family allowlist."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.infra.loss_utils import ForCausalLMLoss, LossMetrics
from easydel.modules._base.causal_lm_module import BaseCausalLMModule
from easydel.trainers import training_utils
from easydel.trainers.trainer import _fn

pytestmark = pytest.mark.cpu_ok


class HostInheritedHead(BaseCausalLMModule):
    """No model initialization or forward computation is performed in host tests."""

    loss_function = property(lambda self: ForCausalLMLoss)
    _router_aux_loss_feature = None


class HostCEOnlyHead(HostInheritedHead):
    _ce_only_forward = True

    def forward(self, *args, **kwargs):
        raise AssertionError("Host tests must not execute model forwards")


class HostUnreviewedHead(HostCEOnlyHead):
    pass


class HostReviewedHead(HostCEOnlyHead):
    _ce_only_forward = True


@pytest.fixture
def host_accumulation_runtime(monkeypatch):
    """Replace accelerator primitives only; keep real selection, token counts, and reduction."""

    def scan(fn, carry, xs, length):
        items = zip(*xs, strict=True) if isinstance(xs, tuple) else xs
        for item in items:
            carry, _ = fn(carry, item)
        return carry, None

    lax = SimpleNamespace(
        scan=scan,
        map=lambda fn, xs: np.asarray([fn(x) for x in xs]),
        dynamic_slice_in_dim=lambda value, start, size, axis: np.take(value, range(start, start + size), axis=axis),
    )
    host_jax = SimpleNamespace(
        tree_util=training_utils.jax.tree_util,
        lax=lax,
        eval_shape=lambda fn, *args: fn(*args),
    )
    monkeypatch.setattr(training_utils, "jax", host_jax)
    monkeypatch.setattr(training_utils, "lax", lax)
    monkeypatch.setattr(training_utils, "jnp", np)
    monkeypatch.setattr(_fn, "jnp", np)
    # An unreviewed head may add aux; reviewed/inherited CE-only heads must not
    # need a forward trace to receive the established valid-token behavior.
    monkeypatch.setattr(_fn, "_compute_loss_emits_aux", lambda *args: True)


@pytest.mark.parametrize(
    ("head_type", "aux_loss", "expected"),
    [
        (HostInheritedHead, 0.0, 6.0),
        (HostCEOnlyHead, 0.0, 6.0),
        (HostReviewedHead, 0.0, 6.0),
        (HostUnreviewedHead, 10.0, 15.0),
    ],
)
def test_capability_selected_accumulation_matches_token_mean_or_preserves_aux(
    host_accumulation_runtime, head_type, aux_loss, expected
):
    """Public accumulator outputs reflect token counts, not uniform means or inherited opt-ins."""
    _assert_capability_accumulation(head_type, aux_loss, expected)


def test_explicit_concrete_class_capability_is_eligible(host_accumulation_runtime):
    _assert_capability_accumulation(HostCEOnlyHead, aux_loss=0.0, expected=6.0)


def test_subclass_must_redeclare_capability(host_accumulation_runtime):
    _assert_capability_accumulation(HostUnreviewedHead, aux_loss=10.0, expected=15.0)
    _assert_capability_accumulation(HostReviewedHead, aux_loss=0.0, expected=6.0)


def _assert_capability_accumulation(head_type, aux_loss, expected):
    """Assert observable loss/gradient/counter behavior through the public accumulator."""
    model = object.__new__(head_type)
    state = SimpleNamespace(model=model, graphstate=None)
    batch = {
        "labels": np.array([[0, 1, -100, -100], [0, 2, 3, 4]]),
        "attention_mask": np.array([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "token_losses": np.array([[0.0, 2.0, 99.0, 99.0], [0.0, 6.0, 10.0, 99.0]]),
    }
    weight_fn = _fn._make_base_spmd_weight({"state": state, "batch": batch})

    def grad_fn(tree, microbatch):
        mask = (microbatch["labels"][:, 1:] != -100) & (microbatch["attention_mask"][:, 1:] != 0)
        count = mask.sum()
        loss = np.asarray((microbatch["token_losses"][:, 1:] * mask).sum() / max(count, 1) + aux_loss)
        return (loss, LossMetrics(loss=loss, weight_sum=np.asarray(count, dtype=np.float32))), loss * 2

    gradient, metrics = training_utils.minibatch_call(
        state, batch, minibatch_size=1, grad_fn=grad_fn, microbatch_weight_fn=weight_fn
    )
    assert float(metrics.loss) == pytest.approx(expected)
    assert float(gradient) == pytest.approx(expected * 2)
    assert float(metrics.weight_sum) == 3.0


@pytest.mark.parametrize("empty_rows", [1, 2])
def test_capability_weighting_handles_zero_label_microbatches(host_accumulation_runtime, empty_rows):
    state = SimpleNamespace(model=object.__new__(HostCEOnlyHead), graphstate=None)
    batch = {"labels": np.array([[0, -100, -100], [0, 1, 2]])}
    if empty_rows == 2:
        batch["labels"][1, 1:] = -100
    weight_fn = _fn._make_base_spmd_weight({"state": state, "batch": batch})

    def grad_fn(tree, microbatch):
        count = np.asarray((microbatch["labels"][:, 1:] != -100).sum(), dtype=np.float32)
        loss = np.asarray(8.0 if count else 0.0)
        return (loss, LossMetrics(loss=loss, weight_sum=count)), loss

    gradient, metrics = training_utils.minibatch_call(
        state, batch, minibatch_size=1, grad_fn=grad_fn, microbatch_weight_fn=weight_fn
    )
    assert float(metrics.loss) == pytest.approx(8.0 if empty_rows == 1 else 0.0)
    assert float(gradient) == pytest.approx(8.0 if empty_rows == 1 else 0.0)
    assert float(metrics.weight_sum) == (2.0 if empty_rows == 1 else 0.0)
