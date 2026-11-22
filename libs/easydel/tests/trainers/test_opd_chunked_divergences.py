"""Accelerator checks for the vocabulary-streamed on-policy distillation divergences and trust-region mode."""

from types import SimpleNamespace

import jax
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.distillation_trainer._fn import distillation_loss
from easydel.trainers.on_policy_distillation_trainer import OnPolicyDistillationConfig
from easydel.trainers.on_policy_distillation_trainer._fn import on_policy_distillation_step
from easydel.trainers.on_policy_distillation_trainer._streamed_divergences import (
    FullVocabProjection,
    VocabChunkedProjection,
    streamed_distillation_loss,
    trust_region_distillation_loss,
)
from jax import numpy as jnp
from jax.sharding import PartitionSpec

B, L, H, V, CHUNK = 2, 9, 16, 1000, 128
HI = jax.lax.Precision.HIGHEST


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    h_s = jnp.asarray(rng.normal(size=(B, L, H)), jnp.float32)
    h_t = jnp.asarray(rng.normal(size=(B, L, H)), jnp.float32)
    w_s = jnp.asarray(rng.normal(size=(H, V)) * 0.5, jnp.float32)
    w_t = jnp.asarray(rng.normal(size=(H, V)) * 0.5, jnp.float32)
    mask = jnp.asarray(rng.integers(0, 2, size=(B, L)), jnp.int32).at[:, -3:].set(1)
    ids = jnp.asarray(rng.integers(0, V, size=(B, L)), jnp.int32)
    return h_s, h_t, w_s, w_t, mask, ids


def _projections(kind, w_s, w_t):
    if kind == "vocab":
        return (
            VocabChunkedProjection(w_s, chunk_size=CHUNK, precision=HI),
            VocabChunkedProjection(w_t, chunk_size=CHUNK, precision=HI),
            None,
        )
    return (
        FullVocabProjection(lambda h: jnp.einsum("...h,hv->...v", h, w_s, precision=HI)),
        FullVocabProjection(lambda h: jnp.einsum("...h,hv->...v", h, w_t, precision=HI)),
        4,
    )


SELECTORS = {
    "forward": {},
    "forward_beta0": {"beta": 0.0},
    "reverse": {"beta": 1.0},
    "jsd": {"beta": 0.3},
    "cakld": {"cakld_gamma": 0.4},
    "topk": {"loss_top_k": 17},
    "topk_tail": {"loss_top_k": 17, "loss_add_tail": True},
}


@pytest.mark.parametrize("kind", ["vocab", "sequence"])
@pytest.mark.parametrize("selector", list(SELECTORS))
def test_streamed_matches_dense_loss_and_grad(kind, selector):
    h_s, h_t, w_s, w_t, mask, _ = _problem()
    options = SELECTORS[selector]
    temperature = 2.0
    s_proj, t_proj, seq_chunk = _projections(kind, w_s, w_t)

    def streamed(hs):
        loss, metrics = streamed_distillation_loss(
            hs,
            h_t,
            s_proj,
            t_proj,
            loss_mask=mask,
            temperature=temperature,
            alpha=1.0,
            sequence_chunk_size=seq_chunk,
            **options,
        )
        return loss, metrics

    def dense(hs):
        loss, metrics = distillation_loss(
            student_logits=jnp.einsum("blh,hv->blv", hs, w_s, precision=HI),
            teacher_logits=jnp.einsum("blh,hv->blv", h_t, w_t, precision=HI),
            loss_mask=mask,
            temperature=temperature,
            alpha=1.0,
            **options,
        )
        return loss, metrics

    (loss_c, metrics_c), grad_c = jax.value_and_grad(streamed, has_aux=True)(h_s)
    (loss_d, metrics_d), grad_d = jax.value_and_grad(dense, has_aux=True)(h_s)
    np.testing.assert_allclose(loss_c, loss_d, rtol=2e-5, atol=2e-6)
    for key in ("kl_loss", "distill_xent_loss", "teacher_entropy_loss"):
        np.testing.assert_allclose(metrics_c[key], metrics_d[key], rtol=2e-5, atol=2e-6, err_msg=key)
    np.testing.assert_allclose(grad_c, grad_d, rtol=1e-4, atol=1e-6)


def _numpy_forward_kl(h_s, h_t, w_s, w_t, mask, temperature):
    hs, ht = np.asarray(h_s, np.float64)[:, :-1], np.asarray(h_t, np.float64)[:, :-1]
    m = np.asarray(mask, np.float64)[:, 1:]

    def log_softmax(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    ls = log_softmax(hs @ np.asarray(w_s, np.float64) / temperature)
    lt = log_softmax(ht @ np.asarray(w_t, np.float64) / temperature)
    kl = (np.exp(lt) * (lt - ls)).sum(-1)
    return (kl * m).sum() / max(m.sum(), 1.0) * temperature**2


def test_topk_covering_vocab_equals_full_forward_kl():
    h_s, h_t, w_s, w_t, mask, _ = _problem(1)
    s_proj, t_proj, _ = _projections("vocab", w_s, w_t)
    loss, _ = streamed_distillation_loss(
        h_s, h_t, s_proj, t_proj, loss_mask=mask, temperature=1.5, alpha=1.0, loss_top_k=V + 5
    )
    expected = _numpy_forward_kl(h_s, h_t, w_s, w_t, mask, 1.5)
    np.testing.assert_allclose(loss, expected, rtol=2e-5)


def _trust_region_reference(hs, h_t, w_s, w_t, ids, mask, accepted, temperature, k):
    ls = jax.nn.log_softmax(jnp.einsum("blh,hv->blv", hs[:, :-1], w_s, precision=HI) / temperature, axis=-1)
    lt = jax.nn.log_softmax(jnp.einsum("blh,hv->blv", h_t[:, :-1], w_t, precision=HI) / temperature, axis=-1)
    tgt = ids[:, 1:, None]
    s_lp = jnp.take_along_axis(ls, tgt, axis=-1)[..., 0]
    t_lp = jnp.take_along_axis(lt, tgt, axis=-1)[..., 0]
    pg = jax.lax.stop_gradient(s_lp - t_lp) * s_lp
    top = jnp.argsort(lt, axis=-1)[..., -k:]
    lt_k = jnp.take_along_axis(lt, top, axis=-1)
    ls_k = jnp.take_along_axis(ls, top, axis=-1)
    # Both branches score the same (sampling-temperature) distributions, so neither carries a T**2 factor.
    kl = jnp.sum(jnp.exp(lt_k) * (lt_k - ls_k), axis=-1)
    per = accepted * pg + (1.0 - accepted) * kl
    m = mask[:, 1:].astype(jnp.float32)
    return jnp.sum(per * m) / jnp.maximum(jnp.sum(m), 1.0)


@pytest.mark.parametrize("kind", ["vocab", "sequence"])
def test_trust_region_matches_reference_with_fixed_mask(kind):
    h_s, h_t, w_s, w_t, mask, ids = _problem(2)
    accepted = jnp.asarray(np.random.default_rng(3).integers(0, 2, size=(B, L - 1)), jnp.float32)
    s_proj, t_proj, seq_chunk = _projections(kind, w_s, w_t)
    temperature, k = 1.3, 64

    def streamed(hs):
        return trust_region_distillation_loss(
            hs,
            h_t,
            s_proj,
            t_proj,
            ids,
            loss_mask=mask,
            temperature=temperature,
            top_k=k,
            acceptance_mask=accepted,
            sequence_chunk_size=seq_chunk,
        )[0]

    def reference(hs):
        return _trust_region_reference(hs, h_t, w_s, w_t, ids, mask, accepted, temperature, k)

    loss_c, grad_c = jax.value_and_grad(streamed)(h_s)
    loss_r, grad_r = jax.value_and_grad(reference)(h_s)
    np.testing.assert_allclose(loss_c, loss_r, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(grad_c, grad_r, rtol=1e-4, atol=1e-6)


def test_trust_region_accepts_everything_when_student_equals_teacher():
    h_s, _, w_s, _, mask, ids = _problem(4)
    s_proj, t_proj, _ = _projections("vocab", w_s, w_s)
    loss, metrics = trust_region_distillation_loss(
        h_s, h_s, s_proj, t_proj, ids, loss_mask=mask, temperature=1.0, top_k=8, rng=jax.random.PRNGKey(0)
    )
    np.testing.assert_allclose(metrics["acceptance_rate"], 1.0)
    np.testing.assert_allclose(loss, 0.0, atol=1e-6)


def test_trust_region_uniform_draw_thresholds_acceptance_probability():
    h_s, h_t, w_s, w_t, mask, ids = _problem(5)
    s_proj, t_proj, _ = _projections("vocab", w_s, w_t)
    uniform = jnp.asarray(np.random.default_rng(6).uniform(size=(B, L - 1)), jnp.float32)
    temperature, k = 0.8, 32
    ls = jax.nn.log_softmax(jnp.einsum("blh,hv->blv", h_s[:, :-1], w_s, precision=HI) / temperature, axis=-1)
    lt = jax.nn.log_softmax(jnp.einsum("blh,hv->blv", h_t[:, :-1], w_t, precision=HI) / temperature, axis=-1)
    tgt = ids[:, 1:, None]
    log_ratio = jnp.take_along_axis(lt, tgt, -1)[..., 0] - jnp.take_along_axis(ls, tgt, -1)[..., 0]
    expected_mask = (uniform < jnp.exp(jnp.minimum(log_ratio, 0.0))).astype(jnp.float32)

    common = dict(loss_mask=mask, temperature=temperature, top_k=k)
    loss_u, metrics_u = trust_region_distillation_loss(
        h_s, h_t, s_proj, t_proj, ids, acceptance_uniform=uniform, **common
    )
    loss_m, metrics_m = trust_region_distillation_loss(
        h_s, h_t, s_proj, t_proj, ids, acceptance_mask=expected_mask, **common
    )
    np.testing.assert_allclose(loss_u, loss_m, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(metrics_u["acceptance_rate"], metrics_m["acceptance_rate"])


class _ConstantDistillModel(spx.Module):
    def __init__(self, logits, mesh):
        self.values = spx.Parameter(jnp.asarray(logits, dtype=jnp.float32))
        self.mesh = mesh

    def forward(self, input_ids, attention_mask, apply_lm_head=True):
        logits = jnp.broadcast_to(self.values.value, (*input_ids.shape, self.values.value.shape[0]))
        return SimpleNamespace(logits=logits if apply_lm_head else None, last_hidden_state=logits)

    def make_lm_head_fn(self):
        return lambda hidden: hidden


def test_trust_region_acceptance_draw_is_independent_of_gradient_accumulation():
    """Accumulated microbatches see the full-batch draw sliced, not one repeated per-microbatch draw."""
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("devices",))
    student = EasyDeLState.create(model=_ConstantDistillModel([0.3, -0.2, 1.1], mesh))
    teacher = EasyDeLState.create(model=_ConstantDistillModel([1.2, 0.4, -0.1], mesh))
    tx = optax.sgd(0.0)
    student = student.replace(tx=tx, opt_state=tx.init(student.graphstate))
    rows, length = 8, 6
    # Token 2 is far likelier under the student than the teacher, so its acceptance probability is in (0, 1)
    # and every row is identical: only the uniform draw distinguishes the two microbatches.
    batch = {
        "input_ids": jnp.full((rows, length), 2, dtype=jnp.int32),
        "attention_mask": jnp.ones((rows, length), dtype=jnp.int32),
        "completion_mask": jnp.array([[0] + [1] * (length - 1)] * rows, dtype=jnp.int32),
    }
    common = dict(
        partition_spec=PartitionSpec(),
        temperature=2.0,
        alpha=1.0,
        distillation_mode="trust_region",
        trust_region_top_k=2,
        trust_region_seed=11,
        sampling_temperature=1.0,
    )
    full = on_policy_distillation_step(student, batch, teacher, is_training=False, **common)
    _, accumulated = on_policy_distillation_step(
        student, batch, teacher, is_training=True, gradient_accumulation_steps=2, **common
    )
    full_rate = float(full.other_metrics["acceptance_rate"])
    assert 0.0 < full_rate < 1.0
    np.testing.assert_allclose(accumulated.other_metrics["acceptance_rate"], full_rate, rtol=1e-6)
    np.testing.assert_allclose(accumulated.loss, full.loss, rtol=1e-5, atol=1e-6)


@pytest.mark.cpu_ok
def test_config_rejects_unknown_distillation_mode():
    with pytest.raises(ValueError, match="distillation_mode"):
        OnPolicyDistillationConfig(distillation_mode="bogus")
    with pytest.raises(ValueError, match="trust_region_top_k"):
        OnPolicyDistillationConfig(distillation_mode="trust_region", trust_region_top_k=0)


@pytest.mark.cpu_ok
def test_config_trust_region_requires_a_real_sampling_temperature():
    config = OnPolicyDistillationConfig(distillation_mode="trust_region", temperature=2.0, temperature_sampling=0.6)
    assert config.trust_region_sampling_temperature == pytest.approx(0.6)
    with pytest.raises(ValueError, match="sampling temperature"):
        OnPolicyDistillationConfig(distillation_mode="trust_region", temperature_sampling=0.0)
    with pytest.raises(ValueError, match="generation_do_sample"):
        OnPolicyDistillationConfig(distillation_mode="trust_region", generation_do_sample=False)
