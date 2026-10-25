# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Value, gradient, masking, dtype, sharding and registry tests for fused_cakld_tv (XLA).

The reference is an independent float64 NumPy implementation with hand-derived analytic
gradients (no JAX, no ejkernel dispatch), so both the forward value and the autodiff gradient
of the implementation are checked against it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.modules.operations import FusedCakldTv, fused_cakld_tv
from jax.sharding import NamedSharding, PartitionSpec

VALUE_TOL = {"rtol": 2e-4, "atol": 2e-5}


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _renyi_with_grad(u, alpha):
    """Per-row ``H_alpha(softmax(u))`` and ``dH/du`` in float64."""
    q = _softmax(u)
    log_q = np.log(q)
    if alpha == 1.0:
        h = -(q * log_q).sum(-1)
        return h, -q * (log_q + h[..., None])
    s = (q**alpha).sum(-1)
    h = np.log(s) / (1.0 - alpha)
    return h, alpha * (q**alpha - q * s[..., None]) / (s[..., None] * (1.0 - alpha))


def _ref_rows(student, teacher, temperature, tv_weight, entropy_weight=0.0, entropy_alpha=1.0, entropy_gate=None):
    """Float64 per-row loss and its gradient w.r.t. the student logits (teacher and gate detached)."""
    s = np.asarray(student, np.float64)
    t = np.asarray(teacher, np.float64)
    u = s / temperature
    p = _softmax(t / temperature)
    q = _softmax(u)
    a = np.log(q) - np.log(p)
    reverse = (q * a).sum(-1)
    forward = -(p * a).sum(-1)
    gamma = p.max(-1)
    blend = gamma * reverse + (1.0 - gamma) * forward
    d_blend = gamma[..., None] * (q * (a - reverse[..., None])) + (1.0 - gamma)[..., None] * (q - p)

    below = (q < p).astype(np.float64)
    overlap = np.minimum(p, q).sum(-1)
    d_overlap = q * below - q * (q * below).sum(-1, keepdims=True)
    row = (1.0 - tv_weight) * blend + tv_weight * 2.0 * (1.0 - overlap)
    d_row = (1.0 - tv_weight) * d_blend - 2.0 * tv_weight * d_overlap

    if entropy_weight:
        h, d_h = _renyi_with_grad(u, entropy_alpha)
        gate = (gamma > entropy_gate).astype(np.float64) if entropy_gate is not None else np.ones_like(gamma)
        row = row + entropy_weight * gate * h
        d_row = d_row + entropy_weight * gate[..., None] * d_h
    return row * temperature**2, d_row * temperature


def _ref_reduce(row, d_row, weights, reduction):
    w = np.asarray(weights, np.float64)
    weighted, d_weighted = row * w, d_row * w[..., None]
    if reduction == "none":
        return weighted, d_weighted
    if reduction == "sum":
        return weighted.sum(), d_weighted
    denom = max(w.sum(), 1e-8)
    return weighted.sum() / denom, d_weighted / denom


def _fixture(seed=0, shape=(4, 9, 33), teacher_scale=1.4):
    ks, kt, kw = jax.random.split(jax.random.PRNGKey(seed), 3)
    student = jax.random.normal(ks, shape, jnp.float32)
    teacher = jax.random.normal(kt, shape, jnp.float32) * teacher_scale
    weights = (jax.random.uniform(kw, shape[:-1]) > 0.3).astype(jnp.float32)
    return student, teacher, weights


def _loss(student, teacher, weights, **kwargs):
    return fused_cakld_tv(student, teacher, weights, platform="xla", **kwargs).loss


def test_registry_lookup_and_signatures():
    impl = kernel_registry.get("fused_cakld_tv", platform=Platform.XLA, backend=Backend.ANY)
    assert impl is not None
    assert kernel_registry.validate_signatures("fused_cakld_tv")


def test_shard_map_wrapper_defined_on_class():
    assert "create_shard_map_wrapper" in FusedCakldTv.__dict__


@pytest.mark.parametrize("tv_weight", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("temperature", [1.0, 2.5])
def test_value_and_grad_match_reference(temperature, tv_weight):
    student, teacher, weights = _fixture()
    row, d_row = _ref_rows(student, teacher, temperature, tv_weight)
    want, want_grad = _ref_reduce(row, d_row, weights, "mean")

    out = fused_cakld_tv(
        student, teacher, weights, temperature=temperature, tv_weight=tv_weight, platform="xla", reduction="mean"
    )
    got_grad = jax.grad(lambda s: _loss(s, teacher, weights, temperature=temperature, tv_weight=tv_weight))(student)

    np.testing.assert_allclose(float(out.loss), want, **VALUE_TOL)
    np.testing.assert_allclose(np.asarray(got_grad), want_grad, **VALUE_TOL)
    assert float(out.weight_sum) == float(weights.sum())


def test_dynamic_gate_limits():
    """A near-one-hot teacher gives gamma -> 1 (pure reverse KL); a uniform teacher gives gamma = 1/V."""
    student, _, weights = _fixture(seed=1)
    teacher = jnp.zeros_like(student).at[..., 0].set(60.0)
    got = _loss(student, teacher, weights, reduction="none")
    q = _softmax(np.asarray(student, np.float64))
    p = _softmax(np.asarray(teacher, np.float64))
    reverse = (q * (np.log(q) - np.log(p))).sum(-1)
    np.testing.assert_allclose(np.asarray(got), reverse * np.asarray(weights), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("entropy_alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("use_gate", [False, True])
def test_entropy_penalty_matches_reference(entropy_alpha, use_gate):
    student, teacher, weights = _fixture(seed=17, teacher_scale=3.0)
    temperature, tv_weight, lam = 1.0, 0.25, 0.03
    gamma = _softmax(np.asarray(teacher, np.float64) / temperature).max(-1)
    gate = float(np.median(gamma)) if use_gate else None
    if use_gate:  # both branches of the hard gate are exercised
        assert 0 < (gamma > gate).sum() < gamma.size

    row, d_row = _ref_rows(student, teacher, temperature, tv_weight, lam, entropy_alpha, gate)
    want, want_grad = _ref_reduce(row, d_row, weights, "mean")
    kwargs = dict(
        temperature=temperature, tv_weight=tv_weight, entropy_weight=lam, entropy_alpha=entropy_alpha, entropy_gate=gate
    )
    got = _loss(student, teacher, weights, **kwargs)
    got_grad = jax.grad(lambda s: _loss(s, teacher, weights, **kwargs))(student)
    np.testing.assert_allclose(float(got), want, **VALUE_TOL)
    np.testing.assert_allclose(np.asarray(got_grad), want_grad, **VALUE_TOL)


def test_entropy_weight_zero_is_exact_noop():
    student, teacher, weights = _fixture(seed=19)
    base = _loss(student, teacher, weights, temperature=2.0, tv_weight=0.25)
    twin = _loss(student, teacher, weights, temperature=2.0, tv_weight=0.25, entropy_weight=0.0, entropy_alpha=2.0)
    assert jnp.array_equal(base, twin)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_reductions_match_reference(reduction):
    student, teacher, weights = _fixture(seed=3)
    row, d_row = _ref_rows(student, teacher, 2.0, 0.25)
    want, want_grad = _ref_reduce(row, d_row, weights, reduction)
    got = _loss(student, teacher, weights, reduction=reduction, temperature=2.0, tv_weight=0.25)
    assert got.shape == (weights.shape if reduction == "none" else ())
    np.testing.assert_allclose(np.asarray(got), want, **VALUE_TOL)
    if reduction != "none":
        grad = jax.grad(lambda s: _loss(s, teacher, weights, reduction=reduction, temperature=2.0, tv_weight=0.25))(
            student
        )
        np.testing.assert_allclose(np.asarray(grad), want_grad, **VALUE_TOL)


def test_teacher_is_detached():
    student, teacher, weights = _fixture(seed=5)
    grad_teacher = jax.grad(lambda t: _loss(student, t, weights, temperature=2.0, tv_weight=0.25, entropy_weight=0.1))(
        teacher
    )
    assert float(jnp.abs(grad_teacher).max()) == 0.0


def test_zero_weight_rows_have_no_loss_and_no_gradient():
    student, teacher, weights = _fixture(seed=7)
    weights = weights.at[0, :].set(0.0)
    kwargs = dict(temperature=2.0, tv_weight=0.25, entropy_weight=0.05)
    base = _loss(student, teacher, weights, **kwargs)
    shifted = _loss(student.at[0].add(37.0), teacher.at[0].mul(-3.0), weights, **kwargs)
    np.testing.assert_allclose(float(base), float(shifted), rtol=1e-6)
    grad = jax.grad(lambda s: _loss(s, teacher, weights, **kwargs))(student)
    assert float(jnp.abs(grad[0]).max()) == 0.0
    assert float(jnp.abs(grad[1:]).max()) > 0.0


def test_all_zero_weights_give_zero_loss_and_grad():
    student, teacher, _ = _fixture(seed=8)
    zeros = jnp.zeros(student.shape[:-1], jnp.float32)
    assert float(_loss(student, teacher, zeros)) == 0.0
    assert float(jnp.abs(jax.grad(lambda s: _loss(s, teacher, zeros))(student)).max()) == 0.0


def test_attention_mask_is_folded_into_weights():
    student, teacher, weights = _fixture(seed=9)
    mask = (jnp.arange(student.shape[1]) < 5).astype(jnp.int32)[None, :] * jnp.ones((student.shape[0], 1), jnp.int32)
    folded = fused_cakld_tv(student, teacher, weights, attention_mask=mask, platform="xla", tv_weight=0.25)
    explicit = fused_cakld_tv(student, teacher, weights * mask, platform="xla", tv_weight=0.25)
    np.testing.assert_allclose(float(folded.loss), float(explicit.loss), rtol=1e-6)
    assert float(folded.weight_sum) == float((weights * mask).sum())


def test_row_chunking_invariance():
    """Summing per-chunk results reproduces the full result, for the value and the gradient."""
    student, teacher, weights = _fixture(seed=10)
    kwargs = dict(temperature=2.0, tv_weight=0.25, entropy_weight=0.02, entropy_alpha=2.0)
    chunks = [slice(0, 3), slice(3, 6), slice(6, 9)]

    full_sum = _loss(student, teacher, weights, reduction="sum", **kwargs)
    chunk_sum = sum(_loss(student[:, c], teacher[:, c], weights[:, c], reduction="sum", **kwargs) for c in chunks)
    np.testing.assert_allclose(float(full_sum), float(chunk_sum), rtol=1e-5)

    full_grad = jax.grad(lambda s: _loss(s, teacher, weights, reduction="sum", **kwargs))(student)
    chunk_grad = jnp.concatenate(
        [
            jax.grad(lambda s, c=c: _loss(s, teacher[:, c], weights[:, c], reduction="sum", **kwargs))(student[:, c])
            for c in chunks
        ],
        axis=1,
    )
    np.testing.assert_allclose(np.asarray(full_grad), np.asarray(chunk_grad), rtol=1e-5, atol=1e-6)

    # The "mean" recombination identity: loss * weight_sum is the weighted sum.
    out = fused_cakld_tv(student, teacher, weights, platform="xla", **kwargs)
    np.testing.assert_allclose(float(out.loss * out.weight_sum), float(full_sum), rtol=1e-5)


def test_leading_shape_invariance():
    student, teacher, weights = _fixture(seed=11)
    kwargs = dict(temperature=1.5, tv_weight=0.5)
    nested = _loss(student, teacher, weights, **kwargs)
    flat = _loss(student.reshape(-1, 33), teacher.reshape(-1, 33), weights.reshape(-1), **kwargs)
    np.testing.assert_allclose(float(nested), float(flat), rtol=1e-6)


def test_large_logits_stay_finite_and_match_reference():
    student, teacher, weights = _fixture(seed=12)
    student, teacher = student * 60.0, teacher * 60.0
    row, d_row = _ref_rows(student, teacher, 1.0, 0.25)
    want, want_grad = _ref_reduce(row, d_row, weights, "mean")
    got = _loss(student, teacher, weights, tv_weight=0.25)
    got_grad = jax.grad(lambda s: _loss(s, teacher, weights, tv_weight=0.25))(student)
    assert bool(jnp.isfinite(got)) and bool(jnp.isfinite(got_grad).all())
    np.testing.assert_allclose(float(got), want, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.asarray(got_grad), want_grad, rtol=1e-3, atol=1e-3)


def test_bfloat16_inputs():
    student, teacher, weights = _fixture(seed=13)
    s16, t16 = student.astype(jnp.bfloat16), teacher.astype(jnp.bfloat16)
    s_ref, t_ref = np.asarray(s16.astype(jnp.float32)), np.asarray(t16.astype(jnp.float32))
    row, d_row = _ref_rows(s_ref, t_ref, 2.0, 0.25)
    want, want_grad = _ref_reduce(row, d_row, weights, "mean")

    got = _loss(s16, t16, weights, temperature=2.0, tv_weight=0.25)
    grad = jax.grad(lambda s: _loss(s, t16, weights, temperature=2.0, tv_weight=0.25))(s16)
    assert got.dtype == jnp.float32  # fp32 accumulation, not the logits dtype
    assert grad.dtype == jnp.bfloat16  # the gradient follows the differentiated input
    np.testing.assert_allclose(float(got), want, **VALUE_TOL)
    np.testing.assert_allclose(np.asarray(grad.astype(jnp.float32)), want_grad, rtol=2e-2, atol=2e-4)

    # A float32 teacher against a bfloat16 student keeps the teacher's precision.
    row, d_row = _ref_rows(s_ref, teacher, 2.0, 0.25)
    want_mixed, _ = _ref_reduce(row, d_row, weights, "mean")
    np.testing.assert_allclose(
        float(_loss(s16, teacher, weights, temperature=2.0, tv_weight=0.25)), want_mixed, **VALUE_TOL
    )


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_teacher_entropy_matches_reference(reduction):
    student, teacher, weights = _fixture(seed=15)
    temperature = 2.0
    out = fused_cakld_tv(
        student,
        teacher,
        weights,
        reduction=reduction,
        temperature=temperature,
        tv_weight=0.25,
        platform="xla",
        return_teacher_entropy=True,
    )
    p = _softmax(np.asarray(teacher, np.float64) / temperature)
    entropy = -(p * np.log(p)).sum(-1) * temperature**2
    want, _ = _ref_reduce(entropy, np.zeros_like(p), weights, reduction)
    np.testing.assert_allclose(np.asarray(out.teacher_entropy), want, **VALUE_TOL)
    assert fused_cakld_tv(student, teacher, weights, platform="xla").teacher_entropy is None


def test_invalid_args_raise():
    student, teacher, weights = _fixture(seed=21)
    with pytest.raises(ValueError, match="tv_weight"):
        fused_cakld_tv(student, teacher, weights, tv_weight=1.5, platform="xla")
    with pytest.raises(ValueError, match="tv_weight"):
        fused_cakld_tv(student, teacher, weights, tv_weight=-0.1, platform="xla")
    with pytest.raises(ValueError, match="reduction"):
        fused_cakld_tv(student, teacher, weights, reduction="avg", platform="xla")
    with pytest.raises(ValueError, match="temperature"):
        fused_cakld_tv(student, teacher, weights, temperature=0.0, platform="xla")
    with pytest.raises(ValueError, match="entropy_weight"):
        fused_cakld_tv(student, teacher, weights, entropy_weight=-0.1, platform="xla")
    with pytest.raises(ValueError, match="entropy_alpha"):
        fused_cakld_tv(student, teacher, weights, entropy_weight=0.03, entropy_alpha=0.0, platform="xla")
    with pytest.raises(ValueError, match="entropy_gate"):
        fused_cakld_tv(student, teacher, weights, entropy_weight=0.03, entropy_gate=1.5, platform="xla")
    with pytest.raises(ValueError, match="shape mismatch"):
        fused_cakld_tv(student, teacher[:, :8], weights, platform="xla")
    with pytest.raises(ValueError, match=r"weights\.shape"):
        fused_cakld_tv(student, teacher, weights[:, :8], platform="xla")
    with pytest.raises((ValueError, TypeError)):
        fused_cakld_tv(student, teacher[..., :32], weights, platform="xla")


@pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices for the (dp, tp) mesh")
@pytest.mark.parametrize("mesh_shape", [(2, 2), (1, 4)])
@pytest.mark.parametrize("vocab_sharded", [True, False])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_shard_map_matches_reference(mesh_shape, vocab_sharded, reduction):
    mesh = jax.make_mesh(mesh_shape, ("dp", "tp"))
    logit_spec = PartitionSpec("dp", None, "tp" if vocab_sharded else None)
    token_spec = PartitionSpec("dp", None)
    student, teacher, weights = _fixture(seed=23, shape=(4, 6, 32), teacher_scale=3.0)
    temperature, tv_weight, lam, alpha = 2.5, 0.25, 0.03, 2.0
    gamma = _softmax(np.asarray(teacher, np.float64) / temperature).max(-1)
    gate = float(np.median(gamma))
    row, d_row = _ref_rows(student, teacher, temperature, tv_weight, lam, alpha, gate)
    want, want_grad = _ref_reduce(row, d_row, weights, reduction)

    student_sh = jax.device_put(student, NamedSharding(mesh, logit_spec))
    teacher_sh = jax.device_put(teacher, NamedSharding(mesh, logit_spec))
    weights_sh = jax.device_put(weights, NamedSharding(mesh, token_spec))

    def _sharded(s):
        return fused_cakld_tv(
            s,
            teacher_sh,
            weights_sh,
            reduction=reduction,
            temperature=temperature,
            tv_weight=tv_weight,
            entropy_weight=lam,
            entropy_alpha=alpha,
            entropy_gate=gate,
            platform="xla",
            mesh=mesh,
            in_specs=(logit_spec, logit_spec, token_spec),
            out_specs=PartitionSpec(),
        ).loss

    with jax.set_mesh(mesh):
        loss = _sharded(student_sh)
        grad = jax.grad(_sharded)(student_sh)

    np.testing.assert_allclose(float(loss), want, **VALUE_TOL)
    np.testing.assert_allclose(np.asarray(grad), want_grad, **VALUE_TOL)


@pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices for the (dp, tp) mesh")
def test_shard_map_per_row_output_and_attention_mask():
    mesh = jax.make_mesh((2, 2), ("dp", "tp"))
    logit_spec = PartitionSpec("dp", None, "tp")
    token_spec = PartitionSpec("dp", None)
    student, teacher, weights = _fixture(seed=25, shape=(4, 6, 32))
    mask = (jnp.arange(6) % 2).astype(jnp.float32)[None, :] * jnp.ones((4, 1), jnp.float32)
    row, d_row = _ref_rows(student, teacher, 2.0, 0.25)
    want, _ = _ref_reduce(row, d_row, np.asarray(weights) * np.asarray(mask), "none")

    with jax.set_mesh(mesh):
        out = fused_cakld_tv(
            jax.device_put(student, NamedSharding(mesh, logit_spec)),
            jax.device_put(teacher, NamedSharding(mesh, logit_spec)),
            jax.device_put(weights, NamedSharding(mesh, token_spec)),
            attention_mask=jax.device_put(mask, NamedSharding(mesh, token_spec)),
            reduction="none",
            temperature=2.0,
            tv_weight=0.25,
            platform="xla",
            mesh=mesh,
            in_specs=(logit_spec, logit_spec, token_spec, token_spec),
            out_specs=token_spec,
        )
    np.testing.assert_allclose(np.asarray(out.loss), want, **VALUE_TOL)
