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

"""Value, gradient, masking, dtype, sharding and registry tests for fused_renyi_entropy (XLA).

The reference is an independent float64 NumPy implementation with hand-derived analytic
gradients (no JAX, no ejkernel dispatch).
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.modules.operations import FusedRenyiEntropy, fused_renyi_entropy
from jax.sharding import NamedSharding, PartitionSpec

VALUE_TOL = {"rtol": 2e-4, "atol": 2e-5}


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _ref_rows(logits, alpha, temperature):
    """Float64 per-row ``H_alpha(softmax(z / T))`` and its gradient w.r.t. ``z``."""
    u = np.asarray(logits, np.float64) / temperature
    # Stable log-softmax: at large logits q underflows to exactly 0 and np.log(q) would be -inf.
    shifted = u - u.max(-1, keepdims=True)
    log_q = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    q = np.exp(log_q)
    if alpha == 1.0:
        h = -(q * log_q).sum(-1)
        d_u = -q * (log_q + h[..., None])
    else:
        s = (q**alpha).sum(-1)
        h = np.log(s) / (1.0 - alpha)
        d_u = alpha * (q**alpha - q * s[..., None]) / (s[..., None] * (1.0 - alpha))
    return h, d_u / temperature


def _ref_reduce(row, d_row, weights, reduction):
    w = np.asarray(weights, np.float64)
    weighted, d_weighted = row * w, d_row * w[..., None]
    if reduction == "none":
        return weighted, d_weighted
    if reduction == "sum":
        return weighted.sum(), d_weighted
    denom = max(w.sum(), 1e-8)
    return weighted.sum() / denom, d_weighted / denom


def _fixture(seed=0, shape=(3, 7, 33)):
    kz, kw = jax.random.split(jax.random.PRNGKey(seed))
    logits = jax.random.normal(kz, shape, jnp.float32) * 1.5
    weights = (jax.random.uniform(kw, shape[:-1]) > 0.3).astype(jnp.float32)
    return logits, weights


def _entropy(logits, weights=None, **kwargs):
    return fused_renyi_entropy(logits, weights, platform="xla", **kwargs).entropy


def test_registry_lookup_and_signatures():
    impl = kernel_registry.get("fused_renyi_entropy", platform=Platform.XLA, backend=Backend.ANY)
    assert impl is not None
    assert kernel_registry.validate_signatures("fused_renyi_entropy")


def test_shard_map_wrapper_defined_on_class():
    assert "create_shard_map_wrapper" in FusedRenyiEntropy.__dict__


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("temperature", [1.0, 2.5])
def test_value_and_grad_match_reference(temperature, alpha):
    logits, weights = _fixture()
    row, d_row = _ref_rows(logits, alpha, temperature)
    want, want_grad = _ref_reduce(row, d_row, weights, "mean")

    out = fused_renyi_entropy(logits, weights, alpha=alpha, temperature=temperature, platform="xla")
    grad = jax.grad(lambda z: _entropy(z, weights, alpha=alpha, temperature=temperature))(logits)

    np.testing.assert_allclose(float(out.entropy), want, **VALUE_TOL)
    np.testing.assert_allclose(np.asarray(grad), want_grad, **VALUE_TOL)
    assert float(out.weight_sum) == float(weights.sum())


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_reductions_match_reference(reduction):
    logits, weights = _fixture(seed=7)
    row, d_row = _ref_rows(logits, 2.0, 1.0)
    want, want_grad = _ref_reduce(row, d_row, weights, reduction)
    got = _entropy(logits, weights, alpha=2.0, reduction=reduction)
    assert got.shape == (weights.shape if reduction == "none" else ())
    np.testing.assert_allclose(np.asarray(got), want, **VALUE_TOL)
    if reduction != "none":
        grad = jax.grad(lambda z: _entropy(z, weights, alpha=2.0, reduction=reduction))(logits)
        np.testing.assert_allclose(np.asarray(grad), want_grad, **VALUE_TOL)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_uniform_distribution_gives_log_vocab(alpha):
    out = _entropy(jnp.zeros((2, 3, 16), jnp.float32), alpha=alpha)
    np.testing.assert_allclose(float(out), np.log(16.0), atol=1e-6)


def test_one_hot_distribution_gives_zero_entropy():
    logits = jnp.zeros((2, 3, 16), jnp.float32).at[..., 4].set(80.0)
    for alpha in (0.5, 1.0, 2.0):
        np.testing.assert_allclose(float(_entropy(logits, alpha=alpha)), 0.0, atol=1e-5)


def test_alpha_near_one_approaches_shannon():
    logits, weights = _fixture(seed=2)
    shannon = float(_entropy(logits, weights, alpha=1.0))
    near = float(_entropy(logits, weights, alpha=1.0 + 1e-3))
    np.testing.assert_allclose(near, shannon, rtol=5e-3)


def test_renyi_monotone_nonincreasing_in_alpha():
    logits, weights = _fixture(seed=3)
    values = [float(_entropy(logits, weights, alpha=a)) for a in (0.5, 1.0, 2.0, 8.0)]
    assert all(a >= b - 1e-6 for a, b in itertools.pairwise(values))


def test_min_entropy_limit_matches_negative_log_max_probability():
    logits, weights = _fixture(seed=5)
    p_max = _softmax(np.asarray(logits, np.float64)).max(-1)
    want = (-np.log(p_max) * np.asarray(weights)).sum() / max(float(weights.sum()), 1e-8)
    np.testing.assert_allclose(float(_entropy(logits, weights, alpha=256.0)), want, atol=2e-2)


def test_zero_weight_rows_have_no_value_and_no_gradient():
    logits, weights = _fixture(seed=4)
    weights = weights.at[0].set(0.0)
    base = _entropy(logits, weights, alpha=2.0)
    shifted = _entropy(logits.at[0].mul(-5.0), weights, alpha=2.0)
    np.testing.assert_allclose(float(base), float(shifted), rtol=1e-6)
    grad = jax.grad(lambda z: _entropy(z, weights, alpha=2.0))(logits)
    assert float(jnp.abs(grad[0]).max()) == 0.0
    assert float(jnp.abs(grad[1:]).max()) > 0.0


def test_all_zero_weights_give_zero():
    logits, _ = _fixture(seed=6)
    zeros = jnp.zeros(logits.shape[:-1], jnp.float32)
    assert float(_entropy(logits, zeros)) == 0.0
    assert float(jnp.abs(jax.grad(lambda z: _entropy(z, zeros))(logits)).max()) == 0.0


def test_attention_mask_is_folded_into_weights():
    logits, weights = _fixture(seed=8)
    mask = (jnp.arange(logits.shape[1]) < 4).astype(jnp.int32)[None, :] * jnp.ones((logits.shape[0], 1), jnp.int32)
    folded = fused_renyi_entropy(logits, weights, attention_mask=mask, alpha=2.0, platform="xla")
    explicit = fused_renyi_entropy(logits, weights * mask, alpha=2.0, platform="xla")
    np.testing.assert_allclose(float(folded.entropy), float(explicit.entropy), rtol=1e-6)
    assert float(folded.weight_sum) == float((weights * mask).sum())


def test_row_chunking_and_leading_shape_invariance():
    logits, weights = _fixture(seed=9)
    full = _entropy(logits, weights, alpha=2.0, reduction="sum")
    chunked = sum(_entropy(logits[:, c], weights[:, c], alpha=2.0, reduction="sum") for c in (slice(0, 3), slice(3, 7)))
    np.testing.assert_allclose(float(full), float(chunked), rtol=1e-5)
    flat = _entropy(logits.reshape(-1, 33), weights.reshape(-1), alpha=2.0, reduction="sum")
    np.testing.assert_allclose(float(full), float(flat), rtol=1e-6)


def test_large_logits_stay_finite():
    logits, weights = _fixture(seed=10)
    logits = logits * 100.0
    for alpha in (0.5, 1.0, 2.0):
        row, d_row = _ref_rows(logits, alpha, 1.0)
        want, want_grad = _ref_reduce(row, d_row, weights, "mean")
        got = _entropy(logits, weights, alpha=alpha)
        grad = jax.grad(lambda z, a=alpha: _entropy(z, weights, alpha=a))(logits)
        assert bool(jnp.isfinite(got)) and bool(jnp.isfinite(grad).all())
        np.testing.assert_allclose(float(got), want, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.asarray(grad), want_grad, rtol=1e-3, atol=1e-3)


def test_bfloat16_inputs_accumulate_in_float32():
    logits, weights = _fixture(seed=11)
    z16 = logits.astype(jnp.bfloat16)
    z_ref = np.asarray(z16.astype(jnp.float32))
    row, d_row = _ref_rows(z_ref, 2.0, 2.0)
    want, want_grad = _ref_reduce(row, d_row, weights, "mean")
    got = _entropy(z16, weights, alpha=2.0, temperature=2.0)
    grad = jax.grad(lambda z: _entropy(z, weights, alpha=2.0, temperature=2.0))(z16)
    assert got.dtype == jnp.float32
    assert grad.dtype == jnp.bfloat16
    np.testing.assert_allclose(float(got), want, **VALUE_TOL)
    np.testing.assert_allclose(np.asarray(grad.astype(jnp.float32)), want_grad, rtol=2e-2, atol=2e-4)


def test_invalid_args_raise():
    logits, weights = _fixture(seed=12)
    with pytest.raises(ValueError, match="alpha"):
        fused_renyi_entropy(logits, weights, alpha=0.0, platform="xla")
    with pytest.raises(ValueError, match="alpha"):
        fused_renyi_entropy(logits, weights, alpha=-1.0, platform="xla")
    with pytest.raises(ValueError, match="reduction"):
        fused_renyi_entropy(logits, weights, reduction="avg", platform="xla")
    with pytest.raises(ValueError, match="temperature"):
        fused_renyi_entropy(logits, weights, temperature=0.0, platform="xla")
    with pytest.raises(ValueError, match=r"weights\.shape"):
        fused_renyi_entropy(logits, weights[:, :5], platform="xla")


@pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices for the (dp, tp) mesh")
@pytest.mark.parametrize("mesh_shape", [(2, 2), (1, 4)])
@pytest.mark.parametrize("vocab_sharded", [True, False])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_shard_map_matches_reference(mesh_shape, vocab_sharded, alpha, reduction):
    mesh = jax.make_mesh(mesh_shape, ("dp", "tp"))
    logit_spec = PartitionSpec("dp", None, "tp" if vocab_sharded else None)
    token_spec = PartitionSpec("dp", None)
    logits, weights = _fixture(seed=13, shape=(4, 6, 32))
    row, d_row = _ref_rows(logits, alpha, 1.5)
    want, want_grad = _ref_reduce(row, d_row, weights, reduction)

    logits_sh = jax.device_put(logits, NamedSharding(mesh, logit_spec))
    weights_sh = jax.device_put(weights, NamedSharding(mesh, token_spec))

    def _sharded(z):
        return fused_renyi_entropy(
            z,
            weights_sh,
            alpha=alpha,
            reduction=reduction,
            temperature=1.5,
            platform="xla",
            mesh=mesh,
            in_specs=(logit_spec, token_spec),
            out_specs=PartitionSpec(),
        ).entropy

    with jax.set_mesh(mesh):
        value = _sharded(logits_sh)
        grad = jax.grad(_sharded)(logits_sh)

    np.testing.assert_allclose(float(value), want, **VALUE_TOL)
    np.testing.assert_allclose(np.asarray(grad), want_grad, **VALUE_TOL)
