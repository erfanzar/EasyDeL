# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""``GatedResidual`` (Qwen4 hyper-connection) parity against the reference math.

The reference (HF ``Qwen4ExpTextGatedResidual``) computes, all from the
*grouped* normed streams ``h = hc_norm(x)`` (per-``hidden``-group RMSNorm with
a zero-centred ``(1 + w)`` scale):

- read:  ``mixed = mean_s( sigmoid(W_up @ silu(W_down @ h / hc)) * h )``
- write: ``inject = 2 * sigmoid(W_inj @ h / hc)``; caller applies
  ``x + flatten(inject_i * y)`` for sub-layer output ``y``.

These tests port that to NumPy with fixed weights and require the JAX module
to match it, for both the combining (sub-layer) form and the ``use_combine``
``False`` (model-entry/exit mixer) form.
"""

import numpy as np
import pytest
import spectrax as spx
from easydel.layers.residual import GatedResidual, expand_streams, inject_streams
from jax import numpy as jnp

HIDDEN = 8
HC = 4
LOWRANK = 3
EPS = 1e-6


def _rng(seed):
    return np.random.default_rng(seed)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _reference_norm(x, weight, group_size, eps=EPS):
    """HF Qwen4ExpTextRMSNorm: grouped, fp32, (1 + w) scale."""
    xf = x.astype(np.float32)
    g = xf.reshape(*xf.shape[:-1], -1, group_size)
    out = g / np.sqrt(np.mean(g**2, axis=-1, keepdims=True) + eps)
    out = out.reshape(xf.shape)
    return (out * (1.0 + weight.astype(np.float32))).astype(x.dtype)


def _reference_forward(x, w_norm, w_down, w_up, w_inject):
    """NumPy port of the reference forward. Weights are HF-layout [out, in]."""
    h = _reference_norm(x, w_norm, HIDDEN)
    mix = _silu(h @ w_down.T / HC)
    mix = _sigmoid(mix @ w_up.T)
    mix = mix.reshape(*mix.shape[:-1], HC, HIDDEN)
    mixed = (mix * h.reshape(*h.shape[:-1], HC, HIDDEN)).mean(axis=-2)
    if w_inject is None:
        return mixed
    inject = 2.0 * _sigmoid(h @ w_inject.T / HC)
    return mixed, inject


def _make_module(use_combine, seed=0):
    rng = _rng(seed)
    module = GatedResidual(
        hidden_size=HIDDEN,
        hc_count=HC,
        hc_lowrank=LOWRANK,
        eps=EPS,
        use_combine=use_combine,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(seed),
    )
    w_norm = rng.standard_normal(HC * HIDDEN).astype(np.float32) * 0.02
    w_down = rng.standard_normal((LOWRANK, HC * HIDDEN)).astype(np.float32) * 0.05
    w_up = rng.standard_normal((HC * HIDDEN, LOWRANK)).astype(np.float32) * 0.05
    module.hc_norm.weight.value = jnp.asarray(w_norm)
    module.input_mix_weight_down.weight.value = jnp.asarray(w_down.T)  # HF [out,in] -> kernel [in,out]
    module.input_mix_weight_up.weight.value = jnp.asarray(w_up.T)
    w_inject = None
    if use_combine:
        w_inject = rng.standard_normal((HC, HC * HIDDEN)).astype(np.float32) * 0.05
        module.block_inject_weight.weight.value = jnp.asarray(w_inject.T)
    return module, w_norm, w_down, w_up, w_inject


def test_read_side_matches_reference():
    module, w_norm, w_down, w_up, w_inject = _make_module(use_combine=True)
    x = jnp.asarray(_rng(1).standard_normal((2, 5, HC * HIDDEN)), jnp.float32)

    mixed, hyper, inject = module(x)
    want_mixed, want_inject = _reference_forward(np.asarray(x), w_norm, w_down, w_up, w_inject)

    np.testing.assert_allclose(np.asarray(mixed), want_mixed, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(inject), want_inject, rtol=1e-5, atol=1e-6)
    assert hyper is not None and jnp.array_equal(hyper, x)


def test_no_combine_form_has_no_inject_weight_and_returns_only_mixed():
    module, w_norm, w_down, w_up, _ = _make_module(use_combine=False, seed=3)
    assert module.block_inject_weight is None
    x = jnp.asarray(_rng(2).standard_normal((2, 3, HC * HIDDEN)), jnp.float32)

    mixed = module(x)
    want = _reference_forward(np.asarray(x), w_norm, w_down, w_up, None)
    np.testing.assert_allclose(np.asarray(mixed), want, rtol=1e-5, atol=1e-6)


def test_zero_init_is_identity_mean_read():
    """At zero-init the mixer weights give sigmoid(0)=0.5 gates: mixed = mean(norm(x))/2-ish."""
    module = GatedResidual(
        hidden_size=HIDDEN,
        hc_count=HC,
        hc_lowrank=LOWRANK,
        use_combine=True,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    # zero the low-rank path and the inject path; hc_norm weight already zero
    module.input_mix_weight_down.weight.value = jnp.zeros((HC * HIDDEN, LOWRANK), jnp.float32)
    module.input_mix_weight_up.weight.value = jnp.zeros((LOWRANK, HC * HIDDEN), jnp.float32)
    module.block_inject_weight.weight.value = jnp.zeros((HC * HIDDEN, HC), jnp.float32)
    x = jnp.asarray(_rng(4).standard_normal((1, 4, HC * HIDDEN)), jnp.float32)

    mixed, _hyper, inject = module(x)
    normed = module.hc_norm(x)
    want = (0.5 * normed.reshape(1, 4, HC, HIDDEN)).mean(axis=-2)
    np.testing.assert_allclose(np.asarray(mixed), np.asarray(want), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(inject), np.ones((1, 4, HC)), rtol=1e-6, atol=1e-7)


def test_inject_streams_matches_reference_write():
    x = jnp.asarray(_rng(5).standard_normal((2, 3, HC * HIDDEN)), jnp.float32)
    y = jnp.asarray(_rng(6).standard_normal((2, 3, HIDDEN)), jnp.float32)
    w = jnp.asarray(_rng(7).random((2, 3, HC), dtype=np.float32), jnp.float32)

    got = inject_streams(x, y, w)
    want = np.asarray(x) + (np.asarray(y)[..., None, :] * np.asarray(w)[..., None]).reshape(2, 3, HC * HIDDEN)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6, atol=1e-7)


def test_expand_streams_repeats_last_axis():
    x = jnp.asarray(np.arange(6, dtype=np.float32).reshape(1, 2, 3))
    got = expand_streams(x, 4)
    assert got.shape == (1, 2, 12)
    np.testing.assert_array_equal(np.asarray(got)[0, 0], np.array([0, 1, 2] * 4, dtype=np.float32))


def test_rejects_wrong_trailing_width():
    module, *_ = _make_module(use_combine=True)
    with pytest.raises(ValueError, match="hyper-connection features"):
        module(jnp.zeros((1, 2, HIDDEN), jnp.float32))


def test_rejects_single_stream():
    with pytest.raises(ValueError, match="hc_count"):
        GatedResidual(
            hidden_size=HIDDEN,
            hc_count=1,
            hc_lowrank=LOWRANK,
            param_dtype=jnp.float32,
            dtype=jnp.float32,
            rngs=spx.Rngs(0),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
