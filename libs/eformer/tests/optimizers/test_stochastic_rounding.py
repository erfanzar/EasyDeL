# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Accelerator tests for exact bitwise BF16 / narrow-float stochastic rounding in eFormer."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from eformer.optimizers import (
    OptimizerFactory,
    SchedulerConfig,
    optax_stochastic_round_updates,
    stochastic_cast,
    stochastic_cast_bf16_with_bits,
)


def test_stochastic_cast_bfloat16_exact_values_unperturbed():
    """Already-exact BF16 values (including +-0.0 and power-of-2 boundaries) suffer 0 perturbation."""
    exact_vals = jnp.array(
        [0.0, -0.0, 1.0, -1.0, 2.0, 1.9921875, -3.0, 0.0078125, 1e-38, -1e-38],
        dtype=jnp.bfloat16,
    )
    tiled = jnp.broadcast_to(exact_vals[:, None], (exact_vals.shape[0], 4096)).astype(jnp.float32)
    rounded = stochastic_cast(tiled, jnp.bfloat16, jax.random.key(123))
    assert rounded.dtype == jnp.bfloat16
    assert bool(jnp.all(rounded == exact_vals[:, None]))


def test_stochastic_cast_bfloat16_unbiased_across_regimes():
    """Verify E[stochastic_cast(x, bf16)] == x across sub-ULP, negative, and power-of-2 boundary values."""
    # BF16 ULP on [1, 2) is 2^-7 = 7.8125e-3; half-ULP is 3.90625e-3.
    # Every test value below sits strictly inside a single BF16 ULP interval where RTN is 100% biased.
    targets = jnp.array(
        [
            1.0 + 1e-4,  # tiny sub-ULP positive
            1.0 + 0.00390625,  # exact midpoint (1 + 2^-8)
            1.0 - 1e-4,  # across power-of-2 boundary below 1.0
            -3.0 + 3e-4,  # LoopGate bias=-3.0 + 10x LR step
            -1.0 - 2.5e-4,  # sub-ULP negative
        ],
        dtype=jnp.float32,
    )
    n_samples = 65536
    tiled = jnp.broadcast_to(targets[:, None], (targets.shape[0], n_samples))
    sr = stochastic_cast(tiled, jnp.bfloat16, jax.random.key(42)).astype(jnp.float32)
    means = np.asarray(jnp.mean(sr, axis=-1))
    expected = np.asarray(targets)
    np.testing.assert_allclose(means, expected, atol=2.5e-4, rtol=2.5e-4)

    # Confirm deterministic RTN fails badly on the first sub-ULP target (rounds straight back to 1.0).
    assert float(targets[0].astype(jnp.bfloat16).astype(jnp.float32)) == 1.0


def test_stochastic_cast_float16_fp8_specials_and_rejection():
    """Verify float16 nextafter bracketing, fp8 bin ULP, NaN/Inf preservation, and int rejection."""
    # float16 ULP on [1, 2) is 2^-10 = 9.765625e-4; add sub-ULP 1.5e-4.
    x16 = jnp.full((65536,), 1.0 + 1.5e-4, dtype=jnp.float32)
    sr16 = stochastic_cast(x16, jnp.float16, jax.random.key(7))
    assert sr16.dtype == jnp.float16
    assert abs(float(jnp.mean(sr16.astype(jnp.float32))) - (1.0 + 1.5e-4)) < 5e-5

    # float8_e4m3fn ULP on [1, 2) is 2^-3 = 0.125.
    x8 = jnp.full((65536,), 1.0 + 0.03, dtype=jnp.float32)
    sr8 = stochastic_cast(x8, jnp.float8_e4m3fn, jax.random.key(9))
    assert sr8.dtype == jnp.float8_e4m3fn
    assert abs(float(jnp.mean(sr8.astype(jnp.float32))) - 1.03) < 5e-3

    # Special values survive intact without turning into garbage bits.
    specials = jnp.array([jnp.nan, jnp.inf, -jnp.inf, 0.0, -0.0], dtype=jnp.float32)
    out_sp = stochastic_cast(specials, jnp.bfloat16, jax.random.key(11))
    assert bool(jnp.isnan(out_sp[0]))
    assert bool(jnp.isposinf(out_sp[1]))
    assert bool(jnp.isneginf(out_sp[2]))
    assert float(out_sp[3]) == 0.0 and float(out_sp[4]) == 0.0

    with pytest.raises(ValueError, match="floating-point"):
        stochastic_cast(jnp.ones((4,), dtype=jnp.float32), jnp.int8, jax.random.key(0))


def test_stochastic_cast_ste_gradient():
    """Both stochastic_cast and stochastic_cast_bf16_with_bits provide exact STE gradients."""
    x = jnp.linspace(-2.0, 2.0, 32, dtype=jnp.float32)
    key = jax.random.key(5)
    g1 = jax.grad(lambda z: jnp.sum(stochastic_cast(z, jnp.bfloat16, key).astype(jnp.float32)))(x)
    np.testing.assert_allclose(np.asarray(g1), np.ones_like(np.asarray(x)), atol=1e-6)

    bits = jax.random.bits(key, x.shape, dtype=jnp.uint32) >> jnp.uint32(16)
    g2 = jax.grad(lambda z: jnp.sum(stochastic_cast_bf16_with_bits(z, bits).astype(jnp.float32)))(x)
    np.testing.assert_allclose(np.asarray(g2), np.ones_like(np.asarray(x)), atol=1e-6)


def test_bf16_sub_ulp_adamw_converges_with_sr_while_rtn_freezes():
    """At lr=3e-5 on BF16 weights, deterministic RTN freezes 100% while SR tracks target smoothly."""
    lr = 3e-5
    steps = 400
    params_init = {"w": jnp.ones((4096,), dtype=jnp.bfloat16)}
    grad_const = {"w": jnp.ones((4096,), dtype=jnp.bfloat16)}  # constant positive grad => step -lr per step

    sched_cfg = SchedulerConfig(scheduler_type=None, learning_rate=lr)
    tx_rtn, _ = OptimizerFactory.create("adamw", scheduler_config=sched_cfg, stochastic_rounding=False)
    tx_sr, _ = OptimizerFactory.create(
        "adamw",
        scheduler_config=sched_cfg,
        stochastic_rounding=True,
        stochastic_rounding_seed=2026,
    )

    def _run(tx, init_p):
        st0 = tx.init(init_p)

        @jax.jit
        def _scan_loop(p0, s0):
            def _body(carry, _):
                p, st = carry
                u, st = tx.update(grad_const, st, p)
                p = optax.apply_updates(p, u)
                return (p, st), None

            (p_final, _), _ = jax.lax.scan(_body, (p0, s0), None, length=steps)
            return p_final

        return _scan_loop(init_p, st0)

    p_rtn = _run(tx_rtn, params_init)["w"]
    p_sr = _run(tx_sr, params_init)["w"]
    assert p_rtn.dtype == jnp.bfloat16
    assert p_sr.dtype == jnp.bfloat16

    # Deterministic BF16 RTN is completely pinned at 1.0 (3e-5 << half-ULP 3.9e-3).
    assert float(jnp.max(jnp.abs(p_rtn.astype(jnp.float32) - 1.0))) == 0.0

    # Stochastic rounding tracks the exact analytical trajectory 1.0 - steps * lr = 0.988!
    expected_mean = 1.0 - steps * lr
    actual_mean = float(jnp.mean(p_sr.astype(jnp.float32)))
    assert abs(actual_mean - expected_mean) < 1.5e-3


def test_optax_stochastic_rounding_multisteps_and_f32_passthrough():
    """Verify MultiSteps compatibility, step-varying RNG, and untouched float32 leaves."""
    params = {
        "bf16_leaf": jnp.ones((512,), dtype=jnp.bfloat16),
        "f32_leaf": jnp.ones((16,), dtype=jnp.float32),
    }
    grads = {
        "bf16_leaf": jnp.ones((512,), dtype=jnp.bfloat16),
        "f32_leaf": jnp.ones((16,), dtype=jnp.float32),
    }
    sched_cfg = SchedulerConfig(scheduler_type=None, learning_rate=5e-4)
    tx, _ = OptimizerFactory.create(
        "adamw",
        scheduler_config=sched_cfg,
        gradient_accumulation_steps=2,
        stochastic_rounding=True,
        stochastic_rounding_seed=7,
    )
    st = tx.init(params)
    # Step 0 (microstep 1 of 2): MultiSteps emits zero updates.
    u0, st = tx.update(grads, st, params)
    assert float(jnp.max(jnp.abs(u0["bf16_leaf"]))) == 0.0
    # Step 1 (microstep 2 of 2): real update fires with SR on bf16 and exact f32 update.
    u1, st = tx.update(grads, st, params)
    p1 = optax.apply_updates(params, u1)
    assert p1["bf16_leaf"].dtype == jnp.bfloat16
    assert p1["f32_leaf"].dtype == jnp.float32
    assert float(jnp.mean(p1["bf16_leaf"].astype(jnp.float32))) < 1.0
    np.testing.assert_allclose(np.asarray(p1["f32_leaf"]), 1.0 - 5e-4, atol=1e-5)

    # Verify MPMD stage-local refuses stochastic_rounding loudly rather than silently ignoring it.
    with pytest.raises(NotImplementedError, match="stochastic_rounding"):
        tx.apply_gradients_stage_local(params=params, grads=grads, opt_state=st)

    # Verify standalone optax_stochastic_round_updates changes RNG draws across steps.
    sr_tx = optax_stochastic_round_updates(seed=99)
    sr_st = sr_tx.init(params)
    tiny_u = {"bf16_leaf": jnp.full((512,), 2e-3, dtype=jnp.float32), "f32_leaf": jnp.full((16,), 1e-5)}
    d0, sr_st = sr_tx.update(tiny_u, sr_st, params)
    d1, _ = sr_tx.update(tiny_u, sr_st, params)
    assert not bool(jnp.array_equal(d0["bf16_leaf"], d1["bf16_leaf"]))
