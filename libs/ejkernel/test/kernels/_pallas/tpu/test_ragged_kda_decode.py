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

"""Pallas TPU ``ragged_kda_decode``: one per-channel KDA decode token per row, recurrent-state pool updated in place.

Checked against a float64 recurrence and against the XLA reference (same arithmetic up to the summation order):
advancing rows, idle rows (zero output, slot bit for bit), fresh rows (zero initial state), explicit slots (an
out-of-range slot does not advance; an idle row naming a live row's slot does not clobber it), bf16 and fp32 pools,
and the head-sharded module operation on four chips.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_kda_decode import ragged_kda_decode_pallas
from ejkernel.kernels._xla.ragged_kda_decode._xla_impl import ragged_kda_decode_xla
from ejkernel.modules.operations import ragged_kda_decode_head_sharded
from jax.sharding import Mesh

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="TPU-only Pallas execution")

ROWS, SLOTS, K, V = 16, 24, 128, 128
LOWER = -5.0
IDLE = (3, 11)  # rows without a token this step
FRESH = (0, 5)  # rows that start from a zero state


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _rounded(x, dtype):
    """``x`` rounded to ``dtype`` by a device convert, as float64."""
    return np.asarray(jnp.asarray(np.asarray(x, np.float32)).astype(dtype).astype(jnp.float32), np.float64)


def _case(seed, in_dtype, state_dtype, heads):
    rng = np.random.default_rng(seed)
    return {
        "q": _rounded(rng.standard_normal((ROWS, heads, K)), in_dtype),
        "k": _rounded(rng.standard_normal((ROWS, heads, K)), in_dtype),
        "v": _rounded(rng.standard_normal((ROWS, heads, V)), in_dtype),
        "beta": _rounded(_sigmoid(rng.standard_normal((ROWS, heads))), in_dtype),
        "decay": _rounded(LOWER * _sigmoid(rng.standard_normal((ROWS, heads, K))), jnp.float32),
        "pool": _rounded(0.1 * rng.standard_normal((SLOTS, heads, K, V)), state_dtype),
    }


def _flags():
    live = np.ones(ROWS, bool)
    live[list(IDLE)] = False
    init = np.ones(ROWS, bool)
    init[list(FRESH)] = False
    return live, init


def _reference(case, slots, live, init):
    """float64 decode step; returns ``(out [R, H, V], pool [S, H, K, V])``."""
    heads = case["q"].shape[1]
    q = case["q"] / np.sqrt(np.sum(case["q"] ** 2, axis=-1, keepdims=True) + 1e-6) * K**-0.5
    k = case["k"] / np.sqrt(np.sum(case["k"] ** 2, axis=-1, keepdims=True) + 1e-6)
    out = np.zeros((ROWS, heads, V))
    pool = case["pool"].copy()
    for r in range(ROWS):
        slot = int(slots[r])
        if not live[r] or not 0 <= slot < SLOTS:
            continue
        state = case["pool"][slot] if init[r] else np.zeros((heads, K, V))
        state = state * np.exp(case["decay"][r])[:, :, None]
        delta = case["beta"][r][:, None] * (case["v"][r] - np.einsum("hk,hkv->hv", k[r], state))
        state = state + k[r][:, :, None] * delta[:, None, :]
        out[r] = np.einsum("hk,hkv->hv", q[r], state)
        pool[slot] = state
    return out, pool


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _f64(x):
    return np.asarray(jnp.asarray(x).astype(jnp.float32), np.float64)


def _device(case, in_dtype, state_dtype):
    tokens = tuple(jnp.asarray(case[name], in_dtype) for name in ("q", "k", "v", "beta"))
    return (*tokens, jnp.asarray(case["decay"], jnp.float32)), jnp.asarray(case["pool"], state_dtype)


def _check(out, pool, case, slots, live, init, tol):
    ref_out, ref_pool = _reference(case, slots, live, init)
    out, pool = _f64(out), _f64(pool)
    advancing = live & (slots >= 0) & (slots < SLOTS)
    np.testing.assert_array_equal(out[~advancing], 0.0, err_msg="rows that do not advance must output zeros")
    err = _rel(out[advancing], ref_out[advancing])
    assert err < tol, f"output relative error {err:.2e}"
    owned = {int(s) for s in slots[advancing]}
    for slot in range(SLOTS):
        if slot in owned:
            err = _rel(pool[slot], ref_pool[slot])
            assert err < tol, f"slot {slot}: state relative error {err:.2e}"
        else:
            np.testing.assert_array_equal(pool[slot], case["pool"][slot], err_msg=f"untouched slot {slot}")


@pytest.mark.parametrize(
    ("in_dtype", "state_dtype", "reduce", "explicit", "tol"),
    [
        pytest.param(jnp.float32, jnp.float32, "mxu", False, 1e-4, id="f32-mxu-identity"),
        pytest.param(jnp.float32, jnp.float32, "vpu", True, 1e-4, id="f32-vpu-explicit-slots"),
        pytest.param(jnp.bfloat16, jnp.bfloat16, "mxu", True, 2e-2, id="bf16-mxu-explicit-slots"),
        pytest.param(jnp.bfloat16, jnp.bfloat16, "vpu", False, 2e-2, id="bf16-vpu-identity"),
    ],
)
def test_pallas_decode_matches_float64_and_xla_reference(in_dtype, state_dtype, reduce, explicit, tol):
    case = _case(0, in_dtype, state_dtype, heads=4)
    live, init = _flags()
    if explicit:
        slots = np.random.default_rng(1).permutation(SLOTS)[:ROWS].astype(np.int32)
        slots[IDLE[0]] = slots[IDLE[0] + 1]  # an idle row naming a live row's slot must not clobber it
        slots[7] = SLOTS + 2  # an out-of-range slot: the row does not advance
        slot_arg = jnp.asarray(slots)
    else:
        slots = np.arange(ROWS, dtype=np.int32)
        slot_arg = None
    tokens, pool = _device(case, in_dtype, state_dtype)
    flags = (jnp.asarray(live), jnp.asarray(init))
    out, new_pool = jax.jit(functools.partial(ragged_kda_decode_pallas, reduce=reduce))(*tokens, pool, slot_arg, *flags)
    assert out.dtype == jnp.float32
    assert new_pool.dtype == jnp.dtype(state_dtype)
    _check(out, new_pool, case, slots, live, init, tol)
    ref_out, ref_pool = jax.jit(ragged_kda_decode_xla)(*tokens, pool, slot_arg, *flags)
    assert _rel(_f64(out), _f64(ref_out)) < 1e-5
    assert _rel(_f64(new_pool), _f64(ref_pool)) < 1e-5


def test_head_sharded_operation_matches_float64_reference():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("needs 4 devices")
    mesh = Mesh(np.asarray(devices[:4]), ("tp",))
    case = _case(2, jnp.bfloat16, jnp.bfloat16, heads=8)  # 2 heads per device
    live, init = _flags()
    tokens, pool = _device(case, jnp.bfloat16, jnp.bfloat16)

    @jax.jit
    def step(tokens, pool, live, init):
        return ragged_kda_decode_head_sharded(*tokens, pool, None, live, init, mesh=mesh, head_axis="tp")

    out, new_pool = step(tokens, pool, jnp.asarray(live), jnp.asarray(init))
    _check(out, new_pool, case, np.arange(ROWS, dtype=np.int32), live, init, 2e-2)
