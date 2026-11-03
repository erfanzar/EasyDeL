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

"""Pallas TPU ``ragged_kda``: the multi-token chunk path in one kernel, recurrent-state pool updated in place.

Checked against a float64 per-token recurrence and against the XLA reference: ragged rows (single-token, empty,
two-token, exactly one chunk, several chunks, and a row ending at the buffer end, whose last window is clamped back),
rows continuing from a non-zero carried state and rows starting fresh, padding tokens, a stream shorter than one chunk,
a step without multi-token rows, explicit slots (permuted, an empty row naming a live row's slot, a multi-token row
with an out-of-range slot), fp32 and bf16 pools, the registry entry with options the kernel does not lower, and the
head-sharded module operation on four chips.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_kda import ragged_kda, ragged_kda_pallas
from ejkernel.kernels._xla.ragged_kda._xla_impl import ragged_kda_xla
from ejkernel.modules.operations import ragged_kda_head_sharded
from jax.sharding import Mesh

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="TPU-only Pallas execution")

K = V = 128
CHUNK = 64
LOWER = -5.0
SLOTS = 12
# Rows: two chunks, decode, empty, three chunks, two tokens, decode, exactly one chunk, and 45 tokens ending at the
# buffer end (without padding its window is clamped back, so its tokens start at lane 19 of the window).
LENS = (70, 1, 0, 130, 2, 1, 64, 45)
INIT = (True, True, False, True, False, True, False, True)  # False: the row starts from a zero state


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _rounded(x, dtype):
    """``x`` rounded to ``dtype`` by a device convert, as float64."""
    return np.asarray(jnp.asarray(np.asarray(x, np.float32)).astype(dtype).astype(jnp.float32), np.float64)


def _case(lens, heads, num_slots, seed, in_dtype, pool_dtype, pad=0):
    rng = np.random.default_rng(seed)
    tokens = int(sum(lens)) + pad
    return {
        "q": _rounded(rng.standard_normal((tokens, heads, K)), in_dtype),
        "k": _rounded(rng.standard_normal((tokens, heads, K)), in_dtype),
        "v": _rounded(rng.standard_normal((tokens, heads, V)), in_dtype),
        "beta": _rounded(_sigmoid(rng.standard_normal((tokens, heads))), in_dtype),
        "decay": _rounded(LOWER * _sigmoid(rng.standard_normal((tokens, heads, K))), jnp.float32),
        "pool": _rounded(0.1 * rng.standard_normal((num_slots, heads, K, V)), pool_dtype),
        "qsl": np.concatenate([[0], np.cumsum(lens)]).astype(np.int32),
    }


def _reference(case, lens, slots, init):
    """float64 per-token recurrence; returns ``(out [T, H, V], pool [S, H, K, V])``."""
    q, k = case["q"], case["k"]
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * K**-0.5
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    out = np.zeros(case["v"].shape)
    pool = case["pool"].copy()
    for r, n in enumerate(lens):
        slot = int(slots[r])
        if n == 0 or not 0 <= slot < pool.shape[0]:
            continue
        state = case["pool"][slot].copy() if init[r] else np.zeros(pool.shape[1:])
        for t in range(int(case["qsl"][r]), int(case["qsl"][r]) + n):
            state = state * np.exp(case["decay"][t])[:, :, None]
            delta = case["beta"][t][:, None] * (case["v"][t] - np.einsum("hk,hkv->hv", k[t], state))
            state = state + k[t][:, :, None] * delta[:, None, :]
            out[t] = np.einsum("hk,hkv->hv", q[t], state)
        pool[slot] = state
    return out, pool


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _f64(x):
    return np.asarray(jnp.asarray(x).astype(jnp.float32), np.float64)


def _run(fn, case, in_dtype, pool_dtype, slots, init, decay_lower_bound=LOWER):
    """``fn`` (a ``ragged_kda`` backend) on the case's device arrays, jitted."""
    tokens = tuple(jnp.asarray(case[name], in_dtype) for name in ("q", "k", "v", "beta"))
    decay = jnp.asarray(case["decay"], jnp.float32)
    pool = jnp.asarray(case["pool"], pool_dtype)

    @jax.jit
    def step(tokens, decay, pool, qsl, slots, init):
        return fn(*tokens, decay, pool, qsl, slots, init, chunk_size=CHUNK, decay_lower_bound=decay_lower_bound)

    slot_arg = None if slots is None else jnp.asarray(slots, jnp.int32)
    return step(tokens, decay, pool, jnp.asarray(case["qsl"]), slot_arg, jnp.asarray(np.asarray(init, bool)))


def _check(out, pool, case, lens, slots, init, tol):
    """float64 parity of live tokens and owned slots; every other token is zero and every other slot bit-equal."""
    ref_out, ref_pool = _reference(case, lens, slots, init)
    out, pool = _f64(out), _f64(pool)
    num_slots = case["pool"].shape[0]
    live = np.zeros(out.shape[0], bool)
    owned = set()
    for r, n in enumerate(lens):
        if n and 0 <= int(slots[r]) < num_slots:
            live[case["qsl"][r] : case["qsl"][r] + n] = True
            owned.add(int(slots[r]))
    np.testing.assert_array_equal(out[~live], 0.0, err_msg="padding tokens and tokens of empty rows must output zeros")
    err = _rel(out[live], ref_out[live])
    assert err < tol, f"output relative error {err:.2e}"
    for slot in range(num_slots):
        if slot in owned:
            err = _rel(pool[slot], ref_pool[slot])
            assert err < tol, f"slot {slot}: state relative error {err:.2e}"
        else:
            np.testing.assert_array_equal(pool[slot], case["pool"][slot], err_msg=f"untouched slot {slot}")


def _explicit_slots(num_rows):
    slots = np.random.default_rng(1).permutation(SLOTS)[:num_rows].astype(np.int32)
    slots[2] = slots[3]  # the empty row names the 130-token row's slot: it must not clobber it
    slots[6] = SLOTS + 3  # the 64-token row has an out-of-range slot: it is empty (zero output, no write)
    return slots


@pytest.mark.parametrize(
    ("in_dtype", "pool_dtype", "heads", "explicit", "pad", "tol"),
    [
        pytest.param(jnp.float32, jnp.float32, 4, False, 0, 1e-4, id="f32-identity-clamped-window"),
        pytest.param(jnp.float32, jnp.float32, 4, False, 27, 1e-4, id="f32-identity-padding"),
        pytest.param(jnp.float32, jnp.float32, 8, True, 0, 1e-4, id="f32-explicit-slots-8-heads"),
        pytest.param(jnp.bfloat16, jnp.bfloat16, 4, False, 0, 2e-2, id="bf16-bf16-pool"),
        pytest.param(jnp.bfloat16, jnp.float32, 4, True, 5, 2e-2, id="bf16-f32-pool-explicit-slots"),
    ],
)
def test_pallas_chunk_path_matches_float64_and_xla_reference(in_dtype, pool_dtype, heads, explicit, pad, tol):
    case = _case(LENS, heads, SLOTS, seed=0, in_dtype=in_dtype, pool_dtype=pool_dtype, pad=pad)
    slots = _explicit_slots(len(LENS)) if explicit else np.arange(len(LENS), dtype=np.int32)
    slot_arg = slots if explicit else None
    out, pool = _run(ragged_kda_pallas, case, in_dtype, pool_dtype, slot_arg, INIT)
    assert out.dtype == jnp.float32
    assert pool.dtype == jnp.dtype(pool_dtype)
    _check(out, pool, case, LENS, slots, INIT, tol)
    ref_out, ref_pool = _run(ragged_kda_xla, case, in_dtype, pool_dtype, slot_arg, INIT)
    assert _rel(_f64(out), _f64(ref_out)) < 1e-5
    # A bf16 pool rounds both final states, so a summation-order difference can flip single bf16 ulps.
    assert _rel(_f64(pool), _f64(ref_pool)) < (1e-5 if pool_dtype == jnp.float32 else 2e-4)


@pytest.mark.parametrize(
    ("lens", "init", "pad"),
    [
        pytest.param((3, 1, 5), (True, True, False), 2, id="stream-shorter-than-a-chunk"),
        pytest.param((1,) * 8, (True,) * 8, 0, id="decode-rows-only"),
    ],
)
def test_short_streams_and_steps_without_multi_token_rows(lens, init, pad):
    # Two pool slots more than rows: slots that no row owns must stay bit-equal.
    case = _case(lens, 4, len(lens) + 2, seed=4, in_dtype=jnp.float32, pool_dtype=jnp.float32, pad=pad)
    out, pool = _run(ragged_kda_pallas, case, jnp.float32, jnp.float32, None, init)
    _check(out, pool, case, lens, np.arange(len(lens), dtype=np.int32), init, 1e-4)


def test_registry_entry_without_a_decay_bound_matches_float64_reference():
    # Without a decay bound the pairwise decay stays dense, which the kernel does not lower: the TPU registry entry
    # then runs the XLA implementation under the same contract.
    case = _case(LENS, 4, SLOTS, seed=9, in_dtype=jnp.float32, pool_dtype=jnp.float32)
    out, pool = _run(ragged_kda, case, jnp.float32, jnp.float32, None, INIT, decay_lower_bound=None)
    _check(out, pool, case, LENS, np.arange(len(LENS), dtype=np.int32), INIT, 1e-4)


def test_head_sharded_operation_matches_float64_reference():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("needs 4 devices")
    mesh = Mesh(np.asarray(devices[:4]), ("tp",))
    case = _case(LENS, 8, SLOTS, seed=5, in_dtype=jnp.bfloat16, pool_dtype=jnp.bfloat16)  # 2 heads per device
    tokens = tuple(jnp.asarray(case[name], jnp.bfloat16) for name in ("q", "k", "v", "beta"))

    @jax.jit
    def step(tokens, decay, pool, qsl, init):
        return ragged_kda_head_sharded(
            *tokens,
            decay,
            pool,
            qsl,
            None,
            init,
            mesh=mesh,
            head_axis="tp",
            chunk_size=CHUNK,
            decay_lower_bound=LOWER,
        )

    out, pool = step(
        tokens,
        jnp.asarray(case["decay"], jnp.float32),
        jnp.asarray(case["pool"], jnp.bfloat16),
        jnp.asarray(case["qsl"]),
        jnp.asarray(np.asarray(INIT, bool)),
    )
    _check(out, pool, case, LENS, np.arange(len(LENS), dtype=np.int32), INIT, 2e-2)
