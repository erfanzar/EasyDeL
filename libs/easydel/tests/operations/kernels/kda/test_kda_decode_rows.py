# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
"""Decode-only eSurge KDA steps through :func:`kda_rows_forward` (window-shift conv + ``ragged_kda_decode``).

The head sizes fill the 128 TPU lanes, so on TPU the recurrence runs in the in-place Pallas kernel. Decode steps are
checked against a float64 per-request causal conv (SiLU) and per-channel KDA step: fresh rows start from zero, rows
without a token (or with an out-of-range slot) leave every slot untouched, explicit slots are followed. Steps with
multi-token rows must leave the router with the flat driver's results.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.caching import KDACacheView
from easydel.caching.kda.cache import KDACacheConfig
from easydel.operations import OperationMetadata
from easydel.operations.kernels import kda_ragged_rows_forward, kda_rows_forward
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax import PartitionAxis

H, HK, HV, DC = 8, 128, 128, 4
KEY_DIM, VALUE_DIM = H * HK, H * HV
DIMS = (KEY_DIM, KEY_DIM, VALUE_DIM)
LEAVES = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
AXES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
LOWER_BOUND = -5.0
POOL = 10  # pool slots (more than the rows)

# (lens, context_lens, bucket). DECODE: continuations, a fresh one-token prompt (row 1), two rows without a token.
DECODE = ((1, 1, 0, 1, 1, 0, 1, 1), (9, 1, 0, 40, 2, 5, 3, 17), 8)
# Explicit slots for DECODE: the empty row 2 names row 3's slot, row 6 has an out-of-range slot (it must not advance).
DECODE_SLOTS = (4, 7, 1, 1, 9, 0, POOL + 3, 2)
# MIXED: a decode row, a fresh 70-token prompt (two chunks), an empty row, a 3-token continuation, a decode row.
MIXED = ((1, 70, 0, 3, 1), (9, 70, 0, 12, 5), 80)
# SHORT_MIXED: a bucket of exactly one token per row with a fresh 3-token prompt (row 1): the router traces its
# conditional and must take the flat driver's branch.
SHORT_MIXED = ((1, 3, 0, 1, 1, 0, 1, 1), (9, 3, 0, 40, 2, 5, 3, 17), 8)


def _starts(lens):
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def _rounded(x, dtype):
    """``x`` rounded to ``dtype`` by a device convert, as float64."""
    return np.asarray(jnp.asarray(np.asarray(x, np.float32)).astype(dtype).astype(jnp.float32), np.float64)


def _case(layout, seed, dtype):
    """Random inputs and pools (``POOL`` slots) rounded to ``dtype`` (decay to float32), stored as float64."""
    bucket = layout[2]
    rng = np.random.default_rng(seed)
    return {
        "q": _rounded(rng.normal(size=(bucket, KEY_DIM)), dtype),
        "k": _rounded(rng.normal(size=(bucket, KEY_DIM)), dtype),
        "v": _rounded(rng.normal(size=(bucket, VALUE_DIM)), dtype),
        "beta": _rounded(1.0 / (1.0 + np.exp(-rng.normal(size=(bucket, H)))), dtype),
        "decay": _rounded(LOWER_BOUND / (1.0 + np.exp(-rng.normal(size=(bucket, H, HK)))), jnp.float32),
        "kernels": [_rounded(0.5 * rng.normal(size=(d, DC)), dtype) for d in DIMS],
        "conv": [_rounded(rng.normal(size=(POOL, d, DC)), dtype) for d in DIMS],
        "rec": _rounded(0.1 * rng.normal(size=(POOL, H, HK, HV)), dtype),
    }


def _view(leaves):
    cfg = KDACacheConfig.create(
        num_hidden_layers=1,
        partition_axis=PartitionAxis(),
        batch_size=POOL,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        d_conv=DC,
        recurrent_state_shape=(H, HK, HV),
    )
    return KDACacheView(**leaves, positions=jnp.zeros((POOL,), jnp.int32), metadata=cfg, layer_index=0)


def _run(driver, case, layout, dtype, *, tp, slots=None):
    """One jitted step of ``driver`` with the pools in the eSurge KDA layout (conv windows by feature, states by head)."""
    if len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} devices")
    lens, context, _ = layout
    mesh = Mesh(np.asarray(jax.devices()[:tp]).reshape(1, 1, 1, 1, tp, 1), AXES)
    metadata = OperationMetadata(runtime_dtype=dtype, _stored_mesh=mesh)
    inputs = {
        "query": jnp.asarray(case["q"], dtype)[None],
        "key": jnp.asarray(case["k"], dtype)[None],
        "value": jnp.asarray(case["v"], dtype)[None],
        "beta": jnp.asarray(case["beta"], dtype)[None],
        "decay": jnp.asarray(case["decay"], jnp.float32)[None],
        "conv_kernels": tuple(jnp.asarray(kernel, dtype) for kernel in case["kernels"]),
        "query_start_loc": jnp.asarray(_starts(lens), jnp.int32),
        "context_lens": jnp.asarray(context, jnp.int32),
    }
    if slots is not None:
        inputs["state_indices"] = jnp.asarray(slots, jnp.int32)
    conv = NamedSharding(mesh, PartitionSpec(None, "tp", None))
    rec = NamedSharding(mesh, PartitionSpec(None, "tp", None, None))
    leaves = {
        name: jax.device_put(jnp.asarray(window, dtype), conv)
        for name, window in zip(LEAVES[:3], case["conv"], strict=True)
    }
    leaves["recurrent_state"] = jax.device_put(jnp.asarray(case["rec"], dtype), rec)

    @jax.jit
    def step(inputs, leaves):
        out, view = driver(
            **inputs,
            cache_view=_view(leaves),
            metadata=metadata,
            conv_dtype=dtype,
            chunk_size=64,
            decay_lower_bound=LOWER_BOUND,
            precision="highest",
        )
        return out, {name: getattr(view, name) for name in LEAVES}

    return step(inputs, leaves)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _reference_decode(case, layout, slots):
    """float64 decode step: ``(out [T, H, HV], conv pools, recurrent pool)``; row r owns slot ``slots[r]``."""
    lens, context, bucket = layout
    starts = _starts(lens)
    out = np.zeros((bucket, H, HV))
    conv = [c.copy() for c in case["conv"]]
    rec = case["rec"].copy()
    for r, n in enumerate(lens):
        slot = slots[r]
        if n == 0 or not 0 <= slot < POOL:
            continue
        assert n == 1, "decode steps carry at most one token per row"
        t = starts[r]
        fresh = context[r] <= n
        mixed = []
        for i, name in enumerate(("q", "k", "v")):
            window = np.zeros_like(conv[i][slot]) if fresh else conv[i][slot]  # [d, DC], last column most recent
            window = np.concatenate([window[:, 1:], case[name][t][:, None]], axis=1)
            mixed.append(_silu(np.sum(window * case["kernels"][i], axis=1)))
            conv[i][slot] = window
        q, k, v = mixed[0].reshape(H, HK), mixed[1].reshape(H, HK), mixed[2].reshape(H, HV)
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * HK**-0.5
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        state = np.zeros((H, HK, HV)) if fresh else rec[slot]
        state = state * np.exp(case["decay"][t])[:, :, None]
        delta = case["beta"][t][:, None] * (v - np.einsum("hk,hkv->hv", k, state))
        state = state + k[:, :, None] * delta[:, None, :]
        out[t] = np.einsum("hk,hkv->hv", q, state)
        rec[slot] = state
    return out, conv, rec


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _f64(x):
    return np.asarray(jnp.asarray(x).astype(jnp.float32), np.float64)


@pytest.mark.parametrize(
    ("dtype", "tp", "slots", "tol"),
    [
        pytest.param(jnp.float32, 1, None, 1e-4, id="f32-tp1"),
        pytest.param(jnp.float32, 4, None, 1e-4, id="f32-tp4"),
        pytest.param(jnp.float32, 4, DECODE_SLOTS, 1e-4, id="f32-tp4-explicit-slots"),
        pytest.param(jnp.bfloat16, 4, None, 3e-2, id="bf16-tp4"),
    ],
)
def test_decode_step_matches_float64_reference(dtype, tp, slots, tol):
    lens, _, bucket = DECODE
    case = _case(DECODE, seed=0, dtype=dtype)
    out, leaves = _run(kda_rows_forward, case, DECODE, dtype, tp=tp, slots=slots)
    slots = list(range(len(lens))) if slots is None else list(slots)
    ref_out, ref_conv, ref_rec = _reference_decode(case, DECODE, slots)
    starts = _starts(lens)
    out = _f64(out)
    assert out.shape == (1, bucket, H, HV)
    out = out[0]
    advancing = [r for r, n in enumerate(lens) if n and 0 <= slots[r] < POOL]
    for r in advancing:
        err = _rel(out[starts[r]], ref_out[starts[r]])
        assert err < tol, f"row {r}: output relative error {err:.2e}"
    silent = [int(starts[r]) for r, n in enumerate(lens) if n and r not in advancing]
    np.testing.assert_array_equal(out[silent + list(range(int(starts[-1]), bucket))], 0.0, err_msg="silent tokens")
    for i, name in enumerate(LEAVES[:3]):
        assert leaves[name].dtype == jnp.dtype(dtype), f"{name} dtype {leaves[name].dtype}"
        np.testing.assert_array_equal(_f64(leaves[name]), ref_conv[i], err_msg=name)
    rec = _f64(leaves["recurrent_state"])
    owned = {slots[r] for r in advancing}
    for slot in range(POOL):
        if slot in owned:
            err = _rel(rec[slot], ref_rec[slot])
            assert err < tol, f"slot {slot}: recurrent relative error {err:.2e}"
        else:
            np.testing.assert_array_equal(rec[slot], case["rec"][slot], err_msg=f"untouched slot {slot}")


@pytest.mark.parametrize(
    "layout",
    [pytest.param(MIXED, id="bucket-longer-than-rows"), pytest.param(SHORT_MIXED, id="bucket-of-one-token-per-row")],
)
def test_steps_with_multi_token_rows_keep_the_flat_driver_results(layout):
    case = _case(layout, seed=1, dtype=jnp.float32)
    routed_out, routed = _run(kda_rows_forward, case, layout, jnp.float32, tp=1)
    flat_out, flat = _run(kda_ragged_rows_forward, case, layout, jnp.float32, tp=1)
    # Same driver inside the conditional's false branch; only XLA's fusion of that branch may differ (f32 rounding).
    np.testing.assert_allclose(_f64(routed_out), _f64(flat_out), rtol=1e-5, atol=1e-5)
    for name in LEAVES:
        np.testing.assert_allclose(_f64(routed[name]), _f64(flat[name]), rtol=1e-5, atol=1e-5, err_msg=name)
