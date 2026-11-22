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


"""Pallas TPU ``ragged_kda_conv_decode``: the fused KDA decode step (conv windows, conv, SiLU, L2 norm, recurrence).

The kernel must reproduce the serving path bit for bit: EasyDeL's decode conv glue (XLA: window shift, fp32 taps,
SiLU, rounded to the conv dtype) followed by the ``ragged_kda_decode`` Pallas kernel, with the four pools updated the
same way. Checked on small and production-like shapes (one device's share of GLM-5.3-Flash at ep4 x tp4: 32 tokens, 4
heads, 128-wide heads, d_conv 4, bf16) with fresh rows, idle tokens, out-of-range slots, the identity and the explicit
slot layouts, several grid steps and several chained steps; against the XLA reference (tolerance: its recurrence
sums in another order); and under the head-sharded module operation on four chips.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_kda_conv_decode._pallas_impl import plan_blocks, ragged_kda_conv_decode_pallas
from ejkernel.kernels._pallas.tpu.ragged_kda_decode._pallas_impl import ragged_kda_decode_pallas
from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.ragged_kda_conv_decode._xla_impl import ragged_kda_conv_decode_xla
from ejkernel.modules.operations import ragged_kda_conv_decode_head_sharded
from jax import numpy as jnp
from jax.sharding import Mesh

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="TPU-only Pallas execution")

BF16, F32 = jnp.bfloat16, jnp.float32
K = V = 128
DC = 4
LOWER = -5.0

# name: (tokens, heads, slots, dtype, explicit slots, idle tokens, fresh tokens, out-of-range tokens, block_tokens)
CASES = {
    "small-explicit": (16, 2, 24, BF16, True, (3, 12, 13), (0, 5), (9,), 0),
    "small-f32": (16, 2, 24, F32, True, (3, 12), (0, 5), (9,), 0),
    "prod-identity": (32, 4, 32, BF16, False, (), (), (), 0),
    "prod-identity-mixed": (32, 4, 40, BF16, False, tuple(range(24, 32)), (1, 5, 9, 13), (), 0),
    "prod-explicit": (32, 4, 48, BF16, True, tuple(range(24, 32)), (1, 5, 9, 13), (7, 11), 0),
    "multi-step-grid": (64, 4, 64, BF16, True, (2, 40), (3, 33), (17,), 16),
}


def _case(seed, tokens, heads, slots, dtype, explicit, idle, fresh, out_of_range):
    rng = np.random.default_rng(seed)
    kd, vd = heads * K, heads * V
    token_slots = rng.permutation(slots)[:tokens].astype(np.int32) if explicit else np.arange(tokens, dtype=np.int32)
    token_slots[list(idle)] = -1
    for i, t in enumerate(out_of_range):
        token_slots[t] = slots + 3 if i % 2 == 0 else -2
    token_init = np.ones((tokens,), bool)
    token_init[list(fresh)] = False
    return {
        "query": jnp.asarray(rng.normal(size=(tokens, kd)), dtype),
        "key": jnp.asarray(rng.normal(size=(tokens, kd)), dtype),
        "value": jnp.asarray(rng.normal(size=(tokens, vd)), dtype),
        "beta": jnp.asarray(1.0 / (1.0 + np.exp(-rng.normal(size=(tokens, heads)))), dtype),
        "decay": jnp.asarray(LOWER / (1.0 + np.exp(-rng.normal(size=(tokens, heads, K)))), F32),
        "conv_weights": tuple(jnp.asarray(0.5 * rng.normal(size=(d, DC)), dtype) for d in (kd, kd, vd)),
        "conv_pools": tuple(jnp.asarray(rng.normal(size=(slots, d, DC)), dtype) for d in (kd, kd, vd)),
        "recurrent_state": jnp.asarray(0.1 * rng.normal(size=(slots, heads, K, V)), dtype),
        "token_slots": jnp.asarray(token_slots),
        "token_init": jnp.asarray(token_init),
    }


def _args(c):
    return tuple(
        c[n]
        for n in (
            "query",
            "key",
            "value",
            "beta",
            "decay",
            "conv_weights",
            "conv_pools",
            "recurrent_state",
            "token_slots",
            "token_init",
        )
    )


def _serving_glue(query, key, value, beta, decay, conv_weights, conv_pools, recurrent_state, token_slots, token_init):
    """EasyDeL's eSurge decode glue (explicit-slot form) + the ``ragged_kda_decode`` Pallas kernel: the serving path."""
    tokens, heads = beta.shape
    num_slots = recurrent_state.shape[0]
    conv_dtype = query.dtype
    slots = token_slots.astype(jnp.int32)
    live = (slots >= 0) & (slots < num_slots)
    fresh = jnp.logical_not(token_init)
    read = jnp.clip(slots, 0, num_slots - 1)
    write = jnp.where(live, slots, num_slots)

    def conv(stream, kernel, pool):
        compute = jnp.promote_types(stream.dtype, pool.dtype)
        window = jnp.where(fresh[:, None, None], jnp.zeros((), compute), pool[read].astype(compute))
        new_window = jnp.concatenate([window[:, :, 1:], stream.astype(compute)[:, :, None]], axis=-1)
        taps = new_window.astype(F32)
        weights = kernel.astype(F32)
        mixed = taps[:, :, 0] * weights[None, :, 0]
        for j in range(1, int(kernel.shape[-1])):
            mixed = mixed + taps[:, :, j] * weights[None, :, j]
        out = jax.nn.silu(mixed).astype(compute).astype(conv_dtype)
        return out, pool.at[write].set(new_window.astype(pool.dtype), mode="drop")

    q, q_pool = conv(query, conv_weights[0], conv_pools[0])
    k, k_pool = conv(key, conv_weights[1], conv_pools[1])
    v, v_pool = conv(value, conv_weights[2], conv_pools[2])
    out, rec = ragged_kda_decode_pallas(
        q.reshape(tokens, heads, -1),
        k.reshape(tokens, heads, -1),
        v.reshape(tokens, heads, -1),
        beta,
        decay,
        recurrent_state,
        slots,
        live,
        token_init,
    )
    return out, (q_pool, k_pool, v_pool, rec)


def _bits(x):
    x = np.asarray(x)
    return x.view({2: np.uint16, 4: np.uint32}[x.dtype.itemsize])


def _assert_bitwise(got, want):
    np.testing.assert_array_equal(_bits(got[0]), _bits(want[0]), err_msg="output")
    for name, a, b in zip(("q_conv", "k_conv", "v_conv", "recurrent"), got[1], want[1], strict=True):
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(_bits(a), _bits(b), err_msg=f"{name} pool")


@pytest.mark.parametrize("name", list(CASES))
def test_fused_step_reproduces_the_serving_glue(name):
    tokens, heads, slots, dtype, explicit, idle, fresh, oor, block_tokens = CASES[name]
    case = _case(0, tokens, heads, slots, dtype, explicit, idle, fresh, oor)
    fused = jax.jit(lambda *a: ragged_kda_conv_decode_pallas(*a, conv_dtype=dtype, block_tokens=block_tokens))
    glue = jax.jit(_serving_glue)
    pools_in = (*case["conv_pools"], case["recurrent_state"])
    got = fused(*_args(case))
    want = glue(*_args(case))
    if dtype == BF16:
        _assert_bitwise(got, want)
    else:
        # fp32 conv outputs are not rounded: the windows still match bit for bit, the L2-norm sums may round apart.
        for a, b in zip(got[1][:3], want[1][:3], strict=True):
            np.testing.assert_array_equal(_bits(a), _bits(b))
        np.testing.assert_allclose(np.asarray(got[0]), np.asarray(want[0]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(got[1][3]), np.asarray(want[1][3]), rtol=1e-5, atol=1e-6)
    # Silent tokens output zeros; slots no advancing token names keep every bit.
    ts = np.asarray(case["token_slots"])
    silent = (ts < 0) | (ts >= slots)
    np.testing.assert_array_equal(np.asarray(got[0])[silent], 0.0, err_msg="silent tokens")
    untouched = np.setdiff1d(np.arange(slots), ts[~silent])
    for name_, new, old in zip(("q_conv", "k_conv", "v_conv", "recurrent"), got[1], pools_in, strict=True):
        np.testing.assert_array_equal(_bits(new)[untouched], _bits(old)[untouched], err_msg=f"untouched {name_}")


def test_chained_steps_stay_bitwise():
    """Three steps feeding each step's pools to the next (state carried by the kernel's own writes)."""
    tokens, heads, slots = 32, 4, 32
    fused = jax.jit(lambda *a: ragged_kda_conv_decode_pallas(*a, conv_dtype=BF16))
    glue = jax.jit(_serving_glue)
    case = _case(1, tokens, heads, slots, BF16, False, (), tuple(range(tokens)), ())
    got_pools = want_pools = (case["conv_pools"], case["recurrent_state"])
    for step in range(3):
        c = _case(10 + step, tokens, heads, slots, BF16, False, (), () if step else tuple(range(tokens)), ())
        args = (*_args(c)[:5], case["conv_weights"])
        tail = (c["token_slots"], c["token_init"])
        got = fused(*args, got_pools[0], got_pools[1], *tail)
        want = glue(*args, want_pools[0], want_pools[1], *tail)
        _assert_bitwise(got, want)
        got_pools = (got[1][:3], got[1][3])
        want_pools = (want[1][:3], want[1][3])


def test_fused_step_matches_the_xla_reference():
    """The XLA fallback: conv windows bit for bit, outputs and states up to the recurrence's summation order."""
    tokens, heads, slots, dtype, explicit, idle, fresh, oor, _ = CASES["prod-explicit"]
    case = _case(2, tokens, heads, slots, dtype, explicit, idle, fresh, oor)
    got = jax.jit(lambda *a: ragged_kda_conv_decode_pallas(*a, conv_dtype=dtype))(*_args(case))
    want = jax.jit(lambda *a: ragged_kda_conv_decode_xla(*a, conv_dtype=dtype))(*_args(case))
    for a, b in zip(got[1][:3], want[1][:3], strict=True):
        np.testing.assert_array_equal(_bits(a), _bits(b))
    np.testing.assert_allclose(np.asarray(got[0]), np.asarray(want[0]), rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(
        np.asarray(got[1][3].astype(F32)), np.asarray(want[1][3].astype(F32)), rtol=2e-2, atol=4e-3
    )


def test_head_sharded_operation_is_bitwise_the_single_device_kernel():
    """16 heads over four chips (4 per chip, as in serving): every chip's heads equal the one-chip kernel's."""
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("needs 4 TPU devices")
    mesh = Mesh(np.asarray(devices[:4]).reshape(4), ("tp",))
    case = _case(3, 32, 16, 40, BF16, True, (4, 30), (0, 9), (6,))
    got = jax.jit(lambda *a: ragged_kda_conv_decode_head_sharded(*a, mesh=mesh, head_axis="tp", conv_dtype=BF16))(
        *_args(case)
    )
    want = jax.jit(lambda *a: ragged_kda_conv_decode_pallas(*a, conv_dtype=BF16))(*_args(case))
    _assert_bitwise(got, want)


def test_registry_signatures_match():
    assert kernel_registry.validate_signatures("ragged_kda_conv_decode") is True


@pytest.mark.cpu_ok
def test_block_plan_follows_the_vmem_budget():
    # Two cores take half the tokens each; blocks are multiples of 16 (or the whole stream) and fit the budget.
    assert plan_blocks(32, 1000, 0, budget_bytes=10**9, num_cores=2) == (2, 16)
    assert plan_blocks(8, 1000, 0, budget_bytes=10**9, num_cores=2) == (1, 8)
    assert plan_blocks(64, 1000, 0, budget_bytes=16_000, num_cores=2) == (2, 16)
    assert plan_blocks(48, 1000, 0, budget_bytes=10**9, num_cores=2, requested=16) == (1, 16)
    assert plan_blocks(32, 10**6, 0, budget_bytes=10**6, num_cores=2) is None
