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

"""Flat-stream KDA serving driver (``kda_ragged_rows_forward``) and its ``"ragged_kda"`` operation.

Row ``r`` of a packed token stream owns a contiguous token span and pool slot ``r`` (the flat driver takes explicit
slots instead when eSurge's row window does not start at slot 0). Both serving drivers must equal an
independent float64 reference that runs the causal depthwise conv (SiLU) and the per-channel KDA recurrence token by
token per request: continuation rows (``context_lens > lens``) start from their carried conv window and recurrent
state, rows that start a request start from zero, rows without tokens and pool slots past the rows keep their state bit
for bit, and bucket padding (garbage inputs, as in eSurge) gets zero output. The row-padded driver uses the recurrent
slot of a row that starts a request as is (eSurge zeroes freed slots), so its cases zero those slots. The conv-window
pools may have another dtype than the stream (an fp8 KV-cache dtype, for example); new windows keep the pool's dtype.
"""

from pathlib import Path

import easydel.operations.kernels.ragged_kda as adapter_module
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.caching import KDACacheView
from easydel.caching.kda.cache import KDACacheConfig
from easydel.infra.sharding import mesh_axis_size
from easydel.operations import OperationMetadata
from easydel.operations.kernels import RaggedKDAOp, kda_packed_rows_forward, kda_ragged_rows_forward
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax import PartitionAxis

H, HK, HV, DC = 4, 16, 16, 4
KEY_DIM, VALUE_DIM = H * HK, H * HV
DIMS = (KEY_DIM, KEY_DIM, VALUE_DIM)
LEAVES = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
AXES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
LOWER_BOUND = -5.0
EXTRA_SLOTS = 2  # pool slots past the rows, filled with values that must survive untouched

# (lens, context_lens, bucket). MIXED: a decode continuation, a fresh prompt over several chunks with a partial tail,
# a multi-token continuation (chunked prefill), an empty row and a fresh prompt shorter than the conv window.
# DECODE: one token per row (one of them a fresh one-token prompt) plus an empty row.
MIXED = ((1, 150, 5, 0, 3), (9, 150, 12, 0, 3), 192)
DECODE = ((1, 1, 1, 1, 0, 1), (5, 1, 33, 2, 0, 7), 8)
# SHORT: multi-token continuations shorter than the conv window (their new windows keep part of the carried one), a
# decode continuation and a fresh two-token prompt.
SHORT = ((2, 3, 1, 2), (9, 4, 7, 2), 16)
FP8 = jnp.float8_e4m3fn


def _starts(lens):
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def _fresh_rows(lens, context):
    return [r for r, (n, c) in enumerate(zip(lens, context, strict=True)) if 0 < n and c <= n]


def _case(layout, seed, dtype, *, zero_fresh_recurrent, pool_dtype=None, pool_slots=None):
    """Random inputs and pools for one layout, rounded to the dtypes the driver sees and stored as float64.

    The conv-window pools are rounded to ``pool_dtype`` (default ``dtype``), the recurrent pool to float32; the pools
    hold ``pool_slots`` slots (default: one per row plus ``EXTRA_SLOTS``).
    """
    lens, context, bucket = layout
    pool = len(lens) + EXTRA_SLOTS if pool_slots is None else int(pool_slots)
    rng = np.random.default_rng(seed)
    pool_dtype = dtype if pool_dtype is None else pool_dtype

    def rounded(x, to=dtype):
        return np.asarray(jnp.asarray(x, to), np.float64)

    case = {
        "q": rounded(rng.normal(size=(bucket, KEY_DIM))),
        "k": rounded(rng.normal(size=(bucket, KEY_DIM))),
        "v": rounded(rng.normal(size=(bucket, VALUE_DIM))),
        "beta": rounded(1.0 / (1.0 + np.exp(-rng.normal(size=(bucket, H))))),
        "decay": rounded(LOWER_BOUND / (1.0 + np.exp(-rng.normal(size=(bucket, H, HK)))), jnp.float32),
        "kernels": [rounded(0.5 * rng.normal(size=(d, DC))) for d in DIMS],
        "conv": [rounded(rng.normal(size=(pool, d, DC)), pool_dtype) for d in DIMS],
        "rec": rounded(0.1 * rng.normal(size=(pool, H, HK, HV)), jnp.float32),
    }
    if zero_fresh_recurrent:
        case["rec"][_fresh_rows(lens, context)] = 0.0
    return case


def _driver_inputs(case, layout, dtype):
    lens, context, _ = layout
    return {
        "query": jnp.asarray(case["q"], dtype)[None],
        "key": jnp.asarray(case["k"], dtype)[None],
        "value": jnp.asarray(case["v"], dtype)[None],
        "beta": jnp.asarray(case["beta"], dtype)[None],
        "decay": jnp.asarray(case["decay"], jnp.float32)[None],
        "conv_kernels": tuple(jnp.asarray(kernel, dtype) for kernel in case["kernels"]),
        "query_start_loc": jnp.asarray(_starts(lens), jnp.int32),
        "context_lens": jnp.asarray(context, jnp.int32),
    }


def _leaves(case, dtype, pool_dtype=None):
    pool_dtype = dtype if pool_dtype is None else pool_dtype
    leaves = {name: jnp.asarray(conv, pool_dtype) for name, conv in zip(LEAVES[:3], case["conv"], strict=True)}
    leaves["recurrent_state"] = jnp.asarray(case["rec"], jnp.float32)
    return leaves


def _view(leaves):
    """KDA cache view over the given pool leaves (built inside the traced step)."""
    pool = int(leaves["recurrent_state"].shape[0])
    cfg = KDACacheConfig.create(
        num_hidden_layers=1,
        partition_axis=PartitionAxis(),
        batch_size=pool,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        d_conv=DC,
        recurrent_state_shape=(H, HK, HV),
    )
    return KDACacheView(**leaves, positions=jnp.zeros((pool,), jnp.int32), metadata=cfg, layer_index=0)


def _mesh(tp):
    if len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} devices")
    return Mesh(np.asarray(jax.devices()[:tp]).reshape(1, 1, 1, 1, tp, 1), AXES)


def _serving_layout(leaves, mesh):
    """Place the pool leaves like the eSurge KDA cache: conv windows split by feature, recurrent states by head (tp)."""
    conv = NamedSharding(mesh, PartitionSpec(None, "tp", None))
    rec = NamedSharding(mesh, PartitionSpec(None, "tp", None, None))
    return {name: jax.device_put(value, rec if name == "recurrent_state" else conv) for name, value in leaves.items()}


def _run_ragged(
    case,
    layout,
    dtype,
    *,
    tp,
    chunk_size,
    decay_lower_bound,
    serving_layout=False,
    pool_dtype=None,
    conv_dtype=None,
    state_indices=None,
):
    mesh = _mesh(tp)
    metadata = OperationMetadata(runtime_dtype=jnp.float32, _stored_mesh=mesh)

    @jax.jit
    def step(inputs, leaves):
        out, view = kda_ragged_rows_forward(
            **inputs,
            cache_view=_view(leaves),
            metadata=metadata,
            conv_dtype=dtype if conv_dtype is None else conv_dtype,
            chunk_size=chunk_size,
            decay_lower_bound=decay_lower_bound,
            precision="highest",
        )
        return out, {name: getattr(view, name) for name in LEAVES}

    inputs = _driver_inputs(case, layout, dtype)
    if state_indices is not None:
        inputs["state_indices"] = jnp.asarray(state_indices, jnp.int32)
    leaves = _leaves(case, dtype, pool_dtype)
    if serving_layout:
        leaves = _serving_layout(leaves, mesh)
    return step(inputs, leaves)


def _run_packed(case, layout, dtype, *, chunk_size):
    @jax.jit
    def step(inputs, leaves):
        out, view = kda_packed_rows_forward(**inputs, cache_view=_view(leaves), conv_dtype=dtype, chunk_size=chunk_size)
        return out, {name: getattr(view, name) for name in LEAVES}

    return step(_driver_inputs(case, layout, dtype), _leaves(case, dtype))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _reference(case, lens, context):
    """float64 per-request causal depthwise conv (SiLU) and per-channel KDA, token by token.

    Returns the packed output ``[T, H, HV]`` and the expected conv and recurrent pools.
    """
    starts = _starts(lens)
    out = np.zeros((case["q"].shape[0], H, HV))
    conv = [c.copy() for c in case["conv"]]
    rec = case["rec"].copy()
    for r, n in enumerate(lens):
        if n == 0:
            continue
        carried = context[r] > n
        span = slice(starts[r], starts[r] + n)
        mixed = []
        for i, name in enumerate(("q", "k", "v")):
            history = conv[i][r] if carried else np.zeros_like(conv[i][r])  # [d, DC], last column most recent
            full = np.concatenate([history.T, case[name][span]], axis=0)  # [DC + n, d]
            kernel = case["kernels"][i].T  # [DC, d], last row is the current-token tap
            mixed.append(_silu(np.stack([np.sum(full[t + 1 : t + 1 + DC] * kernel, axis=0) for t in range(n)])))
            conv[i][r] = full[-DC:].T
        q = mixed[0].reshape(n, H, HK)
        k = mixed[1].reshape(n, H, HK)
        v = mixed[2].reshape(n, H, HV)
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * HK**-0.5
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        state = rec[r] if carried else np.zeros((H, HK, HV))
        beta, decay = case["beta"][span], case["decay"][span]
        for t in range(n):
            state = state * np.exp(decay[t])[:, :, None]
            delta = beta[t][:, None] * (v[t] - np.einsum("hk,hkv->hv", k[t], state))
            state = state + k[t][:, :, None] * delta[:, None, :]
            out[starts[r] + t] = np.einsum("hk,hkv->hv", q[t], state)
        rec[r] = state
    return out, conv, rec


def _reference_slots(case, lens, context, slots):
    """:func:`_reference` with row r reading and writing pool slot ``slots[r]``; rows without tokens write nothing.

    Returns the packed output and the expected full conv and recurrent pools (slots no row with tokens owns unchanged).
    """
    rows_case = dict(case, conv=[c[slots] for c in case["conv"]], rec=case["rec"][slots])
    out, conv_rows, rec_rows = _reference(rows_case, lens, context)
    conv = [c.copy() for c in case["conv"]]
    rec = case["rec"].copy()
    for r, n in enumerate(lens):
        if n:
            for i in range(len(conv)):
                conv[i][slots[r]] = conv_rows[i][r]
            rec[slots[r]] = rec_rows[r]
    return out, conv, rec


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _check(result, case, layout, *, tol, pool_dtype=None, slots=None):
    """Assert a driver result against :func:`_reference_slots` (row r owns slot ``slots[r]``, default ``r``)."""
    lens, context, bucket = layout
    slots = list(range(len(lens))) if slots is None else [int(s) for s in slots]
    out, leaves = result
    ref_out, ref_conv, ref_rec = _reference_slots(case, lens, context, slots)
    starts = _starts(lens)
    out = np.asarray(out, np.float64)
    assert out.shape == (1, bucket, H, HV)
    out = out[0]
    np.testing.assert_array_equal(out[starts[-1] :], 0.0, err_msg="bucket padding output")
    for r, n in enumerate(lens):
        if n:
            span = slice(starts[r], starts[r + 1])
            err = _rel(out[span], ref_out[span])
            assert err < tol, f"row {r}: output relative error {err:.2e}"
    # Conv windows hold raw inputs in the pool's dtype: exact for touched rows, untouched elsewhere.
    for i, name in enumerate(LEAVES[:3]):
        want = ref_conv[i]
        if pool_dtype is not None:
            assert leaves[name].dtype == jnp.dtype(pool_dtype), f"{name} dtype {leaves[name].dtype}"
            # The raw inputs rounded to the pool dtype by a device convert, as the driver stores them.
            want = np.asarray(jnp.asarray(want, jnp.float32).astype(pool_dtype), np.float64)
        np.testing.assert_array_equal(np.asarray(leaves[name], np.float64), want, err_msg=name)
    owned = {slots[r] for r, n in enumerate(lens) if n}
    rec = np.asarray(leaves["recurrent_state"], np.float64)
    for slot in range(rec.shape[0]):
        if slot in owned:
            err = _rel(rec[slot], ref_rec[slot])
            assert err < tol, f"slot {slot}: recurrent relative error {err:.2e}"
        else:
            np.testing.assert_array_equal(rec[slot], case["rec"][slot], err_msg=f"untouched slot {slot}")


SCENARIOS = [
    pytest.param(MIXED, 1, 64, LOWER_BOUND, id="mixed-tp1-cs64-bound"),
    pytest.param(MIXED, 1, 64, None, id="mixed-tp1-cs64-dense"),
    pytest.param(MIXED, 1, 16, LOWER_BOUND, id="mixed-tp1-cs16-bound"),
    pytest.param(MIXED, 4, 64, LOWER_BOUND, id="mixed-tp4-cs64-bound"),
    pytest.param(MIXED, 4, 16, None, id="mixed-tp4-cs16-dense"),
    pytest.param(DECODE, 1, 64, LOWER_BOUND, id="decode-tp1"),
    pytest.param(DECODE, 4, 64, LOWER_BOUND, id="decode-tp4"),
    pytest.param(SHORT, 1, 64, LOWER_BOUND, id="short-continuations-tp1"),
    pytest.param(SHORT, 4, 16, None, id="short-continuations-tp4-cs16-dense"),
]


@pytest.mark.parametrize(("layout", "tp", "chunk_size", "decay_lower_bound"), SCENARIOS)
def test_ragged_driver_matches_reference(layout, tp, chunk_size, decay_lower_bound):
    case = _case(layout, seed=0, dtype=jnp.float32, zero_fresh_recurrent=False)
    result = _run_ragged(case, layout, jnp.float32, tp=tp, chunk_size=chunk_size, decay_lower_bound=decay_lower_bound)
    _check(result, case, layout, tol=1e-4)


@pytest.mark.parametrize("layout", [MIXED, DECODE], ids=["mixed", "decode"])
def test_ragged_driver_with_serving_pool_layout(layout):
    """Pools arriving in the eSurge KDA cache layout (conv windows split by feature, states by head over tp4)."""
    case = _case(layout, seed=3, dtype=jnp.float32, zero_fresh_recurrent=False)
    result = _run_ragged(
        case, layout, jnp.float32, tp=4, chunk_size=64, decay_lower_bound=LOWER_BOUND, serving_layout=True
    )
    _check(result, case, layout, tol=1e-4)


# (layout, pool slots, slot of every row, tp). eSurge runs a step as windows of its request rows (a second window when
# max_num_seqs exceeds the rows per forward; rows packed past zero-token gaps), so row r's state lives in slot
# ``slots[r]``, not ``r``; rows without tokens name slot 0, which another request may own.
SLOT_CASES = [
    pytest.param(MIXED, 12, (5, 6, 7, 0, 9), 1, id="mixed-second-window-tp1"),
    pytest.param(MIXED, 12, ((5, 6, 7, 0, 9),), 4, id="mixed-second-window-1xrows-tp4"),
    pytest.param(MIXED, 7, (4, 0, 6, 0, 2), 1, id="mixed-permuted-empty-row-names-a-live-slot"),
    pytest.param(DECODE, 14, (6, 7, 8, 9, 0, 10), 4, id="decode-second-window-tp4"),
]


@pytest.mark.parametrize(("layout", "pool_slots", "slots", "tp"), SLOT_CASES)
def test_ragged_driver_with_explicit_slots(layout, pool_slots, slots, tp):
    """Row r reads and writes pool slot ``slots[r]``; rows without tokens and every unowned slot are left alone."""
    case = _case(layout, seed=5, dtype=jnp.float32, zero_fresh_recurrent=False, pool_slots=pool_slots)
    result = _run_ragged(
        case,
        layout,
        jnp.float32,
        tp=tp,
        chunk_size=64,
        decay_lower_bound=LOWER_BOUND,
        serving_layout=tp > 1,
        state_indices=slots,
    )
    _check(result, case, layout, tol=1e-4, slots=np.asarray(slots).reshape(-1).tolist())


def test_ragged_driver_rejects_pools_and_slots_that_cannot_cover_the_rows():
    """Without slots row r owns slot r (the pool needs a slot per row); explicit slots need one entry per row."""
    small = _case(MIXED, seed=6, dtype=jnp.float32, zero_fresh_recurrent=False, pool_slots=3)
    with pytest.raises(ValueError, match="at least 5 slots"):
        _run_ragged(small, MIXED, jnp.float32, tp=1, chunk_size=64, decay_lower_bound=LOWER_BOUND)
    case = _case(MIXED, seed=6, dtype=jnp.float32, zero_fresh_recurrent=False)
    with pytest.raises(ValueError, match="one pool slot per row"):
        _run_ragged(
            case, MIXED, jnp.float32, tp=1, chunk_size=64, decay_lower_bound=LOWER_BOUND, state_indices=(0, 1, 2)
        )


def test_ragged_driver_bf16_matches_reference():
    case = _case(MIXED, seed=1, dtype=jnp.bfloat16, zero_fresh_recurrent=False)
    result = _run_ragged(case, MIXED, jnp.bfloat16, tp=1, chunk_size=64, decay_lower_bound=LOWER_BOUND)
    _check(result, case, MIXED, tol=3e-2)


# (stream dtype, conv-window pool dtype, conv output dtype, tolerance): an fp8 KV-cache dtype (eSurge builds the KDA
# views in it) under a bf16 model, an f32 pool under bf16 projections, a bf16 pool under f32 projections, and an fp8
# model whose conv output the layer keeps in bf16.
POOL_DTYPES = [
    pytest.param(jnp.bfloat16, FP8, jnp.bfloat16, 3e-2, id="bf16-stream-fp8-pool"),
    pytest.param(jnp.bfloat16, jnp.float32, jnp.bfloat16, 3e-2, id="bf16-stream-f32-pool"),
    pytest.param(jnp.float32, jnp.bfloat16, jnp.float32, 1e-4, id="f32-stream-bf16-pool"),
    pytest.param(FP8, FP8, jnp.bfloat16, 3e-2, id="fp8-stream-fp8-pool"),
]


@pytest.mark.parametrize(("dtype", "pool_dtype", "conv_dtype", "tol"), POOL_DTYPES)
def test_ragged_driver_conv_pool_dtype_differs_from_stream(dtype, pool_dtype, conv_dtype, tol):
    """Stream and carried windows meet in one compute dtype (float32 for fp8); new windows keep the pool's dtype."""
    case = _case(MIXED, seed=4, dtype=dtype, zero_fresh_recurrent=False, pool_dtype=pool_dtype)
    result = _run_ragged(
        case,
        MIXED,
        dtype,
        tp=1,
        chunk_size=64,
        decay_lower_bound=LOWER_BOUND,
        pool_dtype=pool_dtype,
        conv_dtype=conv_dtype,
    )
    _check(result, case, MIXED, tol=tol, pool_dtype=pool_dtype)


@pytest.mark.parametrize("layout", [MIXED, DECODE], ids=["mixed", "decode"])
def test_row_padded_driver_matches_reference(layout):
    case = _case(layout, seed=2, dtype=jnp.float32, zero_fresh_recurrent=True)
    _check(_run_packed(case, layout, jnp.float32, chunk_size=64), case, layout, tol=1e-4)


def test_adapter_reaches_ejkernel_only_through_its_public_modules():
    """lint-imports enforces only the package layering (foundation libs never import easydel), not this rule."""
    lines = Path(adapter_module.__file__).read_text().splitlines()
    ejkernel_imports = [line for line in lines if line.startswith(("from ", "import ")) and "ejkernel" in line]
    assert ejkernel_imports
    assert all(line.startswith("from ejkernel.modules") for line in ejkernel_imports)


def test_ragged_kda_splits_heads_over_the_metadata_mesh_head_axis():
    mesh = _mesh(4)
    op = RaggedKDAOp(OperationMetadata(runtime_dtype=jnp.float32, _stored_mesh=mesh))
    for heads, tokens in ((8, 128), (8, 1)):
        axis = op.resolve_head_axis(heads, tokens)
        assert axis is not None and mesh_axis_size(mesh, axis) == 4
    assert op.resolve_head_axis(6, 128) is None  # 6 heads do not split 4 ways
    single = RaggedKDAOp(OperationMetadata(runtime_dtype=jnp.float32, _stored_mesh=_mesh(1)))
    assert single.resolve_head_axis(8, 128) is None
