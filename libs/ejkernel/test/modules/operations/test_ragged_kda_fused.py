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

"""``ragged_kda`` with the causal conv fused and per-token state capture against a float64 NumPy reference.

The reference is written token by token in NumPy (float64) from the op's contract, independently of the ejkernel code.
Per row with base slot ``g``, read offset ``a`` and commit length ``c``: the conv windows and the state are read from
slot ``g + a`` (zeros without an initial state); every token runs the depthwise causal conv with SiLU (output rounded to
the stream dtype) and the per-channel KDA delta rule; rows of at most ``W`` tokens round the windows and the state to
the pool dtype after every token and capture them after token ``c - 1 + j`` into slot ``g + j``; longer rows store their
final windows and state into ``g``. Every other slot stays bit for bit; padding tokens and rows that are not live output
zeros.

Backends: the XLA reference (module op, ``platform="xla"``), the Pallas TPU registry entry and the module op's
head-sharded ``shard_map`` path. Heads are 128 wide so the Pallas entry runs its own kernels.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_kda import ragged_kda as pallas_ragged_kda
from ejkernel.modules.operations import ragged_kda, ragged_kda_head_sharded
from jax.sharding import Mesh

K = V = 128  # lane-aligned heads: the Pallas TPU entry runs its kernels
D_CONV = 4
CHUNK = 64
LOWER = -5.0  # safe-gate bound; the Pallas chunk kernel needs the factored pairwise decay it allows
EPS = 1e-6  # the op's L2-norm epsilon
POOLS = ("cq", "ck", "cv", "rec")
DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}
TOL = {"float32": 2e-5, "bfloat16": 2e-2}

# Rows: lens, commit, read offset, has_initial_state. Base slots are group starts (group * W) of a seeded permutation;
# ``empty`` is a row without tokens that names the next row's group (it must not write); ``outside`` is a row whose
# base slot is out of range (a padding row: zero output, no write); ``pad`` padding tokens follow the rows.
LAYOUTS = {
    # Verify rows: commit < lens captures every draft position; read offsets inside the group; two fresh rows.
    "verify": dict(
        W=4,
        lens=(4, 3, 1, 4, 0, 2, 4, 1),
        commit=(2, 3, 1, 1, 1, 1, 4, 1),
        offset=(1, 0, 0, 3, 0, 2, 0, 0),
        init=(1, 1, 0, 1, 1, 1, 0, 1),
        empty=4,
        outside=7,
        pad=5,
    ),
    # Chunk rows (> W tokens; row 0 reads at offset 2, their commits are ignored) mixed with short rows in one call.
    "mixed": dict(
        W=4,
        lens=(70, 3, 1, 130, 0, 9, 4, 2),
        commit=(5, 2, 1, 1, 1, 3, 2, 1),
        offset=(2, 1, 0, 0, 0, 0, 3, 1),
        init=(1, 1, 1, 0, 1, 1, 1, 0),
        empty=4,
        outside=2,
        pad=11,
    ),
    # Ordinary serving step (W = 1): decode rows and prefill / continuation rows with explicit slots.
    "ordinary": dict(
        W=1,
        lens=(1, 70, 1, 0, 37, 1, 2, 1),
        commit=None,
        offset=None,
        init=(1, 0, 1, 1, 1, 0, 1, 1),
        empty=3,
        outside=7,
        pad=6,
    ),
    # The same rows in the identity layout (``state_indices=None``: row r's base slot is r).
    "ordinary_identity": dict(
        W=1,
        lens=(1, 70, 1, 0, 37, 1, 2, 1),
        commit=None,
        offset=None,
        init=(1, 0, 1, 1, 1, 0, 1, 1),
        empty=None,
        outside=None,
        pad=6,
        identity=True,
    ),
}


def _rounded(x, dtype) -> np.ndarray:
    """``x`` rounded to ``dtype``, as float64."""
    return np.asarray(x, np.float64).astype(np.float32).astype(dtype).astype(np.float64)


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))


def _make_streams(rng: np.random.Generator, num_tokens: int, heads: int, dtype) -> dict[str, np.ndarray]:
    kd, vd = heads * K, heads * V
    return {
        "q": _rounded(rng.normal(size=(num_tokens, kd)), dtype),
        "k": _rounded(rng.normal(size=(num_tokens, kd)), dtype),
        "v": _rounded(rng.normal(size=(num_tokens, vd)), dtype),
        "beta": _rounded(1.0 / (1.0 + np.exp(-rng.normal(size=(num_tokens, heads)))), dtype),
        "decay": _rounded(LOWER / (1.0 + np.exp(-rng.normal(size=(num_tokens, heads, K)))), np.float32),
    }


def _make_state(rng: np.random.Generator, num_slots: int, heads: int, dtype) -> dict[str, np.ndarray]:
    kd, vd = heads * K, heads * V
    return {
        "wq": _rounded(0.5 * rng.normal(size=(kd, D_CONV)), dtype),
        "wk": _rounded(0.5 * rng.normal(size=(kd, D_CONV)), dtype),
        "wv": _rounded(0.5 * rng.normal(size=(vd, D_CONV)), dtype),
        "cq": _rounded(rng.normal(size=(num_slots, kd, D_CONV)), dtype),
        "ck": _rounded(rng.normal(size=(num_slots, kd, D_CONV)), dtype),
        "cv": _rounded(rng.normal(size=(num_slots, vd, D_CONV)), dtype),
        "rec": _rounded(0.1 * rng.normal(size=(num_slots, heads, K, V)), dtype),
    }


def _resolve(layout: dict, seed: int):
    """``(lens, commit, offset, init, base slots, state_indices argument, num_slots)`` of a layout."""
    w = layout["W"]
    lens = np.asarray(layout["lens"], np.int64)
    rows = lens.shape[0]
    commit = lens.copy() if layout["commit"] is None else np.asarray(layout["commit"], np.int64)
    offset = np.zeros(rows, np.int64) if layout["offset"] is None else np.asarray(layout["offset"], np.int64)
    init = np.asarray(layout["init"], bool)
    groups = rows + 3
    num_slots = groups * w
    if layout.get("identity"):
        return lens, commit, offset, init, np.arange(rows), None, num_slots
    base = np.random.default_rng(seed + 1).permutation(groups)[:rows] * w
    if layout["empty"] is not None:
        base[layout["empty"]] = base[(layout["empty"] + 1) % rows]
    if layout["outside"] is not None:
        base[layout["outside"]] = num_slots + w
    return lens, commit, offset, init, base, base, num_slots


def _token(state, windows, x, weights, beta, decay, stream_dtype):
    """One token of the contract in float64: causal conv + SiLU (rounded to the stream dtype), then the delta rule."""
    heads = beta.shape[0]
    conv, new_windows = [], []
    for window, xi, w in zip(windows, x, weights, strict=True):
        window = np.concatenate([window[:, 1:], xi[:, None]], axis=1)
        new_windows.append(window)
        mixed = np.sum(window * w, axis=1)
        conv.append(_rounded(mixed / (1.0 + np.exp(-mixed)), stream_dtype))
    q, k = (c.reshape(heads, K) for c in conv[:2])
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + EPS) * K**-0.5
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + EPS)
    v = conv[2].reshape(heads, V)
    state = state * np.exp(decay)[:, :, None]
    delta = (v - np.einsum("hk,hkv->hv", k, state)) * beta[:, None]
    state = state + k[:, :, None] * delta[:, None, :]
    return np.einsum("hk,hkv->hv", q, state), state, new_windows


def _start(state0: dict, slot: int, init: bool):
    """The windows and the state a row starts from (zeros without an initial state)."""
    if init:
        return [state0[name][slot] for name in POOLS[:3]], state0["rec"][slot]
    return [np.zeros_like(state0[name][0]) for name in POOLS[:3]], np.zeros_like(state0["rec"][0])


def _reference(streams, state0, layout, resolved, stream_dtype, pool_dtype):
    """Expected ``(out, pools, written slots, live rows)`` of one call."""
    lens, commit, offset, init, base, _, num_slots = resolved
    w = layout["W"]
    num_tokens, heads = streams["beta"].shape
    starts = np.concatenate([[0], np.cumsum(lens)])
    out = np.zeros((num_tokens, heads, V))
    pools = {name: state0[name].copy() for name in POOLS}
    weights = [state0["wq"], state0["wk"], state0["wv"]]
    written, live_rows = set(), []
    for r, n in enumerate(lens.tolist()):
        g, a = int(base[r]), int(offset[r])
        short = n <= w
        c = min(max(int(commit[r]), 1), max(n, 1)) if short else n
        if n == 0 or g < 0 or a < 0 or g + a >= num_slots or g + (n - c) >= num_slots:
            continue
        live_rows.append(r)
        windows, state = _start(state0, g + a, bool(init[r]))
        for j in range(n):
            t = starts[r] + j
            x = [streams[name][t] for name in ("q", "k", "v")]
            out[t], state, windows = _token(
                state, windows, x, weights, streams["beta"][t], streams["decay"][t], stream_dtype
            )
            if short:
                # Stored between tokens in the pool dtype, exactly like separate decode steps.
                state = _rounded(state, pool_dtype)
                windows = [_rounded(win, pool_dtype) for win in windows]
                if j >= c - 1:
                    slot = g + j - (c - 1)
                    for name, win in zip(POOLS[:3], windows, strict=True):
                        pools[name][slot] = win
                    pools["rec"][slot] = state
                    written.add(slot)
        if not short:
            for name, win in zip(POOLS[:3], windows, strict=True):
                pools[name][g] = _rounded(win, pool_dtype)
            pools["rec"][g] = _rounded(state, pool_dtype)
            written.add(g)
    return out, pools, written, live_rows


def _arrays(streams, state0, layout, resolved, stream_dtype, pool_dtype, pools=None):
    """The op's 17 positional arrays (device pools from ``pools`` when given)."""
    lens, commit, offset, init, _, state_indices, _ = resolved
    num_tokens, heads = streams["beta"].shape

    def dev(x, dtype):
        return jnp.asarray(np.asarray(x, np.float32), dtype)

    if pools is None:
        pools = {name: dev(state0[name], pool_dtype) for name in POOLS}
    qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    return (
        dev(streams["q"], stream_dtype).reshape(num_tokens, heads, K),
        dev(streams["k"], stream_dtype).reshape(num_tokens, heads, K),
        dev(streams["v"], stream_dtype).reshape(num_tokens, heads, V),
        dev(streams["beta"], stream_dtype),
        dev(streams["decay"], jnp.float32),
        pools["rec"],
        jnp.asarray(qsl),
        None if state_indices is None else jnp.asarray(state_indices, jnp.int32),
        jnp.asarray(init),
        None if layout["offset"] is None else jnp.asarray(offset, jnp.int32),
        None if layout["commit"] is None else jnp.asarray(commit, jnp.int32),
        dev(state0["wq"], stream_dtype),
        dev(state0["wk"], stream_dtype),
        dev(state0["wv"], stream_dtype),
        pools["cq"],
        pools["ck"],
        pools["cv"],
    )


def _fn(backend: str, w: int, mesh: Mesh | None = None):
    kwargs = dict(short_row_max=w, chunk_size=CHUNK, decay_lower_bound=LOWER, precision="highest")
    if backend == "xla":
        return jax.jit(functools.partial(ragged_kda, platform="xla", **kwargs))
    if backend == "pallas":
        return jax.jit(functools.partial(pallas_ragged_kda, **kwargs))
    return jax.jit(functools.partial(ragged_kda_head_sharded, mesh=mesh, head_axis="tp", **kwargs))


def _f64(x) -> np.ndarray:
    return np.asarray(jnp.asarray(x, jnp.float32), np.float64)


def _check(got, ref, state0, resolved, dtype_name):
    out_ref, pools_ref, written, live_rows = ref
    lens, num_slots = resolved[0], resolved[-1]
    tol = TOL[dtype_name]
    assert len(got) == 5
    out = _f64(got[0])
    assert out.shape == out_ref.shape
    starts = np.concatenate([[0], np.cumsum(lens)])
    live = np.zeros(out.shape[0], bool)
    for r in live_rows:
        rows = slice(starts[r], starts[r] + lens[r])
        live[rows] = True
        assert _rel(out[rows], out_ref[rows]) < tol, f"row {r}: rel {_rel(out[rows], out_ref[rows]):.3e}"
    assert not np.any(out[~live]), "padding tokens and rows that are not live must output zeros"
    untouched = np.asarray([s for s in range(num_slots) if s not in written], np.int64)
    for name, arr in zip(POOLS, got[1:], strict=True):
        arr = _f64(arr)
        np.testing.assert_array_equal(arr[untouched], state0[name][untouched], err_msg=f"{name}: untouched slot changed")
        for s in sorted(written):
            assert _rel(arr[s], pools_ref[name][s]) < tol, f"{name} slot {s}: rel {_rel(arr[s], pools_ref[name][s]):.3e}"


@pytest.mark.parametrize(
    ("dtype_name", "heads"),
    [("float32", 4), ("float32", 16), ("bfloat16", 4)],
    ids=["f32-h4", "f32-h16", "bf16-h4"],
)
@pytest.mark.parametrize("backend", ["xla", "pallas"])
@pytest.mark.parametrize("layout_name", list(LAYOUTS))
def test_rows_match_float64_reference(layout_name, backend, dtype_name, heads):
    """Outputs, captured windows and states at every group offset, untouched slots, padding rows and tokens."""
    layout, dtype = LAYOUTS[layout_name], DTYPES[dtype_name]
    seed = 100 * heads + sorted(LAYOUTS).index(layout_name)
    resolved = _resolve(layout, seed)
    rng = np.random.default_rng(seed)
    streams = _make_streams(rng, int(resolved[0].sum()) + layout["pad"], heads, dtype)
    state0 = _make_state(rng, resolved[-1], heads, dtype)
    got = _fn(backend, layout["W"])(*_arrays(streams, state0, layout, resolved, dtype, dtype))
    _check(got, _reference(streams, state0, layout, resolved, dtype, dtype), state0, resolved, dtype_name)


# Two steps: step 1 verifies (row r captures g .. g + lens - commit), the sampler accepts ``accepted`` drafts, step 2
# reads slot g + accepted and continues.
CHAIN = dict(
    W=4,
    step1=dict(lens=(4, 4, 3, 2), commit=(1, 2, 3, 1), offset=(0, 1, 0, 2), init=(1, 0, 1, 1)),
    accepted=(2, 0, 0, 1),  # at most lens - commit of step 1
    step2=dict(lens=(4, 1, 3, 2), commit=(2, 1, 1, 2)),
)


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("backend", ["xla", "pallas"])
def test_verify_then_read_at_offset_chain(backend, dtype_name):
    """Step 2's outputs and captures equal one straight token sequence in float64.

    The sequence is step 1's first ``commit + accepted`` tokens followed by step 2's tokens, each rounded to the pool
    dtype after every token, starting from the slot step 1 read.
    """
    dtype, heads, w = DTYPES[dtype_name], 4, CHAIN["W"]
    step1, step2 = CHAIN["step1"], CHAIN["step2"]
    rows = len(step1["lens"])
    groups = rows + 2
    num_slots = groups * w
    base = np.random.default_rng(5).permutation(groups)[:rows] * w
    lens1, lens2 = np.asarray(step1["lens"]), np.asarray(step2["lens"])
    commit1, commit2 = np.asarray(step1["commit"]), np.asarray(step2["commit"])
    offset1, accepted = np.asarray(step1["offset"]), np.asarray(CHAIN["accepted"])
    init1 = np.asarray(step1["init"], bool)
    rng = np.random.default_rng(77)
    streams1 = _make_streams(rng, int(lens1.sum()) + 3, heads, dtype)
    streams2 = _make_streams(rng, int(lens2.sum()) + 2, heads, dtype)
    state0 = _make_state(rng, num_slots, heads, dtype)
    fn = _fn(backend, w)
    layout1 = dict(W=w, **step1)
    got1 = fn(*_arrays(streams1, state0, layout1, (lens1, commit1, offset1, init1, base, base, num_slots), dtype, dtype))
    layout2 = dict(W=w, offset=CHAIN["accepted"], init=(1,) * rows, **step2)
    resolved2 = (lens2, commit2, accepted, np.ones(rows, bool), base, base, num_slots)
    pools1 = dict(zip(POOLS, got1[1:], strict=True))
    got2 = fn(*_arrays(streams2, state0, layout2, resolved2, dtype, dtype, pools=pools1))

    tol = TOL[dtype_name]
    out2 = _f64(got2[0])
    pools2 = {name: _f64(arr) for name, arr in zip(POOLS, got2[1:], strict=True)}
    weights = [state0["wq"], state0["wk"], state0["wv"]]
    starts1 = np.concatenate([[0], np.cumsum(lens1)])
    starts2 = np.concatenate([[0], np.cumsum(lens2)])
    for r in range(rows):
        g = int(base[r])
        windows, state = _start(state0, g + int(offset1[r]), bool(init1[r]))
        tokens = [(streams1, starts1[r] + j) for j in range(int(commit1[r] + accepted[r]))]
        tokens += [(streams2, starts2[r] + j) for j in range(int(lens2[r]))]
        first2 = len(tokens) - int(lens2[r])
        for i, (src, t) in enumerate(tokens):
            x = [src[name][t] for name in ("q", "k", "v")]
            o, state, windows = _token(state, windows, x, weights, src["beta"][t], src["decay"][t], dtype)
            state = _rounded(state, dtype)
            windows = [_rounded(win, dtype) for win in windows]
            j = i - first2
            if j < 0:
                continue
            assert _rel(out2[t], o) < tol, f"row {r} step-2 token {j}: rel {_rel(out2[t], o):.3e}"
            if j >= commit2[r] - 1:
                slot = g + j - int(commit2[r] - 1)
                assert _rel(pools2["rec"][slot], state) < tol, f"row {r} capture slot {slot}"
                for name, win in zip(POOLS[:3], windows, strict=True):
                    assert _rel(pools2[name][slot], win) < tol, f"row {r} {name} capture slot {slot}"


@pytest.mark.parametrize("heads", [4, 16])
def test_head_sharded_matches_reference(heads):
    """The module op's ``shard_map`` over a 4-device head axis (1 and 4 heads per device), default platform."""
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("tp",))
    layout, dtype = LAYOUTS["mixed"], jnp.float32
    seed = 7 * heads
    resolved = _resolve(layout, seed)
    rng = np.random.default_rng(seed)
    streams = _make_streams(rng, int(resolved[0].sum()) + layout["pad"], heads, dtype)
    state0 = _make_state(rng, resolved[-1], heads, dtype)
    got = _fn("sharded", layout["W"], mesh=mesh)(*_arrays(streams, state0, layout, resolved, dtype, dtype))
    _check(got, _reference(streams, state0, layout, resolved, dtype, dtype), state0, resolved, "float32")


def _small_arrays():
    layout = LAYOUTS["verify"]
    resolved = _resolve(layout, 3)
    rng = np.random.default_rng(3)
    streams = _make_streams(rng, int(resolved[0].sum()) + layout["pad"], 2, jnp.float32)
    state0 = _make_state(rng, resolved[-1], 2, jnp.float32)
    return list(_arrays(streams, state0, layout, resolved, jnp.float32, jnp.float32))


def test_partial_conv_arrays_are_rejected():
    arrays = _small_arrays()
    arrays[14] = None  # q_conv_state missing while the other conv arrays are given
    with pytest.raises(ValueError, match="all six conv arrays"):
        ragged_kda(*arrays, short_row_max=4, decay_lower_bound=LOWER, platform="xla")


def test_short_row_max_must_be_positive():
    with pytest.raises(ValueError, match="short_row_max"):
        ragged_kda(*_small_arrays(), short_row_max=0, decay_lower_bound=LOWER, platform="xla")
