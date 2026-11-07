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

"""Public ``ragged_kda`` operation (flat-token KDA over a recurrent-state pool) against a float64 reference.

The reference is a plain numpy float64 loop over rows, heads and tokens that spells out the operation's contract and
shares no code with ejkernel. Per row and head, token by token (``q`` and ``k`` optionally L2-normalised first, then
``q`` scaled by ``K ** -0.5``)::

    S = S * exp(g_t)[:, None];   kv = k_t^T S;   S = S + outer(k_t, beta_t * (v_t - kv));   o_t = S^T q_t

Every case checks the outputs and the touched pool slots against it, plus the "leave everything else alone" half of the
contract: padding tokens and tokens of invalid-slot rows come out exactly zero, and every pool slot not owned by a row
with at least one token is bit-identical afterwards. These are computation tests: the suite's conftest refuses to run
them on a CPU backend. Compilations are shared between tests through :func:`run`.
"""

from __future__ import annotations

import functools
import itertools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels import Backend, Platform, kernel_registry
from ejkernel.modules.operations import RaggedKDA, RaggedKDAConfig, ragged_kda, ragged_kda_head_sharded
from jax.sharding import Mesh, PartitionSpec
from jax.typing import DTypeLike

NUM_HEADS, KEY_DIM, VALUE_DIM = 4, 16, 32
NUM_SLOTS = 12  # pool slots; the 8-row batches own 8 of them, so some slots always belong to nobody
BUCKET = 256  # flat token buffer of the main geometry
DECAY_BOUND = -5.0  # decay_lower_bound used with decay sampled in (-5, 0)
FP32_TOL = 1e-4  # relative L2 error vs the float64 reference, fp32 everything
BF16_TOL = 2e-2  # relative L2 error vs the float64 reference, bf16 inputs / storage

# Row layouts of the main geometry (8 rows, 256-token bucket, 12-slot pool):
#   decode, long (partial chunks), short, empty, decode, one token past a 64-chunk, one 16-chunk, two tokens
MIXED_LENS = (1, 150, 5, 0, 1, 65, 16, 2)  # 240 live tokens + 16 padding tokens
PREFILL_LENS = (40, 30, 64, 17, 2, 33, 50, 20)  # every row multi-token; the last one ends exactly at the bucket end
DECODE_LENS = (1,) * 8  # every row a single token, 248 padding tokens
# has_initial_state for the MIXED_LENS rows: False for a decode row, the short row, the empty row, the 65-token row and
# the 2-token row; True for the long row, the other decode row and the 16-token row.
HAS_INIT = (False, True, False, False, True, False, True, False)
# Token-bucket lengths for calls that must find no entry in the executor's per-signature config cache (see
# test_default_cfg_is_automatic_inverse_and_pairwise): every use takes the next unused length, so the call signature
# is new even when the test runs twice in one process.
_FRESH_BUCKETS = itertools.count(32)


class Batch(tp.NamedTuple):
    """The nine array arguments of one ``ragged_kda`` call, in call order (splat it into the operation)."""

    query: jax.Array
    key: jax.Array
    value: jax.Array
    beta: jax.Array
    decay: jax.Array
    recurrent_state: jax.Array
    query_start_loc: jax.Array
    state_indices: jax.Array
    has_initial_state: jax.Array | None = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_batch(
    seed: int,
    lens: tuple[int, ...],
    *,
    tokens: int = BUCKET,
    num_slots: int = NUM_SLOTS,
    heads: int = NUM_HEADS,
    decay: str = "strong",
    qkvb_dtype: DTypeLike = jnp.float32,
    pool_dtype: DTypeLike = jnp.float32,
) -> Batch:
    """Random packed batch: row ``r`` owns ``lens[r]`` consecutive tokens, the rest of the buffer is padding.

    Padding tokens carry random data like live ones, so any leak of them into outputs or states shows up. The pool
    holds random nonzero states and ``state_indices`` is a random injective map of the rows into it.

    Args:
        seed: RNG seed.
        lens: Token count of every row (zero allowed).
        tokens: Length ``T`` of the flat token buffer (``>= sum(lens)``).
        num_slots: Pool size ``S`` (``>= len(lens)``).
        heads: Head count ``H``.
        decay: ``"strong"`` (log-decay in ``(-5, 0)``), ``"weak"`` (about ``-0.08``) or ``"zero"`` (no decay).
        qkvb_dtype: Dtype of query, key, value and beta.
        pool_dtype: Dtype of the state pool.

    Returns:
        The batch; ``decay`` is always fp32.
    """
    assert sum(lens) <= tokens and len(lens) <= num_slots
    rng = np.random.default_rng(seed)

    def normal(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    if decay == "strong":
        g = -5.0 * _sigmoid(normal(tokens, heads, KEY_DIM))
    elif decay == "weak":
        g = -0.1 * np.logaddexp(0.0, normal(tokens, heads, KEY_DIM))
    elif decay == "zero":
        g = np.zeros((tokens, heads, KEY_DIM))
    else:
        raise ValueError(f"unknown decay style {decay!r}")
    qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    slots = rng.permutation(num_slots)[: len(lens)].astype(np.int32)
    return Batch(
        query=jnp.asarray(normal(tokens, heads, KEY_DIM), qkvb_dtype),
        key=jnp.asarray(normal(tokens, heads, KEY_DIM), qkvb_dtype),
        value=jnp.asarray(normal(tokens, heads, VALUE_DIM), qkvb_dtype),
        beta=jnp.asarray(_sigmoid(normal(tokens, heads)), qkvb_dtype),
        decay=jnp.asarray(g, jnp.float32),
        recurrent_state=jnp.asarray(0.5 * normal(num_slots, heads, KEY_DIM, VALUE_DIM), pool_dtype),
        query_start_loc=jnp.asarray(qsl),
        state_indices=jnp.asarray(slots),
    )


def _f64(x: jax.Array | np.ndarray) -> np.ndarray:
    return np.asarray(x).astype(np.float64)


def rel_l2(a: jax.Array | np.ndarray, b: jax.Array | np.ndarray) -> float:
    """Relative L2 error ``||a - b|| / ||b||`` in float64."""
    a, b = _f64(a), _f64(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def live_rows(batch: Batch) -> list[tuple[int, int, int, int]]:
    """``(row, first_token, end_token, slot)`` of every row that owns at least one token and a valid pool slot."""
    qsl, slots = np.asarray(batch.query_start_loc), np.asarray(batch.state_indices)
    num_slots = batch.recurrent_state.shape[0]
    rows = []
    for row, slot in enumerate(slots.tolist()):
        lo, hi = int(qsl[row]), int(qsl[row + 1])
        if hi > lo and 0 <= slot < num_slots:
            rows.append((row, lo, hi, slot))
    return rows


def reference_ragged_kda(batch: Batch, *, use_qk_l2norm: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Plain float64 numpy loop over rows, heads and tokens (no ejkernel code); returns ``(out, new_pool)``.

    Args:
        batch: Inputs; low-precision arrays are widened exactly, so the reference sees the same values as the kernel.
        use_qk_l2norm: L2-normalise ``q`` and ``k`` (``x / sqrt(sum(x * x) + 1e-6)``) before the ``q`` scaling.

    Returns:
        ``(out [T, H, V], new_pool [S, H, K, V])``; padding and invalid-slot tokens are zero, other slots unchanged.
    """
    q, k, v, beta, g, pool = (
        _f64(x) for x in (batch.query, batch.key, batch.value, batch.beta, batch.decay, batch.recurrent_state)
    )
    scale = q.shape[-1] ** -0.5
    has_init = None if batch.has_initial_state is None else np.asarray(batch.has_initial_state)
    out = np.zeros(v.shape)
    new_pool = pool.copy()
    # Only live tokens are ever read, so whatever the rest of the buffer holds cannot reach the reference.
    for row, lo, hi, slot in live_rows(batch):
        q_row, k_row = q[lo:hi], k[lo:hi]
        if use_qk_l2norm:
            q_row = q_row / np.sqrt(np.sum(q_row * q_row, axis=-1, keepdims=True) + 1e-6)
            k_row = k_row / np.sqrt(np.sum(k_row * k_row, axis=-1, keepdims=True) + 1e-6)
        q_row = q_row * scale
        start_from_zero = has_init is not None and not has_init[row]
        for h in range(q.shape[1]):
            state = np.zeros(pool.shape[2:]) if start_from_zero else pool[slot, h].copy()
            for t in range(lo, hi):
                state = state * np.exp(g[t, h])[:, None]
                kv = k_row[t - lo, h] @ state
                state = state + np.outer(k_row[t - lo, h], beta[t, h] * (v[t, h] - kv))
                out[t, h] = q_row[t - lo, h] @ state
            new_pool[slot, h] = state
    return out, new_pool


def _fmt(errors: dict[int, float]) -> str:
    return ", ".join(f"row {row}: {err:.2e}" for row, err in errors.items())


def check_against_reference(
    batch: Batch,
    result: tuple[jax.Array, jax.Array],
    *,
    out_tol: float = FP32_TOL,
    pool_tol: float | None = None,
    use_qk_l2norm: bool = True,
) -> None:
    """Assert ``result = (out, new_pool)`` honours the whole ``ragged_kda`` contract for ``batch``.

    Per live row (a row with at least one token and a valid slot) the output tokens and the new pool slot must be within
    ``out_tol`` / ``pool_tol`` relative L2 error of the float64 reference. Everything else must be left alone: padding
    tokens and tokens of invalid-slot rows come out exactly zero, and every pool slot not owned by a live row is
    bit-identical to the input pool. Dtypes follow the contract (fp32 outputs, the pool keeps its dtype).

    Args:
        batch: The inputs the result was computed from.
        result: ``(out, new_pool)`` returned by the operation.
        out_tol: Allowed per-row relative L2 error of the outputs.
        pool_tol: Allowed per-row relative L2 error of the touched pool slots (defaults to ``out_tol``).
        use_qk_l2norm: Whether the operation was asked to L2-normalise ``q`` and ``k``.
    """
    pool_tol = out_tol if pool_tol is None else pool_tol
    out, new_pool = np.asarray(result[0]), np.asarray(result[1])
    pool_in = np.asarray(batch.recurrent_state)
    tokens, heads, _ = batch.query.shape
    assert out.shape == (tokens, heads, batch.value.shape[-1])
    assert out.dtype == np.float32
    assert new_pool.shape == pool_in.shape
    assert new_pool.dtype == pool_in.dtype
    assert np.isfinite(out).all()
    assert np.isfinite(new_pool.astype(np.float32)).all()

    ref_out, ref_pool = reference_ragged_kda(batch, use_qk_l2norm=use_qk_l2norm)
    rows = live_rows(batch)
    out_err = {row: rel_l2(out[lo:hi], ref_out[lo:hi]) for row, lo, hi, _ in rows}
    pool_err = {row: rel_l2(new_pool[slot], ref_pool[slot]) for row, _, _, slot in rows}
    worst_out, worst_pool = max(out_err.values(), default=0.0), max(pool_err.values(), default=0.0)
    print(f"max per-row rel L2 vs float64 reference: out {worst_out:.2e}, pool {worst_pool:.2e}")
    assert worst_out < out_tol, f"output rel L2 error exceeds {out_tol:g}: {_fmt(out_err)}"
    assert worst_pool < pool_tol, f"pool rel L2 error exceeds {pool_tol:g}: {_fmt(pool_err)}"

    owned = np.zeros(tokens, bool)
    for _, lo, hi, _ in rows:
        owned[lo:hi] = True
    assert not out[~owned].any(), "padding and invalid-slot tokens must produce exactly zero output"

    touched = {slot for *_, slot in rows}
    untouched = [slot for slot in range(pool_in.shape[0]) if slot not in touched]
    assert new_pool[untouched].tobytes() == pool_in[untouched].tobytes(), "pool slots without a live row changed"


@functools.cache
def _compiled(
    chunk_size: int,
    inverse: str,
    pairwise: str,
    decay_lower_bound: float | None,
    use_qk_l2norm: bool,
    platform: str | None,
):
    return jax.jit(
        functools.partial(
            ragged_kda,
            chunk_size=chunk_size,
            use_qk_l2norm=use_qk_l2norm,
            decay_lower_bound=decay_lower_bound,
            precision="highest",
            platform=platform,
            # Always an explicit config, so no call depends on the executor's per-signature config cache.
            cfg=RaggedKDAConfig(inverse=inverse, pairwise=pairwise),
        )
    )


def run(
    batch: Batch,
    *,
    chunk_size: int = 64,
    inverse: str = "auto",
    pairwise: str = "auto",
    decay_lower_bound: float | None = None,
    use_qk_l2norm: bool = True,
    platform: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Jit-compiled ``ragged_kda`` on ``batch`` (one compilation per distinct option set and input signature)."""
    return _compiled(chunk_size, inverse, pairwise, decay_lower_bound, use_qk_l2norm, platform)(*batch)


def _head_mesh() -> Mesh:
    return Mesh(np.asarray(jax.local_devices()), ("tp",))


def _assert_heads_split(arrays: tuple[jax.Array, ...], heads_per_device: int) -> None:
    """Every device holds only its own heads (axis 1) of every array: the ``shard_map`` path really ran."""
    for array in arrays:
        assert {shard.data.shape[1] for shard in array.addressable_shards} == {heads_per_device}


# --- numerics against the float64 reference -------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("decay", ["weak", "strong", "zero"])
def test_mixed_batch_matches_reference(chunk_size, decay):
    """Decode row, rows crossing partial chunks, short row, empty row and bucket padding in one step."""
    batch = make_batch(1, MIXED_LENS, decay=decay)
    check_against_reference(batch, run(batch, chunk_size=chunk_size))


def test_explicit_xla_platform_matches_reference():
    """The XLA fallback, selected explicitly through ``platform="xla"``."""
    batch = make_batch(2, MIXED_LENS)
    check_against_reference(batch, run(batch, platform="xla"))


def test_has_initial_state_false_rows_start_from_zero():
    """A False row ignores its (nonzero) slot, a True row continues from it, an empty row leaves its slot alone."""
    base = make_batch(3, MIXED_LENS, decay="weak")
    batch = base._replace(has_initial_state=jnp.asarray(HAS_INIT))
    out, new_pool = run(batch)
    check_against_reference(batch, (out, new_pool))

    # Metamorphic: a False flag equals a zeroed slot, and an all-True flag equals no flag at all.
    pool = np.array(base.recurrent_state)
    for slot, has_state in zip(np.asarray(base.state_indices).tolist(), HAS_INIT, strict=True):
        if not has_state:
            pool[slot] = 0.0
    zeroed_out, zeroed_pool = run(base._replace(recurrent_state=jnp.asarray(pool)))
    touched = [slot for *_, slot in live_rows(batch)]
    assert rel_l2(out, zeroed_out) < 1e-5
    assert rel_l2(np.asarray(new_pool)[touched], np.asarray(zeroed_pool)[touched]) < 1e-5

    plain_out, plain_pool = run(base)
    all_true_out, all_true_pool = run(base._replace(has_initial_state=jnp.ones(len(HAS_INIT), bool)))
    assert rel_l2(all_true_out, plain_out) < 1e-5
    assert rel_l2(all_true_pool, plain_pool) < 1e-5
    # ... and the flag is not a no-op for the rows it clears.
    for row, lo, hi, _ in live_rows(batch):
        if not HAS_INIT[row]:
            assert rel_l2(out[lo:hi], plain_out[lo:hi]) > 1e-2, f"row {row} ignores has_initial_state=False"


OOB_CASES = {
    "long_row_slot_equals_num_slots": ((1,), NUM_SLOTS),
    "decode_rows_negative_slot": ((0, 4), -1),
    "empty_row_far_out_of_range": ((3,), NUM_SLOTS + 37),
    "every_row_invalid": (tuple(range(8)), NUM_SLOTS),
}


@pytest.mark.parametrize(("rows", "slot"), list(OOB_CASES.values()), ids=list(OOB_CASES))
def test_out_of_range_slot_rows_are_padding(rows, slot):
    """A row with an out-of-range slot owns no state: zero output for its tokens, no slot touched on its behalf."""
    base = make_batch(4, MIXED_LENS)
    slots = np.array(base.state_indices)
    slots[list(rows)] = slot
    batch = base._replace(state_indices=jnp.asarray(slots))
    check_against_reference(batch, run(batch))


def test_empty_row_sharing_a_live_rows_slot_leaves_it_alone():
    """Padding rows often alias a real slot; with zero tokens they must not clobber it."""
    base = make_batch(5, MIXED_LENS)
    slots = np.array(base.state_indices)
    slots[3] = slots[1]  # the empty row 3 now points at the 150-token row's slot
    batch = base._replace(state_indices=jnp.asarray(slots))
    check_against_reference(batch, run(batch))


@pytest.mark.parametrize("chunk_size", [16, 64])
def test_tokens_before_the_first_row_belong_to_nobody(chunk_size):
    """``query_start_loc[0] > 0``: the leading tokens own no state and come out as exact zeros, like the padding."""
    base = make_batch(27, MIXED_LENS)
    batch = base._replace(query_start_loc=base.query_start_loc + 7)
    check_against_reference(batch, run(batch, chunk_size=chunk_size))


NON_FINITE_FILLS = {"nan": np.nan, "inf": np.inf}


@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("fill", list(NON_FINITE_FILLS.values()), ids=list(NON_FINITE_FILLS))
def test_non_finite_tokens_outside_live_rows_do_not_leak(fill, chunk_size):
    """NaN / inf in the bucket padding and in the tokens of invalid-slot rows must not reach any live row or slot.

    A kernel that masked chunk windows by multiplication (``0 * nan = nan``) instead of by selection would pass the
    random-padding cases above and fail this one. Rows 0 (a decode row) and 5 (65 tokens, between live rows) get an
    invalid slot, so they own no state. Near the end of the buffer a chunk window is clamped back, so the window of a
    live row can contain this garbage too.
    """
    base = make_batch(24, MIXED_LENS)
    invalid_rows = (0, 5)
    slots = np.array(base.state_indices)
    slots[list(invalid_rows)] = NUM_SLOTS
    qsl = np.asarray(base.query_start_loc)
    garbage = np.zeros(BUCKET, bool)
    garbage[qsl[-1] :] = True  # bucket padding
    for row in invalid_rows:
        garbage[qsl[row] : qsl[row + 1]] = True  # the invalid-slot rows' tokens

    def poisoned(x: jax.Array) -> jax.Array:
        data = np.array(x)
        data[garbage] = fill
        return jnp.asarray(data)

    streams = {name: poisoned(getattr(base, name)) for name in ("query", "key", "value", "beta", "decay")}
    batch = base._replace(**streams, state_indices=jnp.asarray(slots))
    check_against_reference(batch, run(batch, chunk_size=chunk_size))


@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("lens", [DECODE_LENS, PREFILL_LENS], ids=["pure_decode", "pure_prefill"])
def test_pure_decode_and_pure_prefill_steps(lens, chunk_size):
    """Only the decode phase (the chunk loop runs zero times) and only the chunk phase."""
    batch = make_batch(6, lens)
    check_against_reference(batch, run(batch, chunk_size=chunk_size))


SHORT_STREAMS = {
    "single_decode_token": ((1,), 1),
    "decode_rows_without_padding": ((1, 1, 1, 1), 4),
    "mixed_rows_with_padding": ((3, 1, 6, 0, 2), 16),
}


@pytest.mark.parametrize(("lens", "tokens"), list(SHORT_STREAMS.values()), ids=list(SHORT_STREAMS))
def test_stream_shorter_than_one_chunk(lens, tokens):
    """A token bucket smaller than ``chunk_size`` (a chunk window must still fit in the buffer)."""
    batch = make_batch(7, lens, tokens=tokens, num_slots=len(lens) + 2)
    check_against_reference(batch, run(batch, chunk_size=64))


def test_state_carries_across_steps():
    """One row served in pieces (prefill, prefill continuation, decode step) ends where the one-shot call does."""
    total, pieces = 100, (60, 39, 1)
    streams = ("query", "key", "value", "beta", "decay")
    one_shot = make_batch(22, (total,), tokens=128, num_slots=3)
    want_out, want_pool = run(one_shot)
    check_against_reference(one_shot, (want_out, want_pool))

    pool, outs, start = one_shot.recurrent_state, [], 0
    for size in pieces:
        # Roll the stream so the row's next tokens come first; the wrapped-around tokens are padding of this step.
        rolled = {name: jnp.roll(getattr(one_shot, name), -start, axis=0) for name in streams}
        step = one_shot._replace(**rolled, recurrent_state=pool, query_start_loc=jnp.asarray([0, size], jnp.int32))
        out, pool = run(step)
        outs.append(np.asarray(out)[:size])
        start += size
    slot = int(one_shot.state_indices[0])
    assert rel_l2(np.concatenate(outs), np.asarray(want_out)[:total]) < FP32_TOL
    assert rel_l2(pool[slot], want_pool[slot]) < FP32_TOL


# --- dtypes -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lens", [MIXED_LENS, DECODE_LENS, PREFILL_LENS], ids=["mixed", "pure_decode", "pure_prefill"])
def test_bf16_inputs_with_fp32_pool(lens):
    """bf16 q/k/v/beta (decay and pool stay fp32); the reference is fed the same bf16-rounded values."""
    batch = make_batch(8, lens, qkvb_dtype=jnp.bfloat16)
    assert batch.query.dtype == jnp.bfloat16 and batch.recurrent_state.dtype == jnp.float32
    check_against_reference(batch, run(batch), out_tol=BF16_TOL, pool_tol=BF16_TOL)


def test_bf16_pool_keeps_its_dtype():
    """The recurrence runs in fp32 but the pool is returned in its own dtype; untouched slots stay bit-identical."""
    batch = make_batch(9, MIXED_LENS, pool_dtype=jnp.bfloat16)
    out, new_pool = run(batch)
    assert out.dtype == jnp.float32 and new_pool.dtype == jnp.bfloat16
    check_against_reference(batch, (out, new_pool), out_tol=FP32_TOL, pool_tol=BF16_TOL)


def test_without_qk_l2norm_matches_reference():
    """``use_qk_l2norm=False``: keys are unit-norm so the delta rule stays contractive without the normalisation."""
    base = make_batch(10, MIXED_LENS)
    keys = np.asarray(base.key)
    batch = base._replace(key=jnp.asarray(keys / np.linalg.norm(keys, axis=-1, keepdims=True)))
    check_against_reference(batch, run(batch, use_qk_l2norm=False), use_qk_l2norm=False)


# --- decay_lower_bound / pairwise / chunk inverse / chunk size ---------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("pairwise", ["auto", "dense"])
def test_decay_lower_bound_matches_reference(pairwise, chunk_size):
    """Decay in [-5, 0) with ``decay_lower_bound=-5``: the factored (auto) and dense pairwise decays both match."""
    batch = make_batch(11, MIXED_LENS, decay="strong")
    assert float(np.asarray(batch.decay).min()) >= DECAY_BOUND
    result = run(batch, chunk_size=chunk_size, decay_lower_bound=DECAY_BOUND, pairwise=pairwise)
    check_against_reference(batch, result)


@pytest.mark.parametrize("chunk_size", [16, 64])
def test_decay_saturated_at_the_lower_bound_stays_accurate(chunk_size):
    """Every log-decay equal to the promised bound: the worst case for the factored pairwise decay's factors."""
    base = make_batch(11, MIXED_LENS, decay="strong")
    batch = base._replace(decay=jnp.full_like(base.decay, DECAY_BOUND))
    check_against_reference(batch, run(batch, chunk_size=chunk_size, decay_lower_bound=DECAY_BOUND))


@pytest.mark.parametrize("chunk_size", [16, 64])
def test_factored_pairwise_decay_agrees_with_dense_up_to_rounding(chunk_size):
    batch = make_batch(11, MIXED_LENS, decay="strong")
    options = dict(chunk_size=chunk_size, decay_lower_bound=DECAY_BOUND)
    auto_out, auto_pool = run(batch, pairwise="auto", **options)
    dense_out, dense_pool = run(batch, pairwise="dense", **options)
    # The bound has to really switch code paths (otherwise comparing the two proves nothing) ...
    assert not np.array_equal(np.asarray(auto_out), np.asarray(dense_out))
    # ... without changing the result beyond rounding.
    assert rel_l2(auto_out, dense_out) < FP32_TOL
    assert rel_l2(auto_pool, dense_pool) < FP32_TOL


def test_default_cfg_is_automatic_inverse_and_pairwise():
    """``cfg=None`` runs the heuristic config (auto / auto): bit-identical to an explicit ``RaggedKDAConfig()``."""
    batch = make_batch(12, (1, 20, 3, 0), tokens=next(_FRESH_BUCKETS), num_slots=6, heads=2)
    options = dict(chunk_size=16, decay_lower_bound=DECAY_BOUND)

    def call(**kwargs):
        return jax.jit(functools.partial(ragged_kda, **options, **kwargs))(*batch)

    # The default call goes first: the executor keeps an explicit cfg in its per-signature config cache, where a later
    # cfg=None call with the same signature would pick it up instead of the heuristic.
    implicit = call()
    explicit = call(cfg=RaggedKDAConfig())
    dense = call(cfg=RaggedKDAConfig(pairwise="dense"))
    check_against_reference(batch, implicit)
    assert np.array_equal(np.asarray(implicit[0]), np.asarray(explicit[0]))
    assert np.array_equal(np.asarray(implicit[1]), np.asarray(explicit[1]))
    # With a decay bound the automatic pairwise form is the factored one, which is not the dense program.
    assert not np.array_equal(np.asarray(implicit[0]), np.asarray(dense[0]))


@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("inverse", ["masked", "solve"])
def test_chunk_inverse_matches_reference(inverse, chunk_size):
    batch = make_batch(13, MIXED_LENS, decay="strong")
    check_against_reference(batch, run(batch, chunk_size=chunk_size, inverse=inverse))


@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("inverse", ["masked", "solve"])
def test_chunk_inverse_handles_nearly_parallel_keys(inverse, chunk_size):
    """Nearly parallel keys, beta 0.99 and almost no decay: the worst conditioning the chunk inverse meets."""
    base = make_batch(25, MIXED_LENS, decay="zero")
    rng = np.random.default_rng(26)
    direction = rng.standard_normal((1, NUM_HEADS, KEY_DIM))
    batch = base._replace(
        key=jnp.asarray(direction + 0.05 * rng.standard_normal(base.key.shape), jnp.float32),
        beta=jnp.full_like(base.beta, 0.99),
        decay=jnp.full_like(base.decay, -0.01),
    )
    check_against_reference(batch, run(batch, chunk_size=chunk_size, inverse=inverse))


@pytest.mark.parametrize("chunk_size", [16, 64])
def test_auto_inverse_is_masked_and_solve_agrees_up_to_rounding(chunk_size):
    """``inverse="auto"`` is the matmul-only ``masked`` inverse for power-of-two chunks (documented on the config)."""
    batch = make_batch(13, MIXED_LENS, decay="strong")
    auto, masked, solve = (run(batch, chunk_size=chunk_size, inverse=name) for name in ("auto", "masked", "solve"))
    assert np.array_equal(np.asarray(auto[0]), np.asarray(masked[0]))
    assert np.array_equal(np.asarray(auto[1]), np.asarray(masked[1]))
    assert not np.array_equal(np.asarray(masked[0]), np.asarray(solve[0])), "inverse=solve selected the same program"
    assert rel_l2(solve[0], masked[0]) < FP32_TOL
    assert rel_l2(solve[1], masked[1]) < FP32_TOL


def test_non_power_of_two_chunk_matches_reference():
    """chunk_size=24 cannot use the ``masked`` inverse, so ``auto`` falls back to the exact solve; factored decay too."""
    batch = make_batch(14, MIXED_LENS, decay="strong")
    check_against_reference(batch, run(batch, chunk_size=24, decay_lower_bound=DECAY_BOUND))


def test_jit_equals_eager():
    batch = make_batch(15, MIXED_LENS)
    eager = ragged_kda(*batch, chunk_size=64, cfg=RaggedKDAConfig())
    jitted = run(batch)
    check_against_reference(batch, eager)
    assert rel_l2(eager[0], jitted[0]) < 1e-5
    assert rel_l2(eager[1], jitted[1]) < 1e-5


# --- head-sharded execution -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("with_flags", [False, True], ids=["no_flags", "has_initial_state"])
def test_head_sharded_matches_unsharded(with_flags):
    """Heads split over every local device, no collectives: same result as the single-program call."""
    n = jax.local_device_count()
    if n < 2:
        pytest.skip("needs a head axis with more than one device")
    batch = make_batch(16, MIXED_LENS, heads=NUM_HEADS if NUM_HEADS % n == 0 else n)
    if with_flags:
        batch = batch._replace(has_initial_state=jnp.asarray(HAS_INIT))
    sharded = jax.jit(
        functools.partial(
            ragged_kda_head_sharded, mesh=_head_mesh(), head_axis="tp", chunk_size=64, cfg=RaggedKDAConfig()
        )
    )
    got = sharded(*batch)
    _assert_heads_split(got, batch.query.shape[1] // n)
    check_against_reference(batch, got)
    want = run(batch)
    assert rel_l2(got[0], want[0]) < 1e-5
    assert rel_l2(got[1], want[1]) < 1e-5


@pytest.mark.parametrize("head_axis", ["tp", ("dp", "tp")], ids=["tp_of_dp_x_tp", "dp_x_tp_combined"])
def test_head_sharded_on_a_two_axis_mesh(head_axis):
    """Heads over one axis of a ``dp x tp`` mesh (``dp`` replicates) or over both axes combined."""
    n = jax.local_device_count()
    if n < 4 or n % 2:
        pytest.skip("needs an even number of devices, at least four")
    mesh = Mesh(np.asarray(jax.local_devices()).reshape(n // 2, 2), ("dp", "tp"))
    ways = 2 if head_axis == "tp" else n
    heads = ways * -(-NUM_HEADS // ways)
    batch = make_batch(29, MIXED_LENS, heads=heads)
    fn = jax.jit(
        functools.partial(ragged_kda_head_sharded, mesh=mesh, head_axis=head_axis, chunk_size=64, cfg=RaggedKDAConfig())
    )
    got = fn(*batch)
    _assert_heads_split(got, heads // ways)
    check_against_reference(batch, got)
    want = run(batch)
    assert rel_l2(got[0], want[0]) < 1e-5
    assert rel_l2(got[1], want[1]) < 1e-5


def test_head_sharded_rejects_indivisible_head_count():
    n = jax.local_device_count()
    if n < 2:
        pytest.skip("needs a head axis with more than one device")
    batch = make_batch(17, MIXED_LENS, heads=n + 1)
    with pytest.raises(ValueError, match="not divisible"):
        ragged_kda_head_sharded(*batch, mesh=_head_mesh(), head_axis="tp", cfg=RaggedKDAConfig())


@pytest.mark.parametrize("fallback", ["mesh_none", "head_axis_none", "one_device_axis"])
def test_head_sharded_falls_back_to_the_unsharded_call(fallback):
    """No mesh, no head axis or a one-device head axis: the plain ``ragged_kda`` runs."""
    mesh, head_axis = {
        "mesh_none": (None, "tp"),
        "head_axis_none": (_head_mesh(), None),
        "one_device_axis": (Mesh(np.asarray(jax.local_devices()[:1]), ("tp",)), "tp"),
    }[fallback]
    batch = make_batch(18, MIXED_LENS)
    fn = jax.jit(
        functools.partial(ragged_kda_head_sharded, mesh=mesh, head_axis=head_axis, chunk_size=64, cfg=RaggedKDAConfig())
    )
    got = fn(*batch)
    check_against_reference(batch, got)
    want = run(batch)
    assert rel_l2(got[0], want[0]) < 1e-6
    assert rel_l2(got[1], want[1]) < 1e-6


@pytest.mark.parametrize(
    "bad_query_spec",
    [PartitionSpec("tp"), PartitionSpec(None, None, "tp")],
    ids=["token_axis", "feature_axis"],
)
def test_shard_map_wrapper_rejects_sharded_token_or_feature_axis(bad_query_spec):
    batch = make_batch(19, (1, 3), tokens=8, num_slots=3, heads=2)
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("tp",))
    token_head, state, replicated = (
        PartitionSpec(None, "tp", None),
        PartitionSpec(None, "tp", None, None),
        PartitionSpec(),
    )
    in_specs = (
        bad_query_spec,
        token_head,
        token_head,
        PartitionSpec(None, "tp"),
        token_head,
        state,
        replicated,
        replicated,
        replicated,
    )
    with pytest.raises(ValueError, match="unsharded token/feature axes"):
        RaggedKDA().create_shard_map_wrapper(
            *batch, mesh=mesh, in_specs=in_specs, out_specs=(token_head, state), cfg=RaggedKDAConfig()
        )


def test_shard_map_wrapper_requires_one_spec_per_array():
    batch = make_batch(19, (1, 3), tokens=8, num_slots=3, heads=2)
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("tp",))
    spec = PartitionSpec()
    with pytest.raises(ValueError, match="must have 9 entries"):
        RaggedKDA().create_shard_map_wrapper(*batch, mesh=mesh, in_specs=(spec,) * 8, out_specs=(spec, spec))


# --- identity slot layout (state_indices=None) -----------------------------------------------------------------------


@pytest.mark.parametrize("extra_slots", [0, 4], ids=["pool_equals_rows", "pool_larger_than_rows"])
@pytest.mark.parametrize("lens", [MIXED_LENS, DECODE_LENS, PREFILL_LENS], ids=["mixed", "pure_decode", "pure_prefill"])
def test_identity_slot_layout_matches_explicit_slots(lens, extra_slots):
    """``state_indices=None``: row ``r`` owns slot ``r``, the same contract as an explicit ``arange`` slot map.

    Slots past the rows and slots of empty rows must come out bit-identical (``check_against_reference``).
    """
    base = make_batch(30, lens, num_slots=len(lens) + extra_slots)
    explicit = base._replace(state_indices=jnp.arange(len(lens), dtype=jnp.int32))
    want = run(explicit)
    got = run(explicit._replace(state_indices=None))
    check_against_reference(explicit, want)
    check_against_reference(explicit, got)
    assert rel_l2(got[0], want[0]) < 1e-6
    assert rel_l2(got[1], want[1]) < 1e-6


def test_identity_slot_layout_with_has_initial_state():
    """``has_initial_state=False`` rows start from zero in the identity layout too."""
    base = make_batch(31, MIXED_LENS, decay="weak")
    explicit = base._replace(
        state_indices=jnp.arange(len(MIXED_LENS), dtype=jnp.int32), has_initial_state=jnp.asarray(HAS_INIT)
    )
    check_against_reference(explicit, run(explicit._replace(state_indices=None)))


def test_identity_slot_layout_bf16_pool():
    """bf16 pool in the identity layout: the decode rows' in-place update keeps the dtype and the untouched slots."""
    base = make_batch(32, MIXED_LENS, pool_dtype=jnp.bfloat16)
    explicit = base._replace(state_indices=jnp.arange(len(MIXED_LENS), dtype=jnp.int32))
    out, new_pool = run(explicit._replace(state_indices=None))
    assert new_pool.dtype == jnp.bfloat16
    check_against_reference(explicit, (out, new_pool), out_tol=FP32_TOL, pool_tol=BF16_TOL)


def test_head_sharded_identity_slot_layout():
    n = jax.local_device_count()
    if n < 2:
        pytest.skip("needs a head axis with more than one device")
    base = make_batch(33, MIXED_LENS, heads=NUM_HEADS if NUM_HEADS % n == 0 else n)
    explicit = base._replace(state_indices=jnp.arange(len(MIXED_LENS), dtype=jnp.int32))
    sharded = jax.jit(
        functools.partial(
            ragged_kda_head_sharded, mesh=_head_mesh(), head_axis="tp", chunk_size=64, cfg=RaggedKDAConfig()
        )
    )
    got = sharded(*explicit._replace(state_indices=None))
    _assert_heads_split(got, explicit.query.shape[1] // n)
    check_against_reference(explicit, got)


def test_identity_slot_layout_needs_a_slot_per_row():
    batch = make_batch(34, (1, 3, 2), tokens=8, num_slots=3, heads=2)
    with pytest.raises(ValueError, match="at least 3 slots"):
        ragged_kda(*batch._replace(state_indices=None, recurrent_state=batch.recurrent_state[:2]), cfg=RaggedKDAConfig())


# --- argument validation and registry ---------------------------------------------------------------------------------

OPTION_ERRORS = {
    "zero_chunk": dict(chunk_size=0),
    "negative_chunk": dict(chunk_size=-8),
    "positive_decay_bound": dict(decay_lower_bound=0.5),
    "nan_decay_bound": dict(decay_lower_bound=float("nan")),
    "unknown_precision": dict(precision="bf16"),
    "unknown_inverse": dict(cfg=RaggedKDAConfig(inverse="bogus")),
    "unknown_pairwise": dict(cfg=RaggedKDAConfig(pairwise="bogus")),
    "masked_inverse_needs_power_of_two_chunk": dict(chunk_size=12, cfg=RaggedKDAConfig(inverse="masked")),
    "masked_inverse_needs_chunk_of_at_least_8": dict(chunk_size=4, cfg=RaggedKDAConfig(inverse="masked")),
}


@pytest.mark.parametrize("options", list(OPTION_ERRORS.values()), ids=list(OPTION_ERRORS))
def test_invalid_options_raise_value_error(options):
    batch = make_batch(20, (1, 3), tokens=8, num_slots=3, heads=2)
    with pytest.raises(ValueError):
        ragged_kda(*batch, **options)


SHAPE_ERRORS = {
    "key_shape": lambda b: b._replace(key=b.key[..., :-1]),
    "value_tokens": lambda b: b._replace(value=b.value[:-1]),
    "beta_shape": lambda b: b._replace(beta=b.beta[:, :-1]),
    "decay_shape": lambda b: b._replace(decay=b.decay[..., :-1]),
    "pool_key_dim": lambda b: b._replace(recurrent_state=b.recurrent_state[:, :, :-1]),
    "query_start_loc_too_short": lambda b: b._replace(query_start_loc=b.query_start_loc[:1]),
    "state_indices_length": lambda b: b._replace(state_indices=b.state_indices[:-1]),
    "has_initial_state_length": lambda b: b._replace(has_initial_state=jnp.ones(len(b.state_indices) + 1, bool)),
}


@pytest.mark.parametrize("corrupt", list(SHAPE_ERRORS.values()), ids=list(SHAPE_ERRORS))
def test_shape_contract_violations_raise_value_error(corrupt):
    batch = corrupt(make_batch(21, (1, 3), tokens=8, num_slots=3, heads=2))
    with pytest.raises(ValueError):
        ragged_kda(*batch, cfg=RaggedKDAConfig())


def test_xla_fallback_is_registered_at_priority_zero():
    specs = kernel_registry.list_implementations("ragged_kda")
    assert any(s.platform == Platform.XLA and s.backend == Backend.ANY and s.priority == 0 for s in specs)


def test_config_roundtrip():
    cfg = RaggedKDAConfig(inverse="solve", pairwise="dense", platform="xla", backend="any")
    assert RaggedKDAConfig.from_dict(cfg.to_dict()) == cfg
    assert RaggedKDAConfig.from_json(cfg.to_json()) == cfg
    assert RaggedKDAConfig() == RaggedKDAConfig(inverse="auto", pairwise="auto", platform="auto", backend="any")
