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

"""Rank-major (eSurge SPMD data-parallel) KDA serving driver against an independent float64 per-rank loop.

eSurge serves with ``data_parallel_size > 1`` by giving every DP rank its own token block, request rows and block of
recurrent pool slots: ``query_start_loc`` ``[dp, rows + 1]`` with rank-local offsets, ``context_lens`` ``[dp * rows]``
and rank-local slots ``[dp, rows]``. :func:`kda_ragged_rows_rank_major_forward` must equal a float64 loop that runs the
causal depthwise conv (SiLU) and the per-channel KDA recurrence token by token for every rank and row, on that rank's
block of slots: continuation rows start from their carried window and state, fresh rows from zero, rows without tokens
(also when they name a slot a live row owns, as eSurge's padding rows name slot 0) and slots no live row owns keep their
state bit for bit, and bucket padding gets zero output. The pools are placed like the eSurge KDA cache (slot axis over
the batch axes, heads / conv features over ``tp``). Computation tests: the suite's conftest refuses them on a CPU
backend.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.axis import register_attention_data_parallel_axis, reset_attention_data_parallel_axis
from easydel.caching import KDACacheView
from easydel.caching.kda.cache import KDACacheConfig
from easydel.operations import OperationMetadata
from easydel.operations.kernels.ragged_kda_rank_major import kda_ragged_rows_rank_major_forward
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax import PartitionAxis

H, HK, HV, DC = 4, 16, 16, 4
# Geometry of the decode-shaped cases: head sizes on the 128 TPU lanes, so the Pallas decode kernel runs per rank.
DECODE_HEADS, DECODE_HEAD_DIM = 4, 128
LEAVES = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
AXES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
LOWER_BOUND = -5.0

# Every rank: (lens, context_lens, rank-local slots). Ranks share the row count, tokens per rank and slots per rank.
# DP2 rank 0: a decode continuation, a fresh 21-token prompt (a full and a partial chunk at chunk size 16), and a
# padding row that names the fresh row's slot 0 (it must not write). DP2 rank 1: a multi-token continuation (chunked prefill), a fresh one-token prompt
# and a fresh prompt shorter than the conv window, on permuted slots (slot 2 of rank 1 is owned by no row).
DP2 = {
    "tokens_per_rank": 40,
    "slots_per_rank": 4,
    "ranks": [
        ((1, 21, 0), (6, 21, 0), (2, 0, 0)),
        ((5, 1, 3), (12, 1, 3), (1, 3, 0)),
    ],
}
# DP4: a fresh 20-token prompt plus a decode row; an empty rank; a continuation shorter than the conv window (its new
# window keeps part of the carried one) plus a fresh two-token prompt; a fresh one-token prompt plus a decode row.
DP4 = {
    "tokens_per_rank": 24,
    "slots_per_rank": 3,
    "ranks": [
        ((20, 1), (20, 9), (1, 0)),
        ((0, 0), (0, 0), (0, 0)),
        ((3, 2), (40, 2), (0, 2)),
        ((1, 1), (1, 33), (2, 1)),
    ],
}
# Decode-shaped token blocks (tokens per rank == rows per rank), as eSurge's smallest buckets give: every rank compiles
# the decode router's conditional and picks its branch for itself. DECODE_DP2 rank 0 is decode-only (two
# continuations, a fresh one-token prompt, and an empty row that names the first row's slot); rank 1 is mixed in the
# same step (a 2-token continuation, a decode row, two empty rows) and must take the flat driver.
DECODE_DP2 = {
    "tokens_per_rank": 4,
    "slots_per_rank": 5,
    "ranks": [
        ((1, 1, 0, 1), (9, 1, 0, 30), (3, 0, 3, 1)),
        ((2, 0, 1, 0), (7, 0, 12, 0), (4, 1, 0, 2)),
    ],
}
# DECODE_DP4: a decode-only rank, an empty rank, a fresh two-token prompt (flat driver), and a decode row next to an
# empty row that names its slot.
DECODE_DP4 = {
    "tokens_per_rank": 2,
    "slots_per_rank": 3,
    "ranks": [
        ((1, 1), (5, 17), (2, 0)),
        ((0, 0), (0, 0), (0, 0)),
        ((2, 0), (2, 0), (1, 0)),
        ((1, 0), (40, 0), (1, 1)),
    ],
}


@pytest.fixture(autouse=True)
def _default_attention_dp_axis():
    # eSurge engines register the attention data-parallel axis process-wide; these meshes use the default ("dp").
    reset_attention_data_parallel_axis()
    yield
    reset_attention_data_parallel_axis()


def _dims(layout):
    return len(layout["ranks"]), int(layout["tokens_per_rank"]), int(layout["slots_per_rank"])


def _starts(lens):
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def _slots(layout, explicit):
    rows = len(layout["ranks"][0][0])
    return [list(slots) if explicit else list(range(rows)) for _, _, slots in layout["ranks"]]


def _case(layout, seed, heads=H, head_dim=HK):
    """Random float32 inputs and pools stored as float64 (inputs ``[dp, T, ...]``, pools ``[dp * slots, ...]``).

    ``heads`` x ``head_dim`` is the head geometry (queries, keys and values share ``head_dim``).
    """
    dp, tokens, slots = _dims(layout)
    rng = np.random.default_rng(seed)
    width = heads * head_dim
    dims = (width, width, width)

    def f32(x):
        return np.asarray(jnp.asarray(x, jnp.float32), np.float64)

    return {
        "geom": (heads, head_dim, head_dim),
        "q": f32(rng.normal(size=(dp, tokens, width))),
        "k": f32(rng.normal(size=(dp, tokens, width))),
        "v": f32(rng.normal(size=(dp, tokens, width))),
        "beta": f32(1.0 / (1.0 + np.exp(-rng.normal(size=(dp, tokens, heads))))),
        "decay": f32(LOWER_BOUND / (1.0 + np.exp(-rng.normal(size=(dp, tokens, heads, head_dim))))),
        "kernels": [f32(0.5 * rng.normal(size=(d, DC))) for d in dims],
        "conv": [f32(rng.normal(size=(dp * slots, d, DC))) for d in dims],
        "rec": f32(0.1 * rng.normal(size=(dp * slots, heads, head_dim, head_dim))),
    }


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _reference(case, layout, slots_by_rank):
    """float64 per-rank, per-row causal depthwise conv (SiLU) and per-channel KDA, token by token.

    Returns the rank-major output ``[dp, T, H, HV]`` and the expected conv and recurrent pools.
    """
    dp, tokens, slots_per_rank = _dims(layout)
    heads, hk, hv = case["geom"]
    out = np.zeros((dp, tokens, heads, hv))
    conv = [c.copy() for c in case["conv"]]
    rec = case["rec"].copy()
    for d, (lens, context, _) in enumerate(layout["ranks"]):
        starts = _starts(lens)
        for r, n in enumerate(lens):
            if n == 0:
                continue
            slot = d * slots_per_rank + slots_by_rank[d][r]
            carried = context[r] > n
            span = slice(starts[r], starts[r] + n)
            mixed = []
            for i, name in enumerate(("q", "k", "v")):
                history = conv[i][slot] if carried else np.zeros_like(conv[i][slot])  # [d, DC], last column newest
                full = np.concatenate([history.T, case[name][d, span]], axis=0)  # [DC + n, d]
                kernel = case["kernels"][i].T  # [DC, d], last row is the current-token tap
                mixed.append(_silu(np.stack([np.sum(full[t + 1 : t + 1 + DC] * kernel, axis=0) for t in range(n)])))
                conv[i][slot] = full[-DC:].T
            q = mixed[0].reshape(n, heads, hk)
            k = mixed[1].reshape(n, heads, hk)
            v = mixed[2].reshape(n, heads, hv)
            q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * hk**-0.5
            k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
            state = rec[slot] if carried else np.zeros((heads, hk, hv))
            beta, decay = case["beta"][d, span], case["decay"][d, span]
            for t in range(n):
                state = state * np.exp(decay[t])[:, :, None]
                delta = beta[t][:, None] * (v[t] - np.einsum("hk,hkv->hv", k[t], state))
                state = state + k[t][:, :, None] * delta[:, None, :]
                out[d, starts[r] + t] = np.einsum("hk,hkv->hv", q[t], state)
            rec[slot] = state
    return out, conv, rec


def _mesh(dp, tp):
    if len(jax.devices()) < dp * tp:
        pytest.skip(f"needs {dp * tp} devices")
    return Mesh(np.asarray(jax.devices()[: dp * tp]).reshape(1, dp, 1, 1, tp, 1), AXES)


def _view(leaves):
    pool = int(leaves["recurrent_state"].shape[0])
    cfg = KDACacheConfig.create(
        num_hidden_layers=1,
        partition_axis=PartitionAxis(),
        batch_size=pool,
        key_dim=int(leaves["q_conv_state"].shape[1]),
        value_dim=int(leaves["v_conv_state"].shape[1]),
        d_conv=DC,
        recurrent_state_shape=tuple(int(d) for d in leaves["recurrent_state"].shape[1:]),
    )
    return KDACacheView(**leaves, positions=jnp.zeros((pool,), jnp.int32), metadata=cfg, layer_index=0)


def _serving_pools(case, mesh, pool_axis="dp"):
    """Place slots over the selected DP axis and heads over tp when distinct."""
    head_axis = "tp" if pool_axis == "dp" else None
    conv = NamedSharding(mesh, PartitionSpec(("fsdp", pool_axis), head_axis, None))
    rec = NamedSharding(mesh, PartitionSpec(("fsdp", pool_axis), head_axis, None, None))
    leaves = {
        name: jax.device_put(jnp.asarray(pool, jnp.float32), conv)
        for name, pool in zip(LEAVES[:3], case["conv"], strict=True)
    }
    leaves["recurrent_state"] = jax.device_put(jnp.asarray(case["rec"], jnp.float32), rec)
    return leaves


def _inputs(case, layout, slots_by_rank, explicit_slots):
    ranks = layout["ranks"]
    inputs = {
        "query": jnp.asarray(case["q"], jnp.float32),
        "key": jnp.asarray(case["k"], jnp.float32),
        "value": jnp.asarray(case["v"], jnp.float32),
        "beta": jnp.asarray(case["beta"], jnp.float32),
        "decay": jnp.asarray(case["decay"], jnp.float32),
        "conv_kernels": tuple(jnp.asarray(kernel, jnp.float32) for kernel in case["kernels"]),
        "query_start_loc": jnp.asarray(np.stack([_starts(lens) for lens, _, _ in ranks]), jnp.int32),
        # Flat [dp * rows], like eSurge's rank_context_lens.
        "context_lens": jnp.asarray([c for _, context, _ in ranks for c in context], jnp.int32),
    }
    if explicit_slots:
        inputs["state_indices"] = jnp.asarray(np.asarray(slots_by_rank, np.int32))
    return inputs


def _run(case, layout, mesh, *, explicit_slots, chunk_size, decay_lower_bound, pool_axis="dp"):
    metadata = OperationMetadata(runtime_dtype=jnp.float32, _stored_mesh=mesh)

    @jax.jit
    def step(inputs, leaves):
        out, view = kda_ragged_rows_rank_major_forward(
            **inputs,
            cache_view=_view(leaves),
            metadata=metadata,
            conv_dtype=jnp.float32,
            chunk_size=chunk_size,
            decay_lower_bound=decay_lower_bound,
            precision="highest",
        )
        return out, {name: getattr(view, name) for name in LEAVES}

    slots_by_rank = _slots(layout, explicit_slots)
    return step(_inputs(case, layout, slots_by_rank, explicit_slots), _serving_pools(case, mesh, pool_axis))


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _check(result, case, layout, slots_by_rank, *, tol):
    dp, tokens, slots_per_rank = _dims(layout)
    out, leaves = result
    out = np.asarray(out, np.float64)
    heads, _, hv = case["geom"]
    assert out.shape == (dp, tokens, heads, hv)
    ref_out, ref_conv, ref_rec = _reference(case, layout, slots_by_rank)
    owned = set()
    for d, (lens, _, _) in enumerate(layout["ranks"]):
        starts = _starts(lens)
        np.testing.assert_array_equal(out[d, starts[-1] :], 0.0, err_msg=f"rank {d}: bucket padding output")
        for r, n in enumerate(lens):
            if n:
                span = slice(starts[r], starts[r + 1])
                err = _rel(out[d, span], ref_out[d, span])
                assert err < tol, f"rank {d} row {r}: output relative error {err:.2e}"
                owned.add(d * slots_per_rank + slots_by_rank[d][r])
    # Conv windows hold raw float32 inputs: exact for owned slots, untouched elsewhere.
    for i, name in enumerate(LEAVES[:3]):
        np.testing.assert_array_equal(np.asarray(leaves[name], np.float64), ref_conv[i], err_msg=name)
    rec = np.asarray(leaves["recurrent_state"], np.float64)
    for slot in range(rec.shape[0]):
        if slot in owned:
            err = _rel(rec[slot], ref_rec[slot])
            assert err < tol, f"slot {slot}: recurrent relative error {err:.2e}"
        else:
            np.testing.assert_array_equal(rec[slot], case["rec"][slot], err_msg=f"untouched slot {slot}")


@pytest.mark.parametrize(
    ("layout", "dp", "tp", "chunk_size", "decay_lower_bound", "explicit_slots"),
    [
        pytest.param(DP2, 2, 2, 16, LOWER_BOUND, True, id="dp2-tp2-cs16-bound-slots"),
        pytest.param(DP2, 2, 2, 64, None, False, id="dp2-tp2-cs64-dense-identity"),
        pytest.param(DP2, 2, 1, 16, LOWER_BOUND, True, id="dp2-tp1-cs16-bound-slots"),
        pytest.param(DP4, 4, 1, 16, LOWER_BOUND, True, id="dp4-tp1-cs16-bound-slots"),
        pytest.param(DP4, 4, 1, 64, None, True, id="dp4-tp1-cs64-dense-slots"),
    ],
)
def test_rank_major_driver_matches_per_rank_reference(layout, dp, tp, chunk_size, decay_lower_bound, explicit_slots):
    mesh = _mesh(dp, tp)
    case = _case(layout, seed=dp * 10 + tp)
    result = _run(
        case,
        layout,
        mesh,
        explicit_slots=explicit_slots,
        chunk_size=chunk_size,
        decay_lower_bound=decay_lower_bound,
    )
    _check(result, case, layout, _slots(layout, explicit_slots), tol=1e-4)


@pytest.mark.parametrize(
    ("layout", "dp", "tp"),
    [
        pytest.param(DECODE_DP2, 2, 2, id="dp2-tp2"),
        pytest.param(DECODE_DP2, 2, 1, id="dp2-tp1"),
        pytest.param(DECODE_DP4, 4, 1, id="dp4-tp1"),
    ],
)
def test_rank_major_decode_shaped_steps_match_per_rank_reference(layout, dp, tp):
    """At most one token per row in every rank block: decode-only ranks and mixed ranks share one step."""
    mesh = _mesh(dp, tp)
    case = _case(layout, seed=100 + dp * 10 + tp, heads=DECODE_HEADS, head_dim=DECODE_HEAD_DIM)
    result = _run(case, layout, mesh, explicit_slots=True, chunk_size=64, decay_lower_bound=LOWER_BOUND)
    _check(result, case, layout, _slots(layout, True), tol=1e-4)


def test_rank_major_dp_over_tp_replicates_heads_and_matches_reference():
    """DP over tp must not emit duplicate tp axes in the KDA shard_map specs."""
    register_attention_data_parallel_axis("tp")
    mesh = _mesh(1, 2)
    case = _case(DP2, seed=77)
    result = _run(case, DP2, mesh, explicit_slots=True, chunk_size=64, decay_lower_bound=LOWER_BOUND, pool_axis="tp")
    _check(result, case, DP2, _slots(DP2, True), tol=1e-4)


def test_rank_major_driver_needs_the_rank_count_on_the_attention_dp_axis():
    """Rank-major metadata for two ranks on a mesh whose ``dp`` axis has one device is refused at trace time."""
    mesh = _mesh(1, 2)
    case = _case(DP2, seed=0)
    with pytest.raises(ValueError, match="attention data-parallel axis"):
        _run(case, DP2, mesh, explicit_slots=True, chunk_size=64, decay_lower_bound=None)


def test_rank_major_driver_refuses_single_rank_row_metadata():
    case = _case(DP2, seed=0)
    inputs = _inputs(case, DP2, _slots(DP2, True), True)
    inputs["query_start_loc"] = inputs["query_start_loc"][0]
    leaves = {name: jnp.asarray(pool, jnp.float32) for name, pool in zip(LEAVES[:3], case["conv"], strict=True)}
    leaves["recurrent_state"] = jnp.asarray(case["rec"], jnp.float32)
    metadata = OperationMetadata(runtime_dtype=jnp.float32, _stored_mesh=_mesh(2, 1))
    with pytest.raises(ValueError, match=r"rank-major query_start_loc \[dp, rows \+ 1\]"):
        kda_ragged_rows_rank_major_forward(**inputs, cache_view=_view(leaves), metadata=metadata)
