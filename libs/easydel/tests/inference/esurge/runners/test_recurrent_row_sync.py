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

"""Tests for plain-recurrent device row-sync on SequenceBuffer moves.

Covers the fix for the recurrent-decode row-instability bug: on the
plain-recurrent path the conv/GDR/SSM cache is indexed by physical
SequenceBuffer row, and ``condense`` / ``reorder_decode_first`` relocate a
surviving request without moving its device state. ``permute_recurrent_slots``
moves the device rows in lockstep. These tests exercise that permutation on
CPU (the end-to-end garbage is TPU-only, but the row bookkeeping is not).

Also covers KDACacheView (which extends BaseCacheView directly, not
RecurrentCacheView): its q/k/v conv windows + recurrent_state must be
zeroed on slot clear and follow row permutations, and its ``positions``
sidecar must obey the same sync contract as RecurrentCacheView.positions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.caching import HybridCache, KDACacheConfig, KDACacheView, RecurrentCacheView
from easydel.inference.esurge.runners.execution_manager import ExecutionManager
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax import PartitionAxis


def _make_view(n: int) -> RecurrentCacheView:
    """Build a RecurrentCacheView whose row ``i`` is filled with value ``i + 1``.

    Distinct per-row fill values make row moves directly observable.
    """
    # Ranks match the real GDR cache: conv_state is [slots, conv_dim, kernel]
    # and recurrent_state is [slots, v_heads, k_head_dim, v_head_dim]. The
    # extra recurrent rank matters — a 3-element PartitionSpec applied to a
    # rank-4 array is where sharding-preservation regressions actually show up.
    conv = jnp.stack([jnp.full((2, 3), float(i + 1)) for i in range(n)], axis=0)
    rec = jnp.stack([jnp.full((2, 2, 2), float(i + 1)) for i in range(n)], axis=0)
    return RecurrentCacheView(
        conv_state=conv,
        recurrent_state=rec,
        positions=None,
        metadata=None,
        layer_index=0,
    )


def _make_manager(n: int) -> ExecutionManager:
    """Construct a bare ExecutionManager carrying only the state permute needs."""
    mgr = ExecutionManager.__new__(ExecutionManager)
    mgr.max_num_reqs = n
    mgr.speculative_recurrent_state_tokens = 0
    mgr.kv_pages = HybridCache(views=[_make_view(n)])
    return mgr


def _row_values(mgr: ExecutionManager) -> list[float]:
    """Return the (constant) conv-state fill value per physical row."""
    conv = np.asarray(mgr.kv_pages.views[0].conv_state)
    return [float(conv[i, 0, 0]) for i in range(conv.shape[0])]


def _make_fsdp_sharded_manager(n: int) -> tuple[ExecutionManager, NamedSharding]:
    """Place recurrent leaves on the cache layout used by the Qwen AOT step."""
    if len(jax.devices()) < 4:
        pytest.skip("requires at least four fake or physical devices")
    mesh = Mesh(
        np.asarray(jax.devices()[:4], dtype=object).reshape((1, 1, 4, 1, 1, 1)),
        ("pp", "dp", "fsdp", "ep", "tp", "sp"),
    )
    cache_sharding = NamedSharding(
        mesh,
        PartitionSpec(("fsdp", "dp"), "tp", "sp"),
    )
    mgr = _make_manager(n)
    view = mgr.kv_pages.views[0]
    mgr.kv_pages = HybridCache(
        views=[
            view.replace(
                conv_state=jax.device_put(view.conv_state, cache_sharding),
                recurrent_state=jax.device_put(view.recurrent_state, cache_sharding),
            )
        ]
    )
    return mgr, cache_sharding


@pytest.mark.parametrize("operation", ["permute", "clear"])
def test_recurrent_row_transforms_preserve_aot_cache_shardings(operation: str) -> None:
    """Row maintenance must not invalidate the AOT model-step cache contract."""
    mgr, required_sharding = _make_fsdp_sharded_manager(4)

    if operation == "permute":
        mgr.permute_recurrent_slots(np.array([1, 0, 2, 3], dtype=np.int32))
    else:
        mgr.clear_recurrent_slots([1])

    view = mgr.kv_pages.views[0]
    assert view.conv_state.sharding == required_sharding
    assert view.recurrent_state.sharding == required_sharding

    # A lower().compile() executable checks shardings at call time instead of
    # silently inserting a reshard, matching the failing production path.
    compiled = (
        jax.jit(
            lambda conv_state, recurrent_state: (conv_state, recurrent_state),
            in_shardings=(required_sharding, required_sharding),
            out_shardings=(required_sharding, required_sharding),
        )
        .lower(view.conv_state, view.recurrent_state)
        .compile()
    )
    jax.block_until_ready(compiled(view.conv_state, view.recurrent_state))


def test_permute_condense_moves_survivor_and_zeroes_freed_row() -> None:
    """condense: middle request removed, last survivor moved into the hole."""
    mgr = _make_manager(4)
    # req at row1 finished; survivor at row3 moved to row1; row3 now free.
    # perm[t] = source row for destination t (-1 => zero).
    mgr.permute_recurrent_slots(np.array([0, 3, 2, -1], dtype=np.int32))

    values = _row_values(mgr)
    assert values[0] == 1.0  # row0 unchanged
    assert values[1] == 4.0  # row1 now holds the survivor formerly at row3
    assert values[2] == 3.0  # row2 unchanged
    assert values[3] == 0.0  # freed row zeroed

    # recurrent_state must move identically to conv_state.
    rec = np.asarray(mgr.kv_pages.views[0].recurrent_state)
    assert [float(rec[i, 0, 0, 0]) for i in range(4)] == [1.0, 4.0, 3.0, 0.0]


def test_permute_swap_exchanges_rows() -> None:
    """reorder_decode_first-style swap: rows 0 and 1 exchange state."""
    mgr = _make_manager(4)
    mgr.permute_recurrent_slots(np.array([1, 0, 2, 3], dtype=np.int32))
    assert _row_values(mgr) == [2.0, 1.0, 3.0, 4.0]


def test_permute_identity_is_noop() -> None:
    """An identity permutation leaves every row untouched."""
    mgr = _make_manager(4)
    before = _row_values(mgr)
    mgr.permute_recurrent_slots(np.array([0, 1, 2, 3], dtype=np.int32))
    assert _row_values(mgr) == before


def test_permute_multi_hole_condense() -> None:
    """Two removed rows: survivors compacted forward, tail rows zeroed."""
    mgr = _make_manager(5)
    # rows 0 and 2 removed; survivors 1,3,4 -> rows 0,1,2; rows 3,4 free.
    mgr.permute_recurrent_slots(np.array([1, 3, 4, -1, -1], dtype=np.int32))
    assert _row_values(mgr) == [2.0, 4.0, 5.0, 0.0, 0.0]


def test_permute_noop_on_non_hybrid_cache() -> None:
    """Permute must be a safe no-op when the cache is not a HybridCache."""
    mgr = ExecutionManager.__new__(ExecutionManager)
    mgr.max_num_reqs = 4
    mgr.speculative_recurrent_state_tokens = 0
    mgr.kv_pages = object()  # not a HybridCache
    mgr.permute_recurrent_slots(np.array([0, 3, 2, -1], dtype=np.int32))
    assert isinstance(mgr.kv_pages, object)  # unchanged, no crash


def test_permute_extends_to_speculative_candidate_rows() -> None:
    """Candidate rows (spec-decode) follow their owning base row.

    Layout is prefix-major: base rows occupy ``[0, n)`` and candidate prefix
    ``k`` for every base row occupies ``[n + k*n, n + (k+1)*n)``. Moving base
    row ``f`` to ``t`` must move every corresponding prefix row too.
    """
    n = 3
    cc = 2
    total = n + n * cc  # 3 base + 6 candidate rows
    mgr = ExecutionManager.__new__(ExecutionManager)
    mgr.max_num_reqs = n
    mgr.speculative_recurrent_state_tokens = cc
    conv = jnp.stack([jnp.full((2, 3), float(i + 1)) for i in range(total)], axis=0)
    rec = jnp.stack([jnp.full((2, 2), float(i + 1)) for i in range(total)], axis=0)
    mgr.kv_pages = HybridCache(
        views=[RecurrentCacheView(conv_state=conv, recurrent_state=rec, positions=None, metadata=None, layer_index=0)]
    )

    # Base row 2 -> row 0; base row 1 removed; base row 0 removed.
    # perm over base rows only: dest0 <- src2, dest1 -1, dest2 -1.
    mgr.permute_recurrent_slots(np.array([2, -1, -1], dtype=np.int32))

    values = _row_values(mgr)
    # Base rows.
    assert values[0] == 3.0  # base row0 now holds former base row2
    assert values[1] == 0.0
    assert values[2] == 0.0
    # Destination base row0 gets source base row2 in every candidate prefix.
    assert values[n + 0 * n + 0] == 6.0  # prefix0: source index n + 0*n + 2
    assert values[n + 1 * n + 0] == 9.0  # prefix1: source index n + 1*n + 2
    # Removed base rows' candidate rows are zeroed in every prefix.
    assert values[n + 0 * n + 1] == 0.0
    assert values[n + 1 * n + 2] == 0.0


# ---------------------------------------------------------------------------
# KDACacheView row sync (extends BaseCacheView directly, not RecurrentCacheView)
# ---------------------------------------------------------------------------

KDA_LEAF_NAMES = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
# Per-leaf fill offsets make row moves traceable leaf by leaf: leaf ``name``
# row ``i`` holds ``offset + i + 1``. ``positions[i] = 50 + i`` so the sidecar
# contract is observable separately from the state leaves.
KDA_LEAF_OFFSETS = {"q_conv_state": 1, "k_conv_state": 11, "v_conv_state": 21, "recurrent_state": 31}
KDA_LEAF_SHAPES = {"q_conv_state": (2, 3), "k_conv_state": (2, 3), "v_conv_state": (3, 3), "recurrent_state": (2, 2)}


def _make_kda_view(n: int) -> KDACacheView:
    """Build a KDACacheView with a distinct fill value per leaf and row."""
    shapes = KDA_LEAF_SHAPES

    def leaf(name: str) -> jax.Array:
        return jnp.stack([jnp.full(shapes[name], float(KDA_LEAF_OFFSETS[name] + i + 1)) for i in range(n)], axis=0)

    return KDACacheView(
        q_conv_state=leaf("q_conv_state"),
        k_conv_state=leaf("k_conv_state"),
        v_conv_state=leaf("v_conv_state"),
        recurrent_state=leaf("recurrent_state"),
        positions=jnp.asarray([50 + i for i in range(n)], dtype=jnp.int32),
        metadata=KDACacheConfig.create(
            num_hidden_layers=1,
            partition_axis=PartitionAxis(),
            batch_size=n,
            key_dim=2,
            value_dim=3,
            d_conv=3,
            recurrent_state_shape=(2, 2),
        ),
        layer_index=0,
    )


def _make_kda_manager(n: int) -> ExecutionManager:
    """Construct a bare ExecutionManager carrying a single-layer KDA cache."""
    mgr = ExecutionManager.__new__(ExecutionManager)
    mgr.max_num_reqs = n
    mgr.speculative_recurrent_state_tokens = 0
    mgr.kv_pages = HybridCache(views=[_make_kda_view(n)])
    return mgr


def _leaf_fill_values(view, name: str) -> list[float]:
    """Return the (constant) first-element fill value per physical row of a leaf."""
    arr = np.asarray(getattr(view, name))
    return [float(arr[i].reshape(-1)[0]) for i in range(arr.shape[0])]


def test_kda_clear_slot_zeroes_all_four_leaves_and_positions() -> None:
    """Slot clear must zero every KDA leaf of the freed slot only."""
    mgr = _make_kda_manager(4)
    mgr.clear_recurrent_slots([1])

    view = mgr.kv_pages.views[0]
    for name in KDA_LEAF_NAMES:
        off = KDA_LEAF_OFFSETS[name]
        assert _leaf_fill_values(view, name) == [off + 1, 0.0, off + 3, off + 4]
        # The whole freed row is zero, not just the first element.
        assert float(np.abs(np.asarray(getattr(view, name))[1]).sum()) == 0.0

    # positions is part of the sync contract: freed slot resets to 0.
    assert np.asarray(view.positions).tolist() == [50, 0, 52, 53]


def test_kda_permute_moves_all_four_leaves_and_positions() -> None:
    """Every KDA leaf and positions must follow the row permutation."""
    mgr = _make_kda_manager(4)
    # Row1 freed; survivor at row3 moved into row1; row3 zeroed.
    mgr.permute_recurrent_slots(np.array([0, 3, 2, -1], dtype=np.int32))

    view = mgr.kv_pages.views[0]
    for name in KDA_LEAF_NAMES:
        off = KDA_LEAF_OFFSETS[name]
        assert _leaf_fill_values(view, name) == [off + 1, off + 4, off + 3, 0.0]
        assert float(np.abs(np.asarray(getattr(view, name))[3]).sum()) == 0.0

    assert np.asarray(view.positions).tolist() == [50, 53, 52, 0]


def test_kda_permute_swap_exchanges_rows() -> None:
    """reorder_decode_first-style swap: rows 0 and 1 exchange KDA state."""
    mgr = _make_kda_manager(4)
    mgr.permute_recurrent_slots(np.array([1, 0, 2, 3], dtype=np.int32))

    view = mgr.kv_pages.views[0]
    for name in KDA_LEAF_NAMES:
        off = KDA_LEAF_OFFSETS[name]
        assert _leaf_fill_values(view, name) == [off + 2, off + 1, off + 3, off + 4]
    assert np.asarray(view.positions).tolist() == [51, 50, 52, 53]


def test_kda_permute_identity_is_noop() -> None:
    """An identity permutation leaves every KDA leaf and positions untouched."""
    mgr = _make_kda_manager(4)
    before = {name: _leaf_fill_values(mgr.kv_pages.views[0], name) for name in KDA_LEAF_NAMES}
    pos_before = np.asarray(mgr.kv_pages.views[0].positions).tolist()

    mgr.permute_recurrent_slots(np.array([0, 1, 2, 3], dtype=np.int32))

    assert {name: _leaf_fill_values(mgr.kv_pages.views[0], name) for name in KDA_LEAF_NAMES} == before
    assert np.asarray(mgr.kv_pages.views[0].positions).tolist() == pos_before


def test_kda_and_recurrent_views_clear_in_one_fused_pass() -> None:
    """A mixed hybrid cache clears the KDA view and the recurrent view together."""
    mgr = _make_kda_manager(4)
    mgr.kv_pages = HybridCache(views=[_make_view(4), _make_kda_view(4)])
    mgr.clear_recurrent_slots([2])

    # Recurrent view (row-indexed conv/recurrent leaves) still handled.
    assert _row_values(mgr) == [1.0, 2.0, 0.0, 4.0]
    # KDA view zeroed at the same slot, other rows untouched.
    kda = mgr.kv_pages.views[1]
    for name in KDA_LEAF_NAMES:
        off = KDA_LEAF_OFFSETS[name]
        assert _leaf_fill_values(kda, name) == [off + 1, off + 2, 0.0, off + 4]
    assert np.asarray(kda.positions).tolist() == [50, 51, 0, 53]


def test_kda_positions_sidecar_cleared_and_permuted() -> None:
    """positions obeys the same contract on both operations (dedicated check)."""
    mgr = _make_kda_manager(4)
    mgr.clear_recurrent_slots([0])
    assert np.asarray(mgr.kv_pages.views[0].positions).tolist() == [0, 51, 52, 53]

    # Fresh manager for the permute half: row3 -> row0, row0 freed -> zeroed.
    mgr = _make_kda_manager(4)
    mgr.permute_recurrent_slots(np.array([3, 1, 2, -1], dtype=np.int32))
    assert np.asarray(mgr.kv_pages.views[0].positions).tolist() == [53, 51, 52, 0]


def _make_fsdp_sharded_kda_manager(n: int) -> tuple[ExecutionManager, NamedSharding]:
    """Place the four KDA leaves on the cache layout used by the AOT step."""
    if len(jax.devices()) < 4:
        pytest.skip("requires at least four fake or physical devices")
    mesh = Mesh(
        np.asarray(jax.devices()[:4], dtype=object).reshape((1, 1, 4, 1, 1, 1)),
        ("pp", "dp", "fsdp", "ep", "tp", "sp"),
    )
    cache_sharding = NamedSharding(mesh, PartitionSpec(("fsdp", "dp"), "tp", "sp"))
    mgr = _make_kda_manager(n)
    view = mgr.kv_pages.views[0]
    mgr.kv_pages = HybridCache(
        views=[
            view.replace(
                **{name: jax.device_put(getattr(view, name), cache_sharding) for name in KDA_LEAF_NAMES},
            )
        ]
    )
    return mgr, cache_sharding


@pytest.mark.parametrize("operation", ["permute", "clear"])
def test_kda_row_transforms_preserve_aot_cache_shardings(operation: str) -> None:
    """KDA leaf maintenance must not invalidate the AOT model-step cache contract."""
    mgr, required_sharding = _make_fsdp_sharded_kda_manager(4)

    if operation == "permute":
        mgr.permute_recurrent_slots(np.array([1, 0, 2, 3], dtype=np.int32))
    else:
        mgr.clear_recurrent_slots([1])

    view = mgr.kv_pages.views[0]
    for name in KDA_LEAF_NAMES:
        assert getattr(view, name).sharding == required_sharding

    # A lower().compile() executable checks shardings at call time instead of
    # silently inserting a reshard, matching the failing production path.
    leaves = tuple(getattr(view, name) for name in KDA_LEAF_NAMES)
    compiled = (
        jax.jit(
            lambda q, k, v, rec: (q, k, v, rec),
            in_shardings=(required_sharding,) * 4,
            out_shardings=(required_sharding,) * 4,
        )
        .lower(*leaves)
        .compile()
    )
    jax.block_until_ready(compiled(*leaves))
