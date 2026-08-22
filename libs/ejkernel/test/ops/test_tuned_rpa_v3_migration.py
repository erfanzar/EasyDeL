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

"""ragged_page_attention_v3's tuned block sizes, after moving to the table.

The values used to be a ~4200-line ``TUNED_BLOCK_SIZES`` literal in
``_utils.py``; they are now rows in the shipped ``tuned_kernels.db``. All 3168
entries were verified to reproduce exactly at migration time, but that check
died with the literal, so these pin a spread of values read off the original
dict. If a regenerated table ever drops or changes them, this fails instead of
quietly handing attention a different tile.
"""

import pytest
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._utils import (
    TUNED_KERNEL_NAME,
    lookup_tuned_block_sizes,
)
from ejkernel.ops.tuned import TunedStore

BF16 = "q_bfloat16_kv_bfloat16"
FP8 = "q_bfloat16_kv_float8_e4m3fn"

# (device, page_size, dtypes, head_key, max_len) -> (bkv_pages, bq), read from
# the pre-migration literal.
KNOWN = {
    ("TPU v6e", 128, BF16, "q_head-128_kv_head-1_head-128", 1024): (8, 16),
    ("TPU v6e", 128, BF16, "q_head-128_kv_head-1_head-128", 2048): (8, 16),
    ("TPU v6e", 128, BF16, "q_head-128_kv_head-1_head-128", 256): (2, 8),
    ("TPU v6e", 128, BF16, "q_head-128_kv_head-1_head-128", 4096): (16, 8),
    ("TPU v6e", 128, BF16, "q_head-128_kv_head-1_head-128", 512): (2, 32),
    ("TPU v6e", 128, BF16, "q_head-128_kv_head-1_head-128", 8192): (16, 16),
    ("TPU v6e", 128, BF16, "q_head-128_kv_head-1_head-256", 2048): (16, 16),
    ("TPU v6e", 128, BF16, "q_head-128_kv_head-1_head-256", 512): (4, 8),
}


@pytest.mark.parametrize(("key", "expected"), sorted(KNOWN.items()))
def test_known_entries_survive_the_migration(key, expected):
    assert lookup_tuned_block_sizes(*key) == expected


def test_the_shipped_table_actually_contains_this_kernel():
    """Guards against shipping a wheel whose package data was dropped."""
    store = TunedStore()
    assert store.available(), f"no tuned table at {store.path}"
    assert TUNED_KERNEL_NAME in store.kernels()
    entries = store.entries(TUNED_KERNEL_NAME)
    assert len(entries) > 3000, f"expected the full migrated table, found {len(entries)} rows"


def test_head_dim_64_entries_are_present_and_reachable():
    """The h64 values were dead data before the migration.

    ``H64TUNED_BLOCK_SIZES`` was defined and never read by anything, and
    ``get_tuned_block_sizes_h64`` indexed the *main* table with ``head-64``
    keys, which that table has no entries for -- so head_dim=64 models always
    silently fell back to the TPU-version default. Both tables are now rows in
    one table keyed by head_dim, which makes those 156 entries reachable.
    """
    entries = [e for e in TunedStore().entries(TUNED_KERNEL_NAME) if "head_dim=64" in e.shape_key]
    assert entries, "head_dim=64 rows missing from the shipped table"
    hit = lookup_tuned_block_sizes(
        entries[0].device,
        int(dict(p.split("=") for p in entries[0].shape_key.split(","))["page_size"]),
        BF16,
        "q_head-{}_kv_head-{}_head-64".format(
            dict(p.split("=") for p in entries[0].shape_key.split(","))["q_heads"],
            dict(p.split("=") for p in entries[0].shape_key.split(","))["kv_heads"],
        ),
        int(dict(p.split("=") for p in entries[0].shape_key.split(","))["max_len"]),
    )
    assert hit is not None, "a head_dim=64 key should now resolve instead of falling through"


def test_unmeasured_combination_returns_none_so_the_caller_keeps_its_default():
    assert lookup_tuned_block_sizes("TPU v99", 128, BF16, "q_head-128_kv_head-1_head-128", 2048) is None
    assert lookup_tuned_block_sizes("TPU v6e", 128, "not a dtype key", "not a head key", 2048) is None


def test_every_row_for_this_kernel_is_well_formed():
    """A regenerated table must not ship rows the reader cannot use."""
    for entry in TunedStore().entries(TUNED_KERNEL_NAME):
        assert entry.platform == "pallas"
        assert set(entry.config) == {"num_kv_pages_per_block", "num_queries_per_block"}
        assert all(isinstance(v, int) and v > 0 for v in entry.config.values())
        assert entry.device.startswith("TPU ")
        assert "head_dim=" in entry.shape_key and "max_len=" in entry.shape_key


def test_fp8_kv_entries_are_kept_distinct_from_bf16():
    """dtype is part of the key; collapsing it would hand fp8 bf16's tiles."""
    bf16 = lookup_tuned_block_sizes("TPU v6e", 128, BF16, "q_head-128_kv_head-1_head-128", 2048)
    fp8 = lookup_tuned_block_sizes("TPU v6e", 128, FP8, "q_head-128_kv_head-1_head-128", 2048)
    assert bf16 is not None
    if fp8 is not None:
        store = TunedStore()
        keys = {e.dtypes for e in store.entries(TUNED_KERNEL_NAME)}
        assert len(keys) > 1, "fp8 and bf16 rows collapsed into one dtype key"
