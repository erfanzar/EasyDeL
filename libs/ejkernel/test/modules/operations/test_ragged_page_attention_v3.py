from __future__ import annotations

import importlib.util

import jax.numpy as jnp
import pytest
from ejkernel.modules.operations import ragged_page_attention_v3
from ejkernel.utils import make_dummy_rpa_inputs

from ._utils import assert_allclose, device_platform

_HAS_CUTLASS = importlib.util.find_spec("cutlass") is not None


def test_ragged_page_attention_v3_shapes_and_attention_sink_runs():
    batch = make_dummy_rpa_inputs(
        rng_seed=0,
        num_seqs=2,
        pages_per_seq=2,
        page_size=16,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        kv_dtype=jnp.bfloat16,
        total_q=8,
    )

    kv_cache0 = batch["kv_cache"].copy()
    out, kv_cache_out = ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache0,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        None,
        softmax_scale=128**-0.5,
        platform="xla",
    )
    assert out.shape == batch["queries"].shape
    assert kv_cache_out.shape == batch["kv_cache"].shape

    sinks = jnp.zeros((batch["queries"].shape[1],), dtype=jnp.float32)
    out2, kv_cache_out2 = ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache_out,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=128**-0.5,
        sliding_window=16,
        logits_soft_cap=10.0,
        platform="xla",
    )
    assert out2.shape == batch["queries"].shape
    assert kv_cache_out2.shape == batch["kv_cache"].shape


def test_make_dummy_rpa_inputs_supports_padded_page_tables_with_smaller_physical_cache():
    num_seqs = 2
    pages_per_seq = 4  # padded capacity
    total_num_pages = 3  # smaller than num_seqs*pages_per_seq (=8)
    batch = make_dummy_rpa_inputs(
        rng_seed=0,
        num_seqs=num_seqs,
        pages_per_seq=pages_per_seq,
        page_size=16,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        kv_dtype=jnp.bfloat16,
        total_q=4,
        total_num_pages=total_num_pages,
    )

    assert batch["block_tables"].shape == (num_seqs * pages_per_seq,)
    assert batch["kv_cache"].shape[0] == total_num_pages
    assert int(batch["block_tables"].min()) >= 0
    assert int(batch["block_tables"].max()) < total_num_pages

    kv_cache0 = batch["kv_cache"].copy()
    out, kv_cache_out = ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache0,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        None,
        softmax_scale=128**-0.5,
        platform="xla",
    )
    assert out.shape == batch["queries"].shape
    assert kv_cache_out.shape == batch["kv_cache"].shape


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only cross-backend comparison (pallas vs xla)")
def test_ragged_page_attention_v3_pallas_matches_xla_on_tpu():
    batch = make_dummy_rpa_inputs(
        rng_seed=1,
        num_seqs=2,
        pages_per_seq=2,
        page_size=16,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        kv_dtype=jnp.bfloat16,
        total_q=8,
    )
    sinks = jnp.zeros((batch["queries"].shape[1],), dtype=jnp.float32)
    kv_cache_xla = batch["kv_cache"].copy()
    kv_cache_pallas = batch["kv_cache"].copy()

    out_xla, cache_xla = ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache_xla,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=128**-0.5,
        sliding_window=16,
        logits_soft_cap=10.0,
        platform="xla",
    )
    out_pallas, cache_pallas = ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache_pallas,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=128**-0.5,
        sliding_window=16,
        logits_soft_cap=10.0,
        platform="pallas",
    )

    assert_allclose(out_pallas, out_xla, atol=0.35)
    assert_allclose(cache_pallas, cache_xla, atol=0.35)


def test_rpa_v3_key_builder_distinguishes_shard_map_layouts():
    """Identical global shapes with different shard_map mesh/in_specs must get distinct cache keys."""
    from types import SimpleNamespace

    from ejkernel.modules.operations.ragged_page_attention_v3 import RaggedPageAttentionv3
    from ejkernel.ops import Invocation
    from jax.sharding import PartitionSpec as P

    batch = make_dummy_rpa_inputs(
        rng_seed=2,
        num_seqs=2,
        pages_per_seq=4,
        page_size=16,
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=128,
        kv_dtype=jnp.bfloat16,
        total_q=16,
    )
    args = (
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        None,
    )
    kernel = RaggedPageAttentionv3()

    inv_tp4 = Invocation(
        op_id=kernel.op_id,
        args=args,
        kwargs={},
        method="shard_map",
        mesh=SimpleNamespace(shape={"tp": 4}),
        in_specs=(P(None, "tp", None), P(None, "tp", None), P(None, "tp", None), P(None, None, "tp", None, None)),
        out_specs=(P(None, "tp", None), P(None, None, "tp", None, None)),
    )
    inv_tp2 = Invocation(
        op_id=kernel.op_id,
        args=args,
        kwargs={},
        method="shard_map",
        mesh=SimpleNamespace(shape={"tp": 2}),
        in_specs=(P(None, "tp", None), P(None, "tp", None), P(None, "tp", None), P(None, None, "tp", None, None)),
        out_specs=(P(None, "tp", None), P(None, None, "tp", None, None)),
    )
    inv_dp4 = Invocation(
        op_id=kernel.op_id,
        args=args,
        kwargs={},
        method="shard_map",
        mesh=SimpleNamespace(shape={"dp": 4}),
        in_specs=(P("dp", None, None), P("dp", None, None), P("dp", None, None), P(None, None, None, None, None)),
        out_specs=(P("dp", None, None), P(None, None, None, None, None)),
    )

    k_tp4 = inv_tp4.make_key(kernel.key_builder)
    k_tp2 = inv_tp2.make_key(kernel.key_builder)
    k_dp4 = inv_dp4.make_key(kernel.key_builder)
    assert len({k_tp4, k_tp2, k_dp4}) == 3


def test_rpa_v3_tpu_heuristic_and_vmem_filtering(monkeypatch):
    """Verify head_dim=64, 16MiB VMEM filtering, 32MiB bq=128, and decode max_num_tokens=1 heuristics."""
    import importlib
    from types import SimpleNamespace

    from ejkernel.ops import Invocation

    rpa_mod = importlib.import_module("ejkernel.modules.operations.ragged_page_attention_v3")
    kernel = rpa_mod.RaggedPageAttentionv3()
    monkeypatch.setattr(rpa_mod, "_is_tpu", lambda: True)
    monkeypatch.setattr(rpa_mod, "_lookup_exact_tuned_pair", lambda workload: None)

    def _make_inv(*, total_q, num_q_heads, num_kv_heads, head_dim, page_size, pages_per_seq):
        q = SimpleNamespace(shape=(total_q, num_q_heads, head_dim), dtype=jnp.bfloat16)
        k = SimpleNamespace(shape=(total_q, num_kv_heads, head_dim), dtype=jnp.bfloat16)
        v = SimpleNamespace(shape=(total_q, num_kv_heads, head_dim), dtype=jnp.bfloat16)
        kv_cache = SimpleNamespace(shape=(pages_per_seq, page_size, num_kv_heads, 2, head_dim), dtype=jnp.bfloat16)
        kv_lens = SimpleNamespace(shape=(1,), dtype=jnp.int32)
        block_tables = SimpleNamespace(shape=(pages_per_seq,), dtype=jnp.int32)
        q_start = SimpleNamespace(shape=(2,), dtype=jnp.int32)
        dist = SimpleNamespace(shape=(3,), dtype=jnp.int32)
        return Invocation(
            op_id=kernel.op_id,
            args=(q, k, v, kv_cache, kv_lens, block_tables, q_start, dist, None),
            kwargs={},
        )

    # 1. head_dim=64 must not pick bq=64 first (bq <= 32).
    monkeypatch.setattr(rpa_mod, "_tpu_vmem_capacity_bytes", lambda: 16 << 20)
    inv_h64 = _make_inv(total_q=64, num_q_heads=16, num_kv_heads=4, head_dim=64, page_size=16, pages_per_seq=32)
    cfg_h64 = kernel.heuristic_cfg(inv_h64)
    assert cfg_h64.num_queries_per_block is not None and cfg_h64.num_queries_per_block <= 32

    # 2. High-head-count head_dim=128 on 16 MiB VMEM must filter out (8, 64) and pick a VMEM-fitting pair.
    inv_high = _make_inv(total_q=256, num_q_heads=64, num_kv_heads=8, head_dim=128, page_size=64, pages_per_seq=64)
    cfg_high = kernel.heuristic_cfg(inv_high)
    wl_high = kernel._extract_workload(inv_high)
    assert wl_high is not None
    assert cfg_high.num_kv_pages_per_block == 8
    assert cfg_high.num_queries_per_block is not None and cfg_high.num_queries_per_block < 64
    est_high = rpa_mod._estimate_tpu_vmem_bytes(wl_high, cfg_high.num_kv_pages_per_block, cfg_high.num_queries_per_block)
    assert est_high <= (12 << 20)

    # 3. Compact-head head_dim=128 on 32 MiB VMEM with max_num_tokens=256 still picks (8, 128).
    monkeypatch.setattr(rpa_mod, "_tpu_vmem_capacity_bytes", lambda: 32 << 20)
    inv_compact = _make_inv(total_q=256, num_q_heads=8, num_kv_heads=2, head_dim=128, page_size=16, pages_per_seq=64)
    cfg_compact = kernel.heuristic_cfg(inv_compact)
    assert cfg_compact.num_kv_pages_per_block == 8
    assert cfg_compact.num_queries_per_block == 128

    # 4. Single-token decode (max_num_tokens == 1) leaves num_kv_pages_per_block=None for Pallas decode boost.
    inv_dec = _make_inv(total_q=1, num_q_heads=8, num_kv_heads=2, head_dim=128, page_size=16, pages_per_seq=64)
    cfg_dec = kernel.heuristic_cfg(inv_dec)
    assert cfg_dec.num_kv_pages_per_block is None
    assert cfg_dec.num_queries_per_block == 1
