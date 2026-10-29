# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OCDBT checkpoint layout: roundtrips, layout auto-detection, write transaction, merge ordering."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_ser
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax.serialization import AsyncCheckpointManager, Checkpointer, OcdbtSaveOptions, _ocdbt
from spectrax.serialization.checkpointer import _rm_checkpoint
from spectrax.serialization.serialization import tree_deserialize_leaves, tree_serialize_leaves

pytestmark = pytest.mark.skipif(len(jax.devices()) < 4, reason="needs a 4-device mesh")


def _file_count(root) -> int:
    return sum(len(files) for _, _, files in os.walk(root))


@pytest.fixture
def mesh4():
    return Mesh(np.array(jax.devices()[:4]), ("fsdp",))


@pytest.fixture
def sharded_tree(mesh4):
    sharding = NamedSharding(mesh4, PartitionSpec("fsdp"))
    rng = np.random.default_rng(0)
    layers = {
        str(i): {"weight": jax.device_put(jnp.asarray(rng.standard_normal((16, 8)), jnp.float32), sharding)}
        for i in range(6)
    }
    return {
        "layers": layers,
        "emb": jax.device_put(jnp.asarray(rng.standard_normal((32, 4)), jnp.bfloat16), sharding),
        "ids": jax.device_put(jnp.arange(8, dtype=jnp.int32), sharding),
        "step": 7,
    }


def _rules(mesh4):
    return [(".*", NamedSharding(mesh4, PartitionSpec("fsdp")))]


def _assert_tree_equal(loaded, expected):
    for got, want in zip(jax.tree.leaves(loaded), jax.tree.leaves(expected), strict=True):
        if hasattr(want, "dtype"):
            assert got.dtype == want.dtype
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        else:
            assert got == want


@pytest.fixture
def async_serialize_spy(monkeypatch):
    calls: list[tuple[dict, dict]] = []
    real = ts_impl.async_serialize

    async def spy(arr, spec, **kwargs):
        calls.append((copy.deepcopy(spec), dict(kwargs)))
        return await real(arr, spec, **kwargs)

    monkeypatch.setattr(ts_impl, "async_serialize", spy)
    return calls


class TestRoundtrip:
    @pytest.mark.parametrize("ocdbt", [False, True])
    def test_sharded_roundtrip_is_bit_exact(self, tmp_path, mesh4, sharded_tree, ocdbt):
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(sharded_tree, str(tmp_path), prefix="model", ocdbt=ocdbt)

        loaded, _ = mgr.load_pytree(str(tmp_path), mesh4, prefix="model", sharding_rules=_rules(mesh4))

        _assert_tree_equal(loaded, sharded_tree)
        assert loaded["layers"]["3"]["weight"].sharding.spec == PartitionSpec("fsdp")
        index = json.loads((tmp_path / "tensorstore_index.json").read_text())
        assert _ocdbt.backend_for_prefix(index, "model") == ("ocdbt" if ocdbt else "tensorstore")
        assert (tmp_path / "model" / "emb").exists() is not ocdbt
        assert _ocdbt.manifest_exists(str(tmp_path), "model") is ocdbt

    @pytest.mark.parametrize("chunk_size", [None, 2])
    @pytest.mark.parametrize("index_only", [False, True])
    def test_ocdbt_load_variants(self, tmp_path, mesh4, sharded_tree, chunk_size, index_only):
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(sharded_tree, str(tmp_path), prefix="model", ocdbt=True)
        if index_only:
            (tmp_path / "model_structure.json").unlink()

        loaded, _ = mgr.load_pytree(
            str(tmp_path),
            mesh4,
            prefix="model",
            sharding_rules=_rules(mesh4),
            chunk_size=chunk_size,
            can_skip_structure=index_only,
            tensorstore_assume_metadata=True,
        )

        np.testing.assert_array_equal(np.asarray(loaded["emb"]), np.asarray(sharded_tree["emb"]))
        np.testing.assert_array_equal(
            np.asarray(loaded["layers"]["5"]["weight"]), np.asarray(sharded_tree["layers"]["5"]["weight"])
        )

    def test_tree_serialize_leaves_roundtrip(self, tmp_path, mesh4, sharded_tree):
        arrays_only = {k: v for k, v in sharded_tree.items() if k != "step"}
        tree_serialize_leaves(str(tmp_path), arrays_only, prefix="model", ocdbt=True)

        for chunk_size in (None, 3):
            loaded = tree_deserialize_leaves(
                str(tmp_path), mesh4, prefix="model", sharding_rules=_rules(mesh4), chunk_size=chunk_size
            )
            _assert_tree_equal(loaded, arrays_only)

    def test_resave_replaces_store(self, tmp_path, mesh4):
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"w": jnp.ones((8,)), "old": jnp.ones((4,))}, str(tmp_path), prefix="model", ocdbt=True)
        mgr.save_pytree({"w": jnp.full((8,), 3.0)}, str(tmp_path), prefix="model", ocdbt=True)

        loaded, _ = mgr.load_pytree(str(tmp_path), mesh4, prefix="model")
        assert set(loaded) == {"w"}
        np.testing.assert_array_equal(np.asarray(loaded["w"]), np.full((8,), 3.0, np.float32))

    def test_missing_store_raises(self, tmp_path, mesh4):
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"w": jnp.ones((8,))}, str(tmp_path), prefix="model", ocdbt=True)
        _rm_checkpoint(_ocdbt.store_dir(str(tmp_path), "model"))

        with pytest.raises(FileNotFoundError, match="OCDBT store"):
            mgr.load_pytree(str(tmp_path), mesh4, prefix="model")


class TestMixedLayouts:
    def test_zarr_and_ocdbt_prefixes_in_one_checkpoint(self, tmp_path, mesh4, sharded_tree):
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(sharded_tree, str(tmp_path), prefix="model", ocdbt=False)
        opt = {"mu": jnp.arange(16, dtype=jnp.float32), "count": jnp.asarray(3, jnp.int32)}
        mgr.save_pytree(opt, str(tmp_path), prefix="tx", ocdbt=True)

        index = json.loads((tmp_path / "tensorstore_index.json").read_text())
        assert _ocdbt.backend_for_prefix(index, "model") == "tensorstore"
        assert _ocdbt.backend_for_prefix(index, "tx") == "ocdbt"

        model, _ = mgr.load_pytree(str(tmp_path), mesh4, prefix="model", sharding_rules=_rules(mesh4))
        tx, _ = mgr.load_pytree(str(tmp_path), mesh4, prefix="tx")
        _assert_tree_equal(model, sharded_tree)
        _assert_tree_equal(tx, opt)

    def test_switching_a_prefix_back_to_zarr_clears_the_backend(self, tmp_path, mesh4):
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"w": jnp.ones((8,))}, str(tmp_path), prefix="model", ocdbt=True)
        mgr.save_pytree({"w": jnp.full((8,), 2.0)}, str(tmp_path), prefix="model", ocdbt=False)

        index = json.loads((tmp_path / "tensorstore_index.json").read_text())
        assert "backends" not in index
        loaded, _ = mgr.load_pytree(str(tmp_path), mesh4, prefix="model")
        np.testing.assert_array_equal(np.asarray(loaded["w"]), np.full((8,), 2.0, np.float32))

    def test_legacy_index_without_backends_reads_as_zarr(self, tmp_path, mesh4, sharded_tree):
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(sharded_tree, str(tmp_path), prefix="model")
        index = json.loads((tmp_path / "tensorstore_index.json").read_text())
        assert "backends" not in index

        loaded, _ = mgr.load_pytree(str(tmp_path), mesh4, prefix="model", sharding_rules=_rules(mesh4))
        _assert_tree_equal(loaded, sharded_tree)


class TestObjectCount:
    def test_ocdbt_writes_far_fewer_objects(self, tmp_path, mesh4):
        sharding = NamedSharding(mesh4, PartitionSpec("fsdp"))
        tree = {str(i): jax.device_put(jnp.full((64, 16), float(i)), sharding) for i in range(24)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, str(tmp_path / "zarr"), prefix="model", ocdbt=False)
        mgr.save_pytree(tree, str(tmp_path / "ocdbt"), prefix="model", ocdbt=True)

        zarr_count = _file_count(tmp_path / "zarr")
        ocdbt_count = _file_count(tmp_path / "ocdbt")
        assert ocdbt_count * 4 < zarr_count, (zarr_count, ocdbt_count)


def _dtype_tree():
    value = 1.0001
    return {
        "router": {"weight": jnp.full((4,), value, jnp.float32)},
        "mlp": {"weight": jnp.full((4,), value, jnp.float32)},
        "ids": jnp.arange(4, dtype=jnp.int32),
    }


def _pick(key: str):
    return jnp.float32 if key.split(".")[-2:] == ["router", "weight"] else jnp.bfloat16


@pytest.mark.parametrize("chunk_size", [None, 1])
@pytest.mark.parametrize("index_only", [False, True])
def test_per_key_dtype_on_ocdbt(tmp_path, mesh4, chunk_size, index_only):
    mgr = AsyncCheckpointManager()
    mgr.save_pytree(_dtype_tree(), str(tmp_path), prefix="model", ocdbt=True)
    if index_only:
        (tmp_path / "model_structure.json").unlink()

    loaded, _ = mgr.load_pytree(
        str(tmp_path), mesh4, prefix="model", dtype=_pick, chunk_size=chunk_size, can_skip_structure=index_only
    )

    assert loaded["router"]["weight"].dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(loaded["router"]["weight"]), np.full((4,), 1.0001, np.float32))
    assert loaded["mlp"]["weight"].dtype == jnp.bfloat16
    assert loaded["ids"].dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(loaded["ids"]), np.arange(4))


class TestRunningEventLoop:
    def test_run_sync_inside_a_running_loop(self):
        async def value():
            return 41 + 1

        async def outer():
            return _ocdbt.run_sync(value)

        assert asyncio.run(outer()) == 42

    def test_save_and_load_from_a_running_loop(self, tmp_path, mesh4, sharded_tree):
        async def roundtrip():
            mgr = AsyncCheckpointManager()
            mgr.save_pytree(sharded_tree, str(tmp_path), prefix="model", ocdbt=True)
            return mgr.load_pytree(
                str(tmp_path), mesh4, prefix="model", sharding_rules=_rules(mesh4), show_progress=False
            )[0]

        _assert_tree_equal(asyncio.run(roundtrip()), sharded_tree)


class TestWriteTransaction:
    def test_one_atomic_committed_transaction_and_shared_context(
        self, tmp_path, mesh4, sharded_tree, async_serialize_spy
    ):
        AsyncCheckpointManager().save_pytree(sharded_tree, str(tmp_path), prefix="model", ocdbt=True)

        assert len(async_serialize_spy) == 8
        transactions = {id(kwargs["transaction"]) for _, kwargs in async_serialize_spy}
        contexts = {id(kwargs["context"]) for _, kwargs in async_serialize_spy}
        assert len(transactions) == 1 and len(contexts) == 1
        transaction = async_serialize_spy[0][1]["transaction"]
        assert transaction.atomic and transaction.commit_started and not transaction.aborted
        assert async_serialize_spy[0][1]["context"] is _ocdbt.default_context()
        for spec, _ in async_serialize_spy:
            assert spec["kvstore"]["driver"] == "ocdbt"
            assert spec["kvstore"]["target_data_file_size"] == 64 * 1024 * 1024
            assert "context" not in spec

    def test_options_reach_specs_and_keep_inherited_resources(self, tmp_path, mesh4, sharded_tree, async_serialize_spy):
        options = OcdbtSaveOptions(data_file_mb=16, io_concurrency=8, gcs_concurrency=64)
        ckpt = Checkpointer(str(tmp_path), save_interval=None, step_policies=[], ocdbt=True, ocdbt_options=options)
        ckpt.save_pytree(sharded_tree, prefix="model")

        context = async_serialize_spy[0][1]["context"]
        assert all(kwargs["context"] is context for _, kwargs in async_serialize_spy)
        assert all(spec["kvstore"]["target_data_file_size"] == 16 * 1024 * 1024 for spec, _ in async_serialize_spy)
        assert context["file_io_concurrency"].to_json() == {"limit": 8}
        assert context["gcs_request_concurrency"].to_json() == {"limit": 64}
        inherited = _ocdbt.default_context()
        for resource in ("data_copy_concurrency", "cache_pool", "cache_pool#remote"):
            assert context[resource].to_json() == inherited[resource].to_json(), resource

        loaded, _ = ckpt.load_pytree(mesh4, prefix="model", path=str(tmp_path), discover_latest=False)
        _assert_tree_equal(loaded, sharded_tree)

    def test_zarr_layout_takes_no_transaction(self, tmp_path, sharded_tree, async_serialize_spy):
        AsyncCheckpointManager().save_pytree(sharded_tree, str(tmp_path), prefix="model")
        assert async_serialize_spy
        for spec, kwargs in async_serialize_spy:
            assert kwargs.get("transaction") is None
            assert spec["kvstore"]["driver"] != "ocdbt"

    def test_failed_write_aborts(self, tmp_path, monkeypatch):
        seen = []

        async def boom(arr, spec, **kwargs):
            seen.append(kwargs["transaction"])
            raise RuntimeError("write failed")

        monkeypatch.setattr(ts_impl, "async_serialize", boom)
        specs = _ocdbt.write_specs(str(tmp_path), "model", ["model/w"])
        with pytest.raises(RuntimeError, match="write failed"):
            _ocdbt.serialize_arrays([jnp.ones((4,))], specs)
        assert seen and seen[0].aborted

    @pytest.mark.parametrize("kwargs", [{"data_file_mb": 0}, {"io_concurrency": -1}, {"gcs_concurrency": True}])
    def test_options_reject_non_positive(self, kwargs):
        with pytest.raises(ValueError, match="positive int"):
            OcdbtSaveOptions(**kwargs)


class TestMultiProcessMerge:
    def test_staged_stores_merge_and_read_back(self, tmp_path, mesh4):
        root = str(tmp_path)
        sharding = NamedSharding(mesh4, PartitionSpec("fsdp"))
        expected = {}
        for process_idx in range(2):
            relpaths = [f"model/p{process_idx}/w{i}" for i in range(3)]
            arrays = [jax.device_put(jnp.full((8,), float(process_idx * 10 + i)), sharding) for i in range(3)]
            expected.update(zip(relpaths, arrays, strict=True))
            specs = _ocdbt.write_specs(root, "model", relpaths, process_idx=process_idx, process_count=2)
            _ocdbt.serialize_arrays(arrays, specs)

        _ocdbt.finalize(root, "model", process_count=2)

        manager = array_ser.GlobalAsyncCheckpointManager(timeout_secs=300)
        specs = [_ocdbt.read_spec(root, "model", rel) for rel in expected]
        loaded = manager.deserialize([sharding] * len(specs), specs)
        for (rel, original), restored in zip(expected.items(), loaded, strict=True):
            np.testing.assert_array_equal(np.asarray(restored), np.asarray(original), err_msg=rel)

        stage = Path(_ocdbt.stage_dir(root, "model", 0))
        assert (stage / "manifest.ocdbt").exists()
        _rm_checkpoint(root)
        assert not Path(root).exists()

    def test_finalize_is_noop_single_process(self, tmp_path):
        _ocdbt.finalize(str(tmp_path), "model", process_count=1)
        assert not Path(_ocdbt.store_dir(str(tmp_path), "model")).exists()

    def test_merge_runs_between_commit_and_merged_barriers(self, tmp_path, sharded_tree, monkeypatch):
        manager = array_ser.GlobalAsyncCheckpointManager(timeout_secs=300)
        events: list[str] = []

        async def write(arr, spec, **kwargs):
            events.append("write")

        monkeypatch.setattr(ts_impl, "async_serialize", write)
        monkeypatch.setattr(jax, "process_count", lambda: 2)
        monkeypatch.setattr(multihost_utils, "sync_global_devices", lambda name: events.append(name.split(":")[2]))
        monkeypatch.setattr(_ocdbt, "finalize", lambda *a, **k: events.append("merge"))

        arrays_only = {k: v for k, v in sharded_tree.items() if k != "step"}
        tree_serialize_leaves(
            str(tmp_path),
            arrays_only,
            manager=manager,
            prefix="model",
            ocdbt=True,
            commit_callback=lambda: events.append("callback"),
        )

        first_write = events.index("write")
        last_write = len(events) - 1 - events[::-1].index("write")
        assert events.index("cleared") < first_write
        assert events[last_write + 1 :] == ["committed", "merge", "merged", "callback"]


def test_checkpointer_gc_removes_ocdbt_checkpoint(tmp_path, mesh4, sharded_tree):
    ckpt = Checkpointer(str(tmp_path), save_interval=None, step_policies=[], ocdbt=True)
    path = ckpt.save_pytree(sharded_tree, prefix="model", step=3)
    assert _ocdbt.manifest_exists(path, "model")

    loaded, meta = ckpt.load_pytree(mesh4, prefix="model", sharding_rules=_rules(mesh4))
    _assert_tree_equal(loaded, sharded_tree)
    assert meta["step"] == 3

    ckpt._queue_checkpoint_removal(path)
    deadline = time.monotonic() + 30
    while Path(path).exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not Path(path).exists()
