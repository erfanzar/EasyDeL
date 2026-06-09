# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for AsyncCheckpointManager (TensorStore-only)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from spectrax.serialization import AsyncCheckpointManager
from spectrax.serialization.async_manager import _tensorstore_spec_for_load


class TestAsyncCheckpointManager:
    """End-to-end tests for AsyncCheckpointManager save/load."""

    def test_load_pytree_template_uses_caller_key_aliases(self, tmp_checkpoint_dir):
        """Template restore may alias keys and resolve shardings from the template layout."""
        mesh = Mesh(np.array(jax.devices()[:1]), ("x",))
        tree = {"old": jnp.arange(4, dtype=jnp.float32)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        template = {"new": jnp.zeros((4,), dtype=jnp.float32)}
        sharding = NamedSharding(mesh, PartitionSpec("x"))
        loaded, _ = mgr.load_pytree(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            template=template,
            key_aliases=lambda key: ("model.old",) if key == "model.new" else (),
            sharding_rules=[("model/new", sharding)],
        )
        assert jnp.allclose(loaded["new"], tree["old"])
        assert loaded["new"].sharding == sharding

    def test_save_load_pytree_roundtrip(self, tmp_checkpoint_dir, mesh, sample_pytree):
        """save_pytree -> load_pytree preserves values and structure."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(sample_pytree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        loaded, meta = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model")

        assert jnp.allclose(loaded["layer0"]["weight"], sample_pytree["layer0"]["weight"])
        assert jnp.allclose(loaded["layer0"]["bias"], sample_pytree["layer0"]["bias"])
        assert loaded["step"] == sample_pytree["step"]
        assert loaded["name"] == sample_pytree["name"]
        assert isinstance(meta, dict)

    def test_nonarray_payload_preserved(self, tmp_checkpoint_dir, mesh):
        """Ints, strings, and None survive the roundtrip."""
        tree = {"a": None, "b": 7, "c": "hello", "arr": jnp.ones((2, 2))}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="tx")

        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="tx")
        assert loaded["a"] is None
        assert loaded["b"] == 7
        assert loaded["c"] == "hello"
        assert jnp.allclose(loaded["arr"], tree["arr"])

    def test_load_pytree_with_template(self, tmp_checkpoint_dir, mesh):
        """Loading into a template coerces shapes when strict_shapes=False."""
        tree = {"x": jnp.arange(6).reshape(2, 3)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        template = {"x": jnp.zeros((2, 3))}
        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model", template=template, strict_shapes=True)
        assert jnp.allclose(loaded["x"], tree["x"])

    def test_sharding_rules_matching(self, tmp_checkpoint_dir, mesh):
        """sharding_rules assigns NamedShardings by regex match."""
        sh = NamedSharding(mesh, PartitionSpec("x", "y"))
        tree = {
            "layers": {
                "0": {"w": jnp.arange(8).reshape(2, 4), "b": jnp.ones(4)},
                "1": {"w": jnp.arange(8).reshape(2, 4), "b": jnp.ones(4)},
            }
        }
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        loaded, _ = mgr.load_pytree(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            sharding_rules=[(".*weight.*|.*w.*", sh)],
        )
        assert jnp.allclose(loaded["layers"]["0"]["w"], tree["layers"]["0"]["w"])
        assert jnp.allclose(loaded["layers"]["0"]["b"], tree["layers"]["0"]["b"])

    def test_chunked_load(self, tmp_checkpoint_dir, mesh):
        """chunk_size loads arrays in batches without error."""
        tree = {f"p{i}": jnp.ones((4, 4)) for i in range(5)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model", chunk_size=2)
        assert set(loaded.keys()) == set(tree.keys())
        for k in tree:
            assert jnp.allclose(loaded[k], tree[k])

    def test_fast_tensorstore_load_options(self, tmp_checkpoint_dir, mesh):
        """TensorStore fast-load options deserialize through the custom path."""
        tree = {f"p{i}": jnp.arange(8, dtype=jnp.float32).reshape(2, 4) + i for i in range(3)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        loaded, _ = mgr.load_pytree(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            concurrent_gb=1,
            tensorstore_io_concurrency=4,
            tensorstore_copy_concurrency=4,
            tensorstore_cache_gb=1,
            tensorstore_assume_metadata=True,
            tensorstore_metadata_workers=2,
            show_progress=False,
            progress_every=2,
        )

        assert set(loaded.keys()) == set(tree.keys())
        for key in tree:
            assert jnp.allclose(loaded[key], tree[key])

    def test_fast_tensorstore_metadata_preserves_bfloat16_dtype(self, tmp_checkpoint_dir, mesh):
        """Assumed zarr metadata uses TensorStore's native bfloat16 dtype spelling."""
        sharding = NamedSharding(mesh, PartitionSpec())
        spec = _tensorstore_spec_for_load(
            str(Path(tmp_checkpoint_dir) / "arr"),
            sharding=sharding,
            shape=(2, 4),
            storage_dtype=jnp.bfloat16,
            assume_metadata=True,
            metadata={"shape": [2, 4], "chunks": [2, 4], "dtype": "bfloat16"},
        )

        assert spec["metadata"]["dtype"] == "bfloat16"

    def test_fast_tensorstore_metadata_not_synthesized_without_sidecar(self, tmp_checkpoint_dir, mesh):
        """Assumed metadata is only embedded when exact sidecar metadata is supplied."""
        spec = _tensorstore_spec_for_load(
            str(Path(tmp_checkpoint_dir) / "arr"),
            sharding=NamedSharding(mesh, PartitionSpec()),
            shape=(2, 4),
            storage_dtype=jnp.bfloat16,
            assume_metadata=True,
        )

        assert "metadata" not in spec

    def test_load_pytree_prefix_mismatch_raises(self, tmp_checkpoint_dir, mesh):
        """Loading with a different prefix than saved raises ValueError."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        import shutil

        wrong = Path(tmp_checkpoint_dir) / "optimizer_structure.json"
        shutil.copy(Path(tmp_checkpoint_dir) / "model_structure.json", wrong)

        with pytest.raises(ValueError, match="prefix"):
            mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="optimizer")

    def test_load_pytree_missing_structure_raises(self, tmp_checkpoint_dir, mesh):
        """Loading without structure is strict unless can_skip_structure=True."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model")
        (Path(tmp_checkpoint_dir) / "model_structure.json").unlink()

        with pytest.raises(FileNotFoundError, match="can_skip_structure"):
            mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model")

    def test_load_pytree_missing_structure_and_index_raises(self, tmp_checkpoint_dir, mesh):
        """Index fallback still requires tensorstore_index.json."""
        mgr = AsyncCheckpointManager()
        with pytest.raises(FileNotFoundError, match="tensorstore_index"):
            mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model", can_skip_structure=True)

    def test_load_pytree_missing_structure_uses_tensorstore_index(self, tmp_checkpoint_dir, mesh):
        """Index-only checkpoints reconstruct array trees when structure JSON is absent."""
        tree = {"model": {"layers": {"0": {"w": jnp.arange(6).reshape(2, 3), "b": jnp.ones(3)}}}}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        (Path(tmp_checkpoint_dir) / "model_structure.json").unlink()

        seen_keys = []

        def remember_key(arr, key):
            """Record callback keys and return arrays unchanged."""
            seen_keys.append(key)
            return arr

        loaded, _ = mgr.load_pytree(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            callback=remember_key,
            can_skip_structure=True,
        )

        assert jnp.allclose(loaded["model"]["layers"]["0"]["w"], tree["model"]["layers"]["0"]["w"])
        assert jnp.allclose(loaded["model"]["layers"]["0"]["b"], tree["model"]["layers"]["0"]["b"])
        assert "model.model.layers.0.w" in seen_keys

    def test_can_skip_structure_with_structure_preserves_exact_tree(self, tmp_checkpoint_dir, mesh):
        """can_skip_structure=True is a no-op when the structure sidecar exists."""
        tree = {"arr": jnp.arange(4), "none": None, "name": "tx", "step": 5}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="tx")

        loaded, meta = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="tx", can_skip_structure=True)

        assert jnp.allclose(loaded["arr"], tree["arr"])
        assert loaded["none"] is None
        assert loaded["name"] == "tx"
        assert loaded["step"] == 5
        assert isinstance(meta, dict)

    def test_load_pytree_missing_structure_chunked_dtype_callback(self, tmp_checkpoint_dir, mesh):
        """Index fallback supports chunked loads, dtype casts, and callbacks."""
        tree = {
            "model": {
                "a": jnp.arange(4, dtype=jnp.float32),
                "b": jnp.ones((2, 2), dtype=jnp.float32),
                "c": jnp.full((1,), 3.0, dtype=jnp.float32),
            }
        }
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")
        (Path(tmp_checkpoint_dir) / "model_structure.json").unlink()

        seen_keys = []

        def add_one(arr, key):
            """Record fallback keys and add one after dtype conversion."""
            seen_keys.append(key)
            return arr + jnp.asarray(1, dtype=arr.dtype)

        loaded, _ = mgr.load_pytree(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            dtype=jnp.bfloat16,
            chunk_size=1,
            callback=add_one,
            can_skip_structure=True,
        )

        assert set(seen_keys) == {"model.model.a", "model.model.b", "model.model.c"}
        assert loaded["model"]["a"].dtype == jnp.bfloat16
        assert loaded["model"]["b"].dtype == jnp.bfloat16
        assert jnp.allclose(loaded["model"]["a"], tree["model"]["a"].astype(jnp.bfloat16) + 1)
        assert jnp.allclose(loaded["model"]["b"], tree["model"]["b"].astype(jnp.bfloat16) + 1)
        assert jnp.allclose(loaded["model"]["c"], tree["model"]["c"].astype(jnp.bfloat16) + 1)

    def test_load_pytree_missing_structure_fast_options(self, tmp_checkpoint_dir, mesh):
        """Index-only fallback supports TensorStore fast-load options."""
        tree = {
            "model": {
                "a": jnp.arange(4, dtype=jnp.float32),
                "b": jnp.ones((2, 2), dtype=jnp.float32),
            }
        }
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")
        (Path(tmp_checkpoint_dir) / "model_structure.json").unlink()

        loaded, _ = mgr.load_pytree(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            can_skip_structure=True,
            concurrent_gb=1,
            tensorstore_io_concurrency=4,
            tensorstore_copy_concurrency=4,
            tensorstore_cache_gb=1,
            tensorstore_assume_metadata=True,
            tensorstore_metadata_workers=2,
            show_progress=False,
            progress_every=2,
        )

        assert jnp.allclose(loaded["model"]["a"], tree["model"]["a"])
        assert jnp.allclose(loaded["model"]["b"], tree["model"]["b"])

    def test_load_pytree_missing_structure_uses_template(self, tmp_checkpoint_dir, mesh):
        """Index-only checkpoints can load into a caller-provided template."""
        tree = {"model": {"w": jnp.arange(4, dtype=jnp.float32)}, "step": 7}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        (Path(tmp_checkpoint_dir) / "model_structure.json").unlink()

        template = {"model": {"w": jnp.zeros(4, dtype=jnp.float32)}, "step": 99}
        loaded, _ = mgr.load_pytree(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            template=template,
            can_skip_structure=True,
        )

        assert jnp.allclose(loaded["model"]["w"], tree["model"]["w"])
        assert loaded["step"] == 99

    def test_load_pytree_missing_structure_missing_prefix_raises(self, tmp_checkpoint_dir, mesh):
        """Index fallback still validates the requested prefix."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"w": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        with pytest.raises(ValueError, match="Prefix 'optimizer' not found"):
            mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="optimizer", can_skip_structure=True)

    def test_structure_file_written(self, tmp_checkpoint_dir, mesh):
        """save_pytree writes pytree_structure.json and checkpoint_metadata.json."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        assert (Path(tmp_checkpoint_dir) / "model_structure.json").exists()
        assert (Path(tmp_checkpoint_dir) / "checkpoint_metadata.json").exists()
        assert (Path(tmp_checkpoint_dir) / "tensorstore_index.json").exists()

    def test_structure_content(self, tmp_checkpoint_dir, mesh):
        """structure file contains expected keys."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        struct_path = Path(tmp_checkpoint_dir) / "model_structure.json"
        data = json.loads(struct_path.read_text())
        assert data["format"] == "pytree-structure"
        assert data["prefix"] == "model"
        assert "treedef_b64" in data
        assert "arr_mask" in data
        assert "array_keys" in data

    def test_save_load_gcs_roundtrip(self, mesh, gcs_auth_ino):
        """Save and load from GCS bucket gs://uscentral1stuff/spx-save-tmp."""
        import uuid

        mgr = AsyncCheckpointManager()
        run_id = str(uuid.uuid4())[:8]
        gcs_path = f"gs://uscentral1stuff/spx-save-tmp/test-{run_id}"

        sh = NamedSharding(mesh, PartitionSpec("x", "y"))
        arr = jax.device_put(jnp.arange(16).reshape(4, 4), sh)
        tree = {"w": arr, "b": jnp.ones(4), "step": 99}

        mgr.save_pytree(tree, gcs_path, mesh=mesh, prefix="model")
        loaded, _meta = mgr.load_pytree(gcs_path, mesh, prefix="model")

        assert jnp.allclose(loaded["w"], tree["w"])
        assert jnp.allclose(loaded["b"], tree["b"])
        assert loaded["step"] == 99

    def test_dtype_casting(self, tmp_checkpoint_dir, mesh):
        """dtype parameter casts floating-point arrays before saving."""
        tree = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model", dtype=jnp.bfloat16)

        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model")
        assert loaded["w"].dtype == jnp.bfloat16

    def test_extras_preserved(self, tmp_checkpoint_dir, mesh):
        """extras dict survives the roundtrip in metadata."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(
            {"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model", extras={"lr": 0.001, "epoch": 5}
        )
        _loaded, meta = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model")
        assert meta.get("lr") == 0.001
        assert meta.get("epoch") == 5

    def test_global_manager_lazy_init(self):
        """global_manager is created lazily on first access."""
        mgr = AsyncCheckpointManager()
        assert mgr._global_manager is None
        _ = mgr.global_manager
        assert mgr._global_manager is not None

    def test_load_pytree_strict_shapes_false(self, tmp_checkpoint_dir, mesh):
        """strict_shapes=False allows shape coercion via template."""
        tree = {"x": jnp.arange(6).reshape(2, 3)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        template = {"x": jnp.zeros((1, 2, 3))}
        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model", template=template, strict_shapes=False)
        assert loaded["x"].shape == (1, 2, 3)

    def test_load_pytree_missing_array_raises(self, tmp_checkpoint_dir, mesh):
        """Missing array files raise FileNotFoundError."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        import os

        os.remove(os.path.join(tmp_checkpoint_dir, "model", "a", ".zarray"))

        with pytest.raises(FileNotFoundError, match="arrays missing"):
            mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model")

    def test_load_pytree_callback_transforms(self, tmp_checkpoint_dir, mesh):
        """callback is applied to every loaded array."""
        tree = {"a": jnp.ones((2, 2)), "b": jnp.zeros(2)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        def negate(arr, key):
            """Negate the input."""
            return -arr

        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model", callback=negate)
        assert jnp.allclose(loaded["a"], -tree["a"])
        assert jnp.allclose(loaded["b"], -tree["b"])
