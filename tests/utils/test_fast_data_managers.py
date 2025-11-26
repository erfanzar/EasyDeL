"""Tests for fast data managers."""

import tempfile

import numpy as np
import pytest

# Skip if fast data manager utilities are not available in this build.
try:
    from easydel.utils.data_managers import ArrayCache, DataCache, DataStreamOptimizer, FastDataLoader, TokenCache
except Exception as exc:  # pragma: no cover - allow partial test runs
    pytest.skip(f"Data manager utilities unavailable: {exc}", allow_module_level=True)


class TestFastDataLoader:
    """Test FastDataLoader functionality."""

    def test_initialization(self):
        """Test loader initialization."""
        loader = FastDataLoader(
            cache_storage="/tmp/test_cache",
            use_async=False,
            num_workers=2,
        )
        assert loader.num_workers == 2
        assert loader.use_async is False

    def test_json_loading(self, tmp_path):
        """Test JSON file loading."""
        import json

        # Create test JSON file
        test_data = {"key": "value", "number": 42}
        json_file = tmp_path / "test.json"
        with open(json_file, "w") as f:
            json.dump(test_data, f)

        loader = FastDataLoader(use_async=False)
        loaded = loader.load_json(str(json_file))
        assert loaded == test_data

    def test_jsonl_loading(self, tmp_path):
        """Test JSONL file loading."""
        import json

        # Create test JSONL file
        test_data = [{"id": 1}, {"id": 2}, {"id": 3}]
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        loader = FastDataLoader(use_async=False)
        loaded = loader.load_json(str(jsonl_file), lines=True)
        assert loaded == test_data

    def test_stream_jsonl(self, tmp_path):
        """Test JSONL streaming."""
        import json

        # Create larger JSONL file
        jsonl_file = tmp_path / "stream.jsonl"
        with open(jsonl_file, "w") as f:
            for i in range(2500):
                f.write(json.dumps({"id": i}) + "\n")

        loader = FastDataLoader(use_async=False)
        batches = list(loader.stream_jsonl(str(jsonl_file), batch_size=1000))

        assert len(batches) == 3
        assert len(batches[0]) == 1000
        assert len(batches[1]) == 1000
        assert len(batches[2]) == 500


class TestDataCache:
    """Test DataCache functionality."""

    def test_cache_operations(self):
        """Test basic cache operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(
                cache_dir=tmpdir,
                max_size_gb=0.1,
                ttl_hours=1.0,
                use_compression=True,
            )

            # Test set and get
            test_data = {"key": "value", "list": [1, 2, 3]}
            cache.set("test_key", test_data)

            retrieved = cache.get("test_key")
            assert retrieved == test_data

            # Test with params
            cache.set("param_key", test_data, params={"version": 1})
            retrieved = cache.get("param_key", params={"version": 1})
            assert retrieved == test_data

            # Test cache miss
            assert cache.get("nonexistent") is None

    def test_cache_invalidation(self):
        """Test cache invalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            cache.set("key1", "value1")
            cache.set("key2", "value2")

            # Invalidate specific key
            cache.invalidate("key1")
            assert cache.get("key1") is None
            assert cache.get("key2") == "value2"

            # Invalidate all
            cache.invalidate()
            assert cache.get("key2") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            cache.set("key1", "value1")
            cache.set("key2", [1, 2, 3, 4, 5])

            stats = cache.get_stats()
            assert stats["num_entries"] == 2
            assert stats["usage_percent"] >= 0


class TestArrayCache:
    """Test ArrayCache functionality."""

    def test_array_save_load(self):
        """Test array saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ArrayCache(cache_dir=tmpdir)

            # Create test array
            rng = np.random.default_rng(42)
            test_array = rng.standard_normal((100, 50)).astype(np.float32)

            # Save and load
            cache.save_array("test_array", test_array)
            loaded = cache.load_array("test_array")

            np.testing.assert_array_equal(loaded, test_array)

    def test_memmap_mode(self):
        """Test memory-mapped array loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ArrayCache(cache_dir=tmpdir, use_memmap=True)

            test_array = np.ones((100, 100), dtype=np.float32)
            cache.save_array("memmap_test", test_array)

            # Load as memmap
            loaded = cache.load_array("memmap_test", mmap_mode="r")
            assert isinstance(loaded, np.memmap)
            assert loaded.shape == test_array.shape


class TestTokenCache:
    """Test TokenCache functionality."""

    def test_token_caching(self):
        """Test token caching and retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TokenCache(cache_dir=tmpdir)

            # Create test tokens
            text = "This is a test sentence"
            text_hash = cache.hash_text(text)
            input_ids = np.array([1, 2, 3, 4, 5])
            attention_mask = np.array([1, 1, 1, 1, 1])
            metadata = {"length": 5}

            # Cache tokens
            success = cache.cache_tokens(text_hash, input_ids, attention_mask, metadata)
            assert success

            # Retrieve tokens
            loaded_ids, loaded_mask, loaded_meta = cache.get_tokens(text_hash)

            np.testing.assert_array_equal(loaded_ids, input_ids)
            np.testing.assert_array_equal(loaded_mask, attention_mask)
            assert loaded_meta == metadata

    def test_hash_consistency(self):
        """Test text hashing consistency."""
        cache = TokenCache()

        text = "Test text"
        hash1 = cache.hash_text(text)
        hash2 = cache.hash_text(text)

        assert hash1 == hash2
        assert len(hash1) == 16


class TestDataStreamOptimizer:
    """Test DataStreamOptimizer functionality."""

    def test_batch_stream(self):
        """Test stream batching."""
        optimizer = DataStreamOptimizer()

        data = range(100)
        batched = list(optimizer.batch_stream(iter(data), batch_size=10))

        assert len(batched) == 10
        assert all(len(batch) == 10 for batch in batched)

    def test_interleave_streams(self):
        """Test stream interleaving."""
        optimizer = DataStreamOptimizer()

        stream1 = [f"a{i}" for i in range(10)]
        stream2 = [f"b{i}" for i in range(10)]

        interleaved = list(
            optimizer.interleave_streams(
                [stream1, stream2],
                probabilities=[0.5, 0.5],
                seed=42,
            )
        )

        assert len(interleaved) == 20
        assert any("a" in item for item in interleaved)
        assert any("b" in item for item in interleaved)
