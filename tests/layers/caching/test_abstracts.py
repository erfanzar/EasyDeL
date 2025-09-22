"""Test abstract base classes for caching system."""

from typing import Any

import jax.numpy as jnp
import pytest
from eformer.pytree import auto_pytree

from easydel.layers.caching._abstracts import (
    BaseCache,
    BaseCacheMetadata,
    BaseCacheView,
    BaseRunTimeMetadata,
)


# Concrete implementations for testing
@auto_pytree
class TestMetadata(BaseCacheMetadata):
    """Test implementation of BaseCacheMetadata."""

    test_param: int
    another_param: str

    @classmethod
    def create(cls, test_param: int, another_param: str = "default") -> "TestMetadata":
        """Create validated metadata."""
        if test_param <= 0:
            raise ValueError("test_param must be positive")
        if not another_param:
            raise ValueError("another_param cannot be empty")
        return cls(test_param=test_param, another_param=another_param)


@auto_pytree
class TestRuntimeMetadata(BaseRunTimeMetadata):
    """Test implementation of BaseRunTimeMetadata."""

    current_position: int
    batch_indices: Any


@auto_pytree
class TestCacheView(BaseCacheView):
    """Test implementation of BaseCacheView."""

    metadata: TestMetadata
    layer_index: int | None
    cache_data: Any

    @classmethod
    def init(
        cls,
        metadata: BaseCacheMetadata,
        layer_index: int | None = None,
        **kwargs,
    ) -> "TestCacheView":
        """Initialize test cache view."""
        if not isinstance(metadata, TestMetadata):
            raise TypeError("Expected TestMetadata")
        cache_data = jnp.zeros((metadata.test_param, 10))
        return cls(metadata=metadata, layer_index=layer_index, cache_data=cache_data)

    def concatenate_to_cache(self, data: Any, **kwargs) -> tuple[Any, "TestCacheView"]:
        """Update cache with new data."""
        new_cache_data = self.cache_data + data
        new_view = TestCacheView(
            metadata=self.metadata,
            layer_index=self.layer_index,
            cache_data=new_cache_data,
        )
        return new_cache_data, new_view


@auto_pytree
class TestCache(BaseCache):
    """Test implementation of BaseCache."""

    views: list[TestCacheView | None]

    @classmethod
    def init_cache(
        cls,
        metadata: BaseCacheMetadata,
        num_layers: int = 4,
        **kwargs,
    ) -> "TestCache":
        """Initialize test cache with views."""
        views = []
        for i in range(num_layers):
            view = TestCacheView.init(metadata, layer_index=i)
            views.append(view)
        return cls(views=views)

    @classmethod
    def init_empty(cls, num_layers: int = 4) -> "TestCache":
        """Initialize empty test cache."""
        views = [None] * num_layers
        return cls(views=views)


class TestBaseCacheMetadata:
    """Test BaseCacheMetadata abstract class."""

    def test_metadata_creation(self):
        """Test metadata creation with validation."""
        # Valid creation
        metadata = TestMetadata.create(test_param=10, another_param="test")
        assert metadata.test_param == 10
        assert metadata.another_param == "test"

        # Invalid creation
        with pytest.raises(ValueError, match="test_param must be positive"):
            TestMetadata.create(test_param=0)

        with pytest.raises(ValueError, match="test_param must be positive"):
            TestMetadata.create(test_param=-1)

        with pytest.raises(ValueError, match="another_param cannot be empty"):
            TestMetadata.create(test_param=5, another_param="")

    def test_metadata_is_pytree(self):
        """Test that metadata works as JAX pytree."""
        metadata = TestMetadata.create(test_param=5)
        # Should be able to tree_flatten and tree_unflatten
        import jax.tree_util as tree

        leaves, treedef = tree.tree_flatten(metadata)
        reconstructed = tree.tree_unflatten(treedef, leaves)
        assert reconstructed.test_param == metadata.test_param
        assert reconstructed.another_param == metadata.another_param


class TestBaseRunTimeMetadata:
    """Test BaseRunTimeMetadata class."""

    def test_runtime_metadata_creation(self):
        """Test runtime metadata creation."""
        runtime = TestRuntimeMetadata(current_position=10, batch_indices=jnp.array([0, 1, 2]))
        assert runtime.current_position == 10
        assert runtime.batch_indices.shape == (3,)

    def test_runtime_metadata_is_pytree(self):
        """Test that runtime metadata works as JAX pytree."""
        runtime = TestRuntimeMetadata(current_position=5, batch_indices=jnp.array([0, 1]))
        import jax.tree_util as tree

        leaves, treedef = tree.tree_flatten(runtime)
        reconstructed = tree.tree_unflatten(treedef, leaves)
        assert reconstructed.current_position == runtime.current_position
        assert jnp.allclose(reconstructed.batch_indices, runtime.batch_indices)


class TestBaseCacheView:
    """Test BaseCacheView abstract class."""

    def test_view_initialization(self):
        """Test cache view initialization."""
        metadata = TestMetadata.create(test_param=8)
        view = TestCacheView.init(metadata, layer_index=0)

        assert view.metadata == metadata
        assert view.layer_index == 0
        assert view.cache_data.shape == (8, 10)
        assert jnp.allclose(view.cache_data, 0)

    def test_view_initialization_with_wrong_metadata(self):
        """Test view initialization with wrong metadata type."""

        # Create a different metadata type
        @auto_pytree
        class WrongMetadata(BaseCacheMetadata):
            @classmethod
            def create(cls):
                return cls()

        wrong_metadata = WrongMetadata.create()

        with pytest.raises(TypeError, match="Expected TestMetadata"):
            TestCacheView.init(wrong_metadata)

    def test_view_concatenate_to_cache(self):
        """Test cache view update."""
        metadata = TestMetadata.create(test_param=4)
        view = TestCacheView.init(metadata, layer_index=1)

        # Update with new data
        new_data = jnp.ones((4, 10))
        updated_data, new_view = view.concatenate_to_cache(new_data)

        assert jnp.allclose(updated_data, 1.0)
        assert jnp.allclose(new_view.cache_data, 1.0)
        assert new_view.layer_index == view.layer_index
        assert new_view.metadata == view.metadata

        # Original view should be unchanged (functional update)
        assert jnp.allclose(view.cache_data, 0.0)


class TestBaseCache:
    """Test BaseCache abstract class."""

    def test_cache_init(self):
        """Test cache initialization."""
        metadata = TestMetadata.create(test_param=6)
        cache = TestCache.init_cache(metadata, num_layers=3)

        assert len(cache) == 3
        for i in range(3):
            assert cache[i] is not None
            assert cache[i].layer_index == i
            assert cache[i].metadata == metadata

    def test_cache_init_empty(self):
        """Test empty cache initialization."""
        cache = TestCache.init_empty(num_layers=5)

        assert len(cache) == 5
        for i in range(5):
            assert cache[i] is None

    def test_cache_indexing(self):
        """Test cache indexing operations."""
        metadata = TestMetadata.create(test_param=4)
        cache = TestCache.init_cache(metadata, num_layers=4)

        # Test __getitem__
        assert cache[0].layer_index == 0
        assert cache[-1].layer_index == 3

        # Test slicing
        middle_views = cache[1:3]
        assert len(middle_views) == 2
        assert middle_views[0].layer_index == 1
        assert middle_views[1].layer_index == 2

        # Test out of range
        with pytest.raises(IndexError):
            _ = cache[10]

    def test_cache_setitem(self):
        """Test cache view replacement."""
        metadata = TestMetadata.create(test_param=3)
        cache = TestCache.init_cache(metadata, num_layers=3)

        # Replace a view
        new_view = TestCacheView.init(metadata, layer_index=99)
        cache[1] = new_view

        assert cache[1].layer_index == 99

        # Set to None
        cache[2] = None
        assert cache[2] is None

    def test_cache_len(self):
        """Test cache length."""
        metadata = TestMetadata.create(test_param=2)
        cache = TestCache.init_cache(metadata, num_layers=7)
        assert len(cache) == 7

    def test_cache_without_views(self):
        """Test error when views not initialized."""

        @auto_pytree
        class BrokenCache(BaseCache):
            # Doesn't initialize views attribute

            @classmethod
            def init_cache(cls, metadata, **kwargs):
                return cls()  # Missing views

            @classmethod
            def init_empty(cls):
                return cls()  # Missing views

        broken = BrokenCache.init_empty()
        with pytest.raises(AttributeError, match="The 'views' attribute has not been initialized"):
            len(broken)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
