"""Tests for :meth:`Module.structure_hash` and :meth:`Module.shape_hash`."""

from __future__ import annotations

import jax.numpy as jnp

import spectrax as spx


class ShapeOnly(spx.Module):
    """Fixture module for testing."""

    def __init__(self, shape: tuple[int, ...], fill: float = 0.0):
        """Initialize with weight."""
        super().__init__()
        self.weight = spx.Parameter(jnp.full(shape, fill, dtype=jnp.float32))

    def forward(self, x):
        """Run the forward pass."""
        return x


class Config:
    """Simple configuration object for testing."""

    def __init__(self, hidden_size: int):
        """Initialize with hidden_size."""
        self.hidden_size = hidden_size

    def to_dict(self):
        """Return a dictionary representation."""
        return {"hidden_size": self.hidden_size}


class Configured(spx.Module):
    """Fixture module for testing."""

    def __init__(self, config: Config):
        """Initialize with config, weight."""
        super().__init__()
        self.config = config
        self.weight = spx.Parameter(jnp.zeros((1,), dtype=jnp.float32))

    def forward(self, x):
        """Run the forward pass."""
        return x + self.config.hidden_size


def test_structure_hash_is_stable_and_excludes_values():
    """Structure hash is stable and excludes values."""
    left = ShapeOnly((2, 3), fill=0.0)
    right = ShapeOnly((2, 3), fill=1.0)

    assert isinstance(left.structure_hash(), str)
    assert len(left.structure_hash()) == 64
    assert left.structure_hash() == right.structure_hash()

    left.weight.value = jnp.ones((2, 3), dtype=jnp.float32) * 7
    assert left.structure_hash() == right.structure_hash()


def test_shape_hash_is_stable_and_excludes_values():
    """Shape hash is stable and excludes values."""
    left = ShapeOnly((2, 3), fill=0.0)
    right = ShapeOnly((2, 3), fill=1.0)

    assert isinstance(left.shape_hash(), str)
    assert len(left.shape_hash()) == 64
    assert left.shape_hash() == right.shape_hash()

    right.weight.value = jnp.ones((2, 3), dtype=jnp.float32) * 9
    assert left.shape_hash() == right.shape_hash()


def test_shape_hash_changes_on_shape_without_hashing_values():
    """Shape hash changes on shape without hashing values."""
    left = ShapeOnly((2, 3))
    right = ShapeOnly((4, 3))

    assert left.structure_hash() == right.structure_hash()
    assert left.shape_hash() != right.shape_hash()


def test_structure_hash_includes_canonical_opaque_config_signature():
    """Structure hash includes canonical opaque config signature."""
    left = Configured(Config(hidden_size=128))
    right = Configured(Config(hidden_size=256))

    assert left.config.hidden_size == 128
    assert right.config.hidden_size == 256
    assert left.structure_hash() != right.structure_hash()
    assert left.shape_hash() != right.shape_hash()
