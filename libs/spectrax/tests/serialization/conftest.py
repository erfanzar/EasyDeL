# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Serialization-test fixtures for TPU."""

from __future__ import annotations

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

_USCENTRAL1STUFF_TESTS_ENV = "SPECTRAX_RUN_USCENTRAL1STUFF_TESTS"


def _env_truthy(name: str) -> bool:
    """Return whether an environment variable is set to a common truthy value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@pytest.fixture
def gcs_auth_ino():
    """Skip tests that need authenticated access to gs://uscentral1stuff."""
    if not _env_truthy(_USCENTRAL1STUFF_TESTS_ENV):
        pytest.skip(f"requires authenticated access to gs://uscentral1stuff; set {_USCENTRAL1STUFF_TESTS_ENV}=1 to run")


@pytest.fixture
def mesh():
    """A two-axis mesh over local devices, with a single-device CPU fallback."""
    devices = np.array(jax.devices())
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    else:
        devices = devices.reshape(devices.size, 1)
    return Mesh(devices, ("x", "y"))


@pytest.fixture
def tmp_checkpoint_dir():
    """Temporary directory for checkpoint I/O."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_pytree(mesh):
    """Nested pytree with arrays and non-array leaves."""
    sh = NamedSharding(mesh, PartitionSpec("x", "y"))
    arr = jax.device_put(jnp.arange(16).reshape(4, 4), sh)
    return {
        "layer0": {
            "weight": arr,
            "bias": jnp.ones(4),
        },
        "step": 42,
        "name": "test",
    }
