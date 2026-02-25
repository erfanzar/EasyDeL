from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _load_ragged_utils_symbols():
    """Import ragged cache update utilities with a lightweight fallback loader.

    The full EasyDeL import graph can pull in optional dependencies (e.g. ejkernel).
    These tests only need a small subset of ragged utils, so we stub the JIT
    decorator when optional deps are unavailable.
    """
    try:
        from easydel.caching.ragged_page.utils import (
            kv_cache_update_jax,
            localize_slice_indices_for_page_shard,
        )

        return kv_cache_update_jax, localize_slice_indices_for_page_shard
    except Exception:
        repo_root = Path(__file__).resolve().parents[3]
        easydel_dir = repo_root / "easydel"

        def _ensure_pkg(name: str, path: Path) -> types.ModuleType:
            module = sys.modules.get(name)
            if module is None:
                module = types.ModuleType(name)
                sys.modules[name] = module
            module.__path__ = [str(path)]
            return module

        _ensure_pkg("easydel", easydel_dir)
        _ensure_pkg("easydel.utils", easydel_dir / "utils")
        _ensure_pkg("easydel.layers", easydel_dir / "layers")
        _ensure_pkg("easydel.layers.caching", easydel_dir / "layers" / "caching")
        _ensure_pkg("easydel.layers.caching.ragged_page", easydel_dir / "layers" / "caching" / "ragged_page")

        # Stub `ejit` to avoid importing ejkernel in this test environment.
        compiling_utils = types.ModuleType("easydel.utils.compiling_utils")

        def ejit(*jit_args, **jit_kwargs):
            def deco(fn):
                return jax.jit(fn, *jit_args, **jit_kwargs)

            return deco

        compiling_utils.ejit = ejit
        sys.modules["easydel.utils.compiling_utils"] = compiling_utils

        utils_path = easydel_dir / "layers" / "caching" / "ragged_page" / "utils.py"
        spec = importlib.util.spec_from_file_location("easydel.layers.caching.ragged_page.utils", utils_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return module.kv_cache_update_jax, module.localize_slice_indices_for_page_shard


kv_cache_update_jax, localize_slice_indices_for_page_shard = _load_ragged_utils_symbols()


def test_kv_cache_update_jax_matches_slice_mapping():
    rng = np.random.default_rng(0)

    num_tokens_bucket = 16
    num_heads = 4
    head_dim = 8
    cache_size = 64

    new_tokens = jnp.asarray(rng.standard_normal((num_tokens_bucket, num_heads, head_dim), dtype=np.float32))
    cache = jnp.zeros((cache_size, num_heads, head_dim), dtype=jnp.float32)

    # Three valid slices that update the first 7 tokens in `new_tokens`, then padding.
    lens = np.array([3, 2, 2, 0, 0], dtype=np.int32)
    src_starts = np.concatenate(([0], np.cumsum(lens[:-1], dtype=np.int32))).astype(np.int32)
    dst_starts = np.array([10, 20, 30, 0, 0], dtype=np.int32)

    slice_indices = jnp.asarray(np.stack([dst_starts, src_starts, lens], axis=0), dtype=jnp.int32)
    num_valid_slices = jnp.asarray([3], dtype=jnp.int32)

    updated = kv_cache_update_jax(
        new_kv_tokens=new_tokens,
        slice_indices=slice_indices,
        kv_cache_pages=cache,
        total_update_slices=num_valid_slices,
        page_size=4,
    )

    expected = np.zeros((cache_size, num_heads, head_dim), dtype=np.float32)
    new_tokens_np = np.asarray(new_tokens)
    for i in range(int(num_valid_slices[0])):
        dst = int(dst_starts[i])
        src = int(src_starts[i])
        ln = int(lens[i])
        expected[dst : dst + ln] = new_tokens_np[src : src + ln]

    np.testing.assert_allclose(np.asarray(updated), expected, atol=0.0, rtol=0.0)


def test_localize_slice_indices_for_page_shard_zeroes_non_local_slices():
    slice_indices = jnp.asarray(
        [
            [0, 5, 8, 13],  # global dst starts
            [0, 1, 2, 4],  # src starts
            [1, 1, 2, 1],  # lengths
        ],
        dtype=jnp.int32,
    )
    localized = localize_slice_indices_for_page_shard(
        slice_indices,
        total_update_slices=jnp.asarray([4], dtype=jnp.int32),
        page_size=4,
        local_flat_cache_positions=8,  # 2 local pages
        page_shard_index=1,  # pages [2, 3]
    )
    expected = np.asarray(
        [
            [0, 0, 0, 5],
            [0, 0, 2, 4],
            [0, 0, 2, 1],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(np.asarray(localized), expected)


def test_kv_cache_update_jax_with_page_shard_index_uses_local_offsets():
    num_heads = 2
    head_dim = 4
    cache = jnp.zeros((8, num_heads, head_dim), dtype=jnp.float32)  # local shard cache (2 pages * page_size=4)
    new_tokens = jnp.arange(6 * num_heads * head_dim, dtype=jnp.float32).reshape(6, num_heads, head_dim)

    # Global page-space mapping: first slice belongs to shard0 (drop),
    # second/third belong to shard1 (local updates).
    slice_indices = jnp.asarray(
        [
            [0, 8, 12, 0],  # global dst starts
            [0, 2, 4, 0],  # src starts
            [2, 2, 2, 0],  # lengths
        ],
        dtype=jnp.int32,
    )
    num_valid_slices = jnp.asarray([3], dtype=jnp.int32)

    updated = kv_cache_update_jax(
        new_kv_tokens=new_tokens,
        slice_indices=slice_indices,
        kv_cache_pages=cache,
        total_update_slices=num_valid_slices,
        page_size=4,
        page_shard_index=jnp.int32(1),
    )

    expected = np.zeros((8, num_heads, head_dim), dtype=np.float32)
    new_tokens_np = np.asarray(new_tokens)
    expected[0:2] = new_tokens_np[2:4]
    expected[4:6] = new_tokens_np[4:6]
    np.testing.assert_allclose(np.asarray(updated), expected, atol=0.0, rtol=0.0)
