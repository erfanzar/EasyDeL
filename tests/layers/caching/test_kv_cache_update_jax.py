from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _load_kv_cache_update_jax():
    """Import kv_cache_update_jax with a lightweight fallback loader.

    The full EasyDeL import graph can pull in optional dependencies (e.g. ejkernel).
    This test only needs `kv_cache_update_jax`, so we stub the JIT decorator when
    those deps are unavailable.
    """
    try:
        from easydel.layers.caching.ragged_page.utils import kv_cache_update_jax

        return kv_cache_update_jax
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

        return module.kv_cache_update_jax


kv_cache_update_jax = _load_kv_cache_update_jax()


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
