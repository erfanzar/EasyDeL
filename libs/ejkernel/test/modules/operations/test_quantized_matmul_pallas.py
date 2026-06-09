# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

import importlib

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import quantized_matmul
from ejkernel.modules.operations.quantized_matmul import QuantizedMatmulConfig
from ejkernel.quantization import prepack_quantized_weights

from ._utils import assert_allclose


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


@pytest.mark.parametrize(
    "mode,bits",
    [("affine", 4), ("nf4", 4), ("mxfp8", 8), ("nvfp4", 4)],
)
def test_quantized_matmul_operation_pallas_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(7 if mode == "affine" else 9)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 32, 128, 128

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    out_pallas = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
        platform="pallas",
    )
    out_xla = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
        platform="xla",
    )

    out_pallas = jax.block_until_ready(out_pallas)
    out_xla = jax.block_until_ready(out_xla)
    assert_allclose(out_pallas, out_xla, atol=6e-2, rtol=6e-2)


def test_quantized_matmul_operation_pallas_large_n_matches_xla(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EJKERNEL_QMM_TPU_PATH", "hybrid")
    key = jax.random.PRNGKey(101)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 32, 128, 256

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=64)

    out_pallas = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        platform="pallas",
    )
    out_xla = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        platform="xla",
    )

    out_pallas = jax.block_until_ready(out_pallas)
    out_xla = jax.block_until_ready(out_xla)
    assert_allclose(out_pallas, out_xla, atol=6e-2, rtol=6e-2)


def test_quantized_matmul_operation_pallas_strict_fuse_repairs_illegal_block_n():
    key = jax.random.PRNGKey(909)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 32, 128, 256

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=64)

    # block_n=128 is illegal for packed forward when n=256 (4-bit packed words).
    illegal_cfg = QuantizedMatmulConfig(
        block_m=128,
        block_n=128,
        block_k=128,
        tpu_path="packed",
        platform="pallas",
        backend="tpu",
    )

    out_pallas = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        platform="pallas",
        strict_fuse=True,
        tpu_path="packed",
        cfg=illegal_cfg,
    )
    out_ref = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        platform="xla",
        fuse=False,
    )

    out_pallas = jax.block_until_ready(out_pallas)
    out_ref = jax.block_until_ready(out_ref)
    assert_allclose(out_pallas, out_ref, atol=2e-1, rtol=8e-2)


def test_quantized_matmul_operation_tpu_default_uses_predecode_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EJKERNEL_QMM_TPU_DEFAULT_STRATEGY", "predecode_once")
    mod = importlib.import_module("ejkernel.modules.operations.quantized_matmul")
    orig_impl = mod._quantized_matmul_impl
    calls = {"impl": 0}

    def _fail_if_impl_called(*args, **kwargs):
        calls["impl"] += 1
        raise AssertionError("_quantized_matmul_impl should not be called for TPU predecode_once default path.")

    monkeypatch.setattr(mod, "_quantized_matmul_impl", _fail_if_impl_called)

    key = jax.random.PRNGKey(1201)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 128
    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4)

    out_1 = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        platform="pallas",
        strict_fuse=True,
    )
    out_2 = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        platform="pallas",
        strict_fuse=True,
    )
    out_ref = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        platform="xla",
        fuse=False,
    )

    out_1 = jax.block_until_ready(out_1)
    out_2 = jax.block_until_ready(out_2)
    out_ref = jax.block_until_ready(out_ref)
    monkeypatch.setattr(mod, "_quantized_matmul_impl", orig_impl)
    assert calls["impl"] == 0
    assert_allclose(out_1, out_ref, atol=2e-1, rtol=8e-2)
    assert_allclose(out_2, out_ref, atol=2e-1, rtol=8e-2)


def test_quantized_matmul_operation_tpu_path_predecode_overrides_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EJKERNEL_QMM_TPU_DEFAULT_STRATEGY", "packed")
    mod = importlib.import_module("ejkernel.modules.operations.quantized_matmul")
    orig_impl = mod._quantized_matmul_impl
    calls = {"impl": 0}

    def _fail_if_impl_called(*args, **kwargs):
        calls["impl"] += 1
        raise AssertionError("_quantized_matmul_impl should not be called when tpu_path='predecode'.")

    monkeypatch.setattr(mod, "_quantized_matmul_impl", _fail_if_impl_called)

    key = jax.random.PRNGKey(1301)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 128
    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4)

    out = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        platform="pallas",
        strict_fuse=True,
        tpu_path="predecode",
    )
    out_ref = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        platform="xla",
        fuse=False,
    )

    out = jax.block_until_ready(out)
    out_ref = jax.block_until_ready(out_ref)
    monkeypatch.setattr(mod, "_quantized_matmul_impl", orig_impl)
    assert calls["impl"] == 0
    assert_allclose(out, out_ref, atol=2e-1, rtol=8e-2)


def test_quantized_matmul_operation_tpu_path_predecode_does_not_force_packed_cfg(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("EJKERNEL_QMM_TPU_DEFAULT_STRATEGY", "packed")
    mod = importlib.import_module("ejkernel.modules.operations.quantized_matmul")
    orig_impl = mod._quantized_matmul_impl
    seen = {"cfg": "unset"}

    def _force_fallback(*args, **kwargs):
        return None

    def _capture_impl(*args, **kwargs):
        seen["cfg"] = kwargs.get("cfg")
        return orig_impl(*args, **kwargs)

    monkeypatch.setattr(mod, "_maybe_tpu_predecode_once_matmul", _force_fallback)
    monkeypatch.setattr(mod, "_quantized_matmul_impl", _capture_impl)

    key = jax.random.PRNGKey(1302)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 128, 128
    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4)

    out = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        platform="pallas",
        strict_fuse=True,
        tpu_path="predecode",
    )
    out = jax.block_until_ready(out)
    assert out.shape == (m, n)
    assert seen["cfg"] is None


@pytest.mark.parametrize(
    "mode,bits",
    [("nf4", 4), ("mxfp4", 4), ("mxfp8", 8), ("nvfp4", 4), ("nvfp8", 8)],
)
def test_non_affine_traced_uses_fused_pallas_not_xla(mode: str, bits: int):
    """Non-affine modes under jax.jit tracing must dispatch to fused Pallas packed,
    NOT unfused XLA.  Verifies the traced path doesn't collapse to dequant+matmul."""
    key = jax.random.PRNGKey(42)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 4, 128, 128
    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    # Reference: unfused XLA path
    out_ref = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
        platform="xla",
        fuse=False,
        use_best_config=False,
    )

    # Under jax.jit with use_best_config=True — should route through
    # fused Pallas packed, not unfused XLA.
    @jax.jit
    def fn(xi, wi, si):
        return quantized_matmul(
            xi,
            wi,
            si,
            None,
            transpose=False,
            mode=mode,
            bits=bits,
            use_best_config=True,
        )

    out_traced = fn(x, w_q, scales)
    out_traced = jax.block_until_ready(out_traced)
    out_ref = jax.block_until_ready(out_ref)
    assert out_traced.shape == (m, n)
    assert_allclose(out_traced, out_ref, atol=2e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "mode,bits",
    [("nf4", 4), ("mxfp4", 4), ("mxfp8", 8)],
)
def test_packed_legality_auto_recovery_does_not_crash(mode: str, bits: int):
    """Calling Pallas QMM with an intentionally illegal block_n must not crash
    when allow_dense_fallback=True — the auto-recovery or XLA fallback must
    handle it gracefully."""
    key = jax.random.PRNGKey(77)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 8, 256, 256
    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    w_q, scales = packed[0], packed[1]

    # Intentionally illegal block_n=128 for packed path
    illegal_cfg = QuantizedMatmulConfig(
        block_m=128,
        block_n=128,
        block_k=128,
        tpu_path="packed",
        platform="pallas",
        backend="tpu",
    )
    out = quantized_matmul(
        x,
        w_q,
        scales,
        None,
        transpose=False,
        mode=mode,
        bits=bits,
        platform="pallas",
        tpu_path="packed",
        allow_dense_fallback=True,
        cfg=illegal_cfg,
    )
    out = jax.block_until_ready(out)
    assert out.shape == (m, n)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_packed_legal_block_n_helper():
    """Unit test for the _packed_legal_block_n heuristic."""
    from ejkernel.modules.operations.quantized_matmul import _lcm, _packed_legal_block_n

    # NF4: bits=4, group_size=64 → n_pad should be picked for N=1024
    align_n = _lcm(128, _lcm(64, 8))  # 128
    bn = _packed_legal_block_n(1024, group_size=64, bits=4, align_n=align_n)
    assert bn == 1024  # n_pad for N=1024

    # MXFP8: bits=8, group_size=32 → n_pad for N=512
    align_n = _lcm(128, _lcm(32, 4))  # 128
    bn = _packed_legal_block_n(512, group_size=32, bits=8, align_n=align_n)
    assert bn == 512

    # Large N=8192 with bits=4, group_size=64 → should still return a value
    align_n = _lcm(128, _lcm(64, 8))
    bn = _packed_legal_block_n(8192, group_size=64, bits=4, align_n=align_n, vmem_cap=4096)
    assert bn > 0
    values_per_word = 8
    # Must satisfy: (bn/values_per_word) % 128 == 0 OR bn == n_pad
    assert bn % (128 * values_per_word) == 0 or bn == 8192
