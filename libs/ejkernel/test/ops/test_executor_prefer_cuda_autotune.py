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

"""Unit tests for ``Executor._prefer_cuda_cfg`` interaction with autotune & caches."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
import pytest
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    PersistentCache,
    Tuner,
    overlay_cache,
)
from ejkernel.ops.config.persistent import PROVENANCE_AUTOTUNE
from ejkernel.ops.utils.fingerprint import device_fingerprint

pytestmark = pytest.mark.cpu_ok


@dataclass
class DummyGpuConfig:
    """Minimal GPU config with platform, backend, and Triton tuning knobs."""

    platform: str = "triton"
    backend: str = "gpu"
    block_m: int = 128
    num_warps: int = 4
    num_stages: int = 2


class DummyGpuKernel(Kernel[DummyGpuConfig, DummyGpuConfig]):
    """Pure-Python kernel whose ``run`` returns the selected config without array compute."""

    def __init__(self, op_id: str = "dummy_gpu_op", heuristic_platform: str = "triton"):
        super().__init__(op_id)
        self._heuristic_platform = heuristic_platform

    def run(self, x, cfg: DummyGpuConfig, **kwargs) -> DummyGpuConfig:
        del x, kwargs
        return cfg

    def heuristic_cfg(self, inv) -> DummyGpuConfig:
        del inv
        return DummyGpuConfig(platform=self._heuristic_platform, backend="any", block_m=128, num_warps=4, num_stages=2)

    def candidate_cfgs(self, inv):
        del inv
        return [
            DummyGpuConfig(platform="triton", backend="gpu", block_m=64, num_warps=8, num_stages=3),
            DummyGpuConfig(platform="cuda", backend="gpu", block_m=128, num_warps=4, num_stages=2),
        ]


class FixedWinnerTuner(Tuner[DummyGpuConfig]):
    """Deterministic tuner returning a fixed winner without compiling or executing XLA."""

    def __init__(self, winner: DummyGpuConfig | None):
        super().__init__(warmup=0, iters=1)
        self.winner = winner
        self.calls = 0

    def autotune(self, make_fn, args, kwargs, candidates):
        del make_fn, args, kwargs, candidates
        self.calls += 1
        return self.winner


class ExplodingTuner(Tuner[DummyGpuConfig]):
    """Asserts autotune is never invoked when a persistent cache entry should hit."""

    def autotune(self, make_fn, args, kwargs, candidates):
        raise AssertionError("autotune should not have been called")


def _simulate_nvidia_gpu(monkeypatch, executor: Executor, *, has_cute: bool, has_cuda: bool) -> None:
    exec_mod = importlib.import_module("ejkernel.ops.execution.executor")
    sel_mod = importlib.import_module("ejkernel.ops.config.selection")
    monkeypatch.setattr(exec_mod, "get_device_platform", lambda: "gpu")
    monkeypatch.setattr(sel_mod, "get_device_platform", lambda: "gpu")
    monkeypatch.setattr(executor, "_is_nvidia_gpu", lambda: True)
    monkeypatch.setattr(executor, "_has_cute_impl", lambda op_id: has_cute)
    monkeypatch.setattr(executor, "_has_cuda_impl", lambda op_id: has_cuda)


def _make_key(kernel: Kernel, x) -> tuple[str, str, str]:
    inv = Invocation(op_id=kernel.op_id, args=(x,), kwargs={})
    return (device_fingerprint(), f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}", inv.make_key(kernel.key_builder))


@pytest.mark.parametrize("heuristic_platform", ["triton", "auto"])
@pytest.mark.parametrize(
    ("has_cute", "has_cuda", "expected_platform"),
    [
        (True, True, "cute"),
        (False, True, "cuda"),
    ],
)
def test_heuristic_triton_and_auto_still_upgrade_on_nvidia_gpu(
    monkeypatch,
    heuristic_platform,
    has_cute,
    has_cuda,
    expected_platform,
):
    """Heuristic ``triton`` and ``auto`` configs upgrade to ``cute``/``cuda`` and keep 3-arg spy compatibility."""
    selector = ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=False, cache_miss_fallback="heuristics"),
        persistent=None,
    )
    executor = Executor(selector)
    _simulate_nvidia_gpu(monkeypatch, executor, has_cute=has_cute, has_cuda=has_cuda)

    # Mirror the 3-positional-arg monkeypatch used by test_pr306_kernel_fixes.py.
    prefer_calls: list[DummyGpuConfig] = []
    orig_prefer = executor._prefer_cuda_cfg

    def spy_prefer(cfg, kernel, inv):
        out = orig_prefer(cfg, kernel, inv)
        prefer_calls.append(out)
        return out

    monkeypatch.setattr(executor, "_prefer_cuda_cfg", spy_prefer)

    kernel = DummyGpuKernel(heuristic_platform=heuristic_platform)
    x = np.zeros((2, 8), dtype=np.float32)

    # Call 1 (heuristic fallback) and Call 2 (in-memory cache hit of heuristic).
    cfg1 = executor(kernel, x)
    cfg2 = executor(kernel, x)
    cfg3 = executor.choose_config(kernel, x)

    for cfg in (cfg1, cfg2, cfg3):
        assert cfg.platform == expected_platform
        assert cfg.backend == "gpu"
    assert len(prefer_calls) == 3
    assert all(c.platform == expected_platform and c.backend == "gpu" for c in prefer_calls)


@pytest.mark.parametrize(("has_cute", "has_cuda"), [(True, True), (False, True)])
def test_autotuned_triton_winner_is_never_overwritten_to_cute_or_cuda(monkeypatch, has_cute, has_cuda):
    """A concrete ``platform='triton'`` winner from autotuning must survive ``_prefer_cuda_cfg`` intact."""
    triton_winner = DummyGpuConfig(platform="triton", backend="gpu", block_m=64, num_warps=8, num_stages=3)
    tuner = FixedWinnerTuner(triton_winner)
    selector = ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="autotune"),
        tuner=tuner,
        persistent=None,
    )
    executor = Executor(selector)
    _simulate_nvidia_gpu(monkeypatch, executor, has_cute=has_cute, has_cuda=has_cuda)

    kernel = DummyGpuKernel()
    x = np.zeros((2, 8), dtype=np.float32)

    cfg_first = executor(kernel, x)
    cfg_second = executor(kernel, x)
    cfg_chosen = executor.choose_config(kernel, x)

    assert tuner.calls == 1
    assert cfg_first == triton_winner
    assert cfg_second == triton_winner
    assert cfg_chosen == triton_winner


@pytest.mark.parametrize("cache_miss_fallback", ["heuristics", "autotune"])
@pytest.mark.parametrize(("has_cute", "has_cuda"), [(True, True), (False, True)])
def test_persistent_cached_triton_winner_is_never_overwritten(
    monkeypatch,
    tmp_path,
    cache_miss_fallback,
    has_cute,
    has_cuda,
):
    """A trusted persistent-cache ``platform='triton'`` winner must not be overwritten in a fresh process."""
    path = tmp_path / "dummy-gpu-op.json"
    kernel = DummyGpuKernel()
    x = np.zeros((2, 8), dtype=np.float32)
    dev, op_id_v, call_key = _make_key(kernel, x)

    triton_winner = DummyGpuConfig(platform="triton", backend="gpu", block_m=64, num_warps=8, num_stages=3)
    PersistentCache("dummy-gpu-op", path=str(path), cfg_type=DummyGpuConfig).put(
        dev,
        op_id_v,
        call_key,
        triton_winner,
        provenance=PROVENANCE_AUTOTUNE,
    )

    # Fresh process: empty in-memory cache and empty _autotuned_keys.
    selector = ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback=cache_miss_fallback),
        tuner=ExplodingTuner(),
        persistent=PersistentCache("dummy-gpu-op", path=str(path), cfg_type=DummyGpuConfig),
    )
    executor = Executor(selector)
    _simulate_nvidia_gpu(monkeypatch, executor, has_cute=has_cute, has_cuda=has_cuda)

    # Call 1 loads from persistent cache into memory; Call 2 hits memory cache.
    cfg_first = executor(kernel, x)
    cfg_second = executor(kernel, x)
    cfg_chosen = executor.choose_config(kernel, x)

    assert cfg_first == triton_winner
    assert cfg_second == triton_winner
    assert cfg_chosen == triton_winner


@pytest.mark.parametrize("cache_miss_fallback", ["heuristics", "autotune"])
def test_overlay_triton_is_preserved_while_overlay_auto_upgrades(monkeypatch, cache_miss_fallback):
    """Overlay ``platform='triton'`` is respected, while overlay ``platform='auto'`` still upgrades."""
    selector = ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=False, cache_miss_fallback=cache_miss_fallback),
        persistent=None,
    )
    executor = Executor(selector)
    _simulate_nvidia_gpu(monkeypatch, executor, has_cute=True, has_cuda=True)

    kernel = DummyGpuKernel()
    x = np.zeros((2, 8), dtype=np.float32)
    key = _make_key(kernel, x)

    pinned_triton = DummyGpuConfig(platform="triton", backend="gpu", block_m=32, num_warps=2, num_stages=4)
    with overlay_cache({key: pinned_triton}):
        assert executor(kernel, x) == pinned_triton
        assert executor.choose_config(kernel, x) == pinned_triton

    auto_cfg = DummyGpuConfig(platform="auto", backend="any", block_m=128, num_warps=4, num_stages=2)
    with overlay_cache({key: auto_cfg}):
        upgraded = executor(kernel, x)
        assert upgraded.platform == "cute"
        assert upgraded.backend == "gpu"


def test_failed_autotune_falling_back_to_heuristics_still_upgrades(monkeypatch):
    """When all autotune candidates fail and selection falls back to heuristics, upgrade still applies."""
    tuner = FixedWinnerTuner(winner=None)
    selector = ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, allow_heuristics=True, cache_miss_fallback="autotune"),
        tuner=tuner,
        persistent=None,
    )
    executor = Executor(selector)
    _simulate_nvidia_gpu(monkeypatch, executor, has_cute=False, has_cuda=True)

    kernel = DummyGpuKernel(heuristic_platform="triton")
    x = np.zeros((2, 8), dtype=np.float32)

    cfg = executor(kernel, x)
    assert tuner.calls == 1
    assert cfg.platform == "cuda"
    assert cfg.backend == "gpu"
