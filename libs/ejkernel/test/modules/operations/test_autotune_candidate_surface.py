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

"""Candidate-output and selection contracts, without compiling or timing kernels.

Platform-specific overrides are optional: generic candidates may defer platform
resolution with ``auto``. Specialized candidate lists must identify the concrete
implementations to benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest
from ejkernel.modules.base import KernelConfig
from ejkernel.modules.operations.all_gather import AllGather
from ejkernel.modules.operations.all_reduce import AllReduce
from ejkernel.modules.operations.all_to_all import AllToAll
from ejkernel.modules.operations.compressed_window_attention import CompressedWindowAttention
from ejkernel.modules.operations.configs import (
    AllGatherConfig,
    AllReduceConfig,
    AllToAllConfig,
    CompressedWindowAttentionConfig,
    FlashAttentionConfig,
    FusedMlpConfig,
)
from ejkernel.modules.operations.flash_attention import FlashAttention
from ejkernel.modules.operations.fused_mlp import FusedMlp
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Invocation, Kernel
from ejkernel.ops.config import selection


class _HeuristicOnlyKernel(Kernel):
    """Exercise the base class's optional candidate implementation."""

    def heuristic_cfg(self, inv):
        return KernelConfig(block_q=inv.kwargs["block_q"], platform="xla", backend="any")


class _InheritedAllGather(AllGather):
    """An operation subclass need not redeclare its parent's candidate methods."""


class _CandidatePicker:
    """Replace hardware timing only; retain the selector's real candidate dispatch."""

    def __init__(self):
        self.candidates = ()

    def autotune(self, make_fn, args, kwargs, candidates):
        self.candidates = tuple(candidates)
        assert self.candidates, "Autotuning must offer at least one candidate"
        return self.candidates[-1]


def _choose_candidates(monkeypatch, kernel, inv, backend):
    """Run public config selection with simulated hardware and no kernel execution."""
    monkeypatch.setattr(selection, "device_fingerprint", lambda: "candidate-contract-test")
    monkeypatch.setattr(selection, "get_device_platform", lambda: backend)
    picker = _CandidatePicker()
    selector = ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            allow_heuristics=False,
            cache_miss_fallback="autotune",
            validate_backward=False,
        ),
        tuner=picker,
        persistent=None,
    )
    chosen = selector.choose(inv, kernel)
    return chosen, picker.candidates


@pytest.mark.parametrize("backend", ["cpu", "gpu", "tpu"])
@pytest.mark.parametrize("block_q", [64, 320])
def test_inherited_base_candidates_use_invocation_heuristic(monkeypatch, backend, block_q):
    kernel = _HeuristicOnlyKernel()
    inv = Invocation(op_id=kernel.op_id, args=(), kwargs={"block_q": block_q})
    chosen, candidates = _choose_candidates(monkeypatch, kernel, inv, backend)

    expected = KernelConfig(block_q=block_q, platform="xla", backend="any")
    assert candidates == (expected,)
    assert chosen == expected


@pytest.mark.parametrize("method", [None, "shard_map"])
@pytest.mark.parametrize(
    ("kernel_type", "backend", "expected"),
    [
        pytest.param(AllGather, "gpu", AllGatherConfig(platform="auto", backend="any"), id="all-gather-gpu"),
        pytest.param(AllReduce, "gpu", AllReduceConfig(platform="auto", backend="any"), id="all-reduce-gpu"),
        *[
            pytest.param(AllToAll, backend, AllToAllConfig(platform="auto", backend="any"), id=f"all-to-all-{backend}")
            for backend in ("cpu", "gpu", "tpu")
        ],
        *[
            pytest.param(
                CompressedWindowAttention,
                backend,
                CompressedWindowAttentionConfig(fwd_params=None, platform="auto", backend="any"),
                id=f"compressed-window-{backend}",
            )
            for backend in ("cpu", "gpu", "tpu")
        ],
    ],
)
def test_operations_without_specialized_candidates_select_generic_configs(
    monkeypatch, kernel_type, backend, expected, method
):
    kernel = kernel_type()
    inv = Invocation(op_id=kernel.op_id, args=(), kwargs={}, method=method)
    chosen, candidates = _choose_candidates(monkeypatch, kernel, inv, backend)

    assert candidates == (expected,)
    assert chosen == expected


@pytest.mark.parametrize("method", [None, "shard_map"])
@pytest.mark.parametrize(
    ("kernel_type", "config_type"),
    [(AllGather, AllGatherConfig), (AllReduce, AllReduceConfig), (_InheritedAllGather, AllGatherConfig)],
)
def test_tpu_collective_candidates_include_concrete_algorithms_and_xla_fallback(
    monkeypatch, kernel_type, config_type, method
):
    kernel = kernel_type()
    inv = Invocation(op_id=kernel.op_id, args=(), kwargs={}, method=method)
    chosen, candidates = _choose_candidates(monkeypatch, kernel, inv, "tpu")

    expected = (
        config_type(mode="one_shot", platform="pallas", backend="tpu"),
        config_type(mode="ring", platform="pallas", backend="tpu"),
        config_type(mode="auto", platform="xla", backend="any"),
    )
    assert len(candidates) == len(expected)
    assert set(candidates) == set(expected)
    assert chosen in expected
    assert chosen.platform != "auto"


@pytest.mark.parametrize(
    ("backend", "requested", "expected_platforms"),
    [
        ("gpu", None, {"triton", "cuda", "cute", "tilelang", "xla"}),
        ("gpu", "auto", {"triton", "cuda", "cute", "tilelang", "xla"}),
        *[("gpu", platform, {platform}) for platform in ("triton", "cuda", "cute", "tilelang", "xla")],
        ("tpu", None, {"pallas", "xla"}),
        ("tpu", "auto", {"pallas", "xla"}),
    ],
)
def test_flash_attention_candidates_resolve_concrete_platforms(monkeypatch, backend, requested, expected_platforms):
    kernel = FlashAttention()
    # Host arrays provide shape/dtype metadata only; no accelerator work is run.
    query = np.empty((1, 256, 4, 64), dtype=np.float16)
    key = np.empty((1, 256, 2, 64), dtype=np.float16)
    inv = Invocation(
        op_id=kernel.op_id,
        args=(),
        kwargs={"query": query, "key": key, "causal": True, "platform": requested},
    )
    chosen, candidates = _choose_candidates(monkeypatch, kernel, inv, backend)

    assert {cfg.platform for cfg in candidates} == expected_platforms
    for cfg in candidates:
        assert isinstance(cfg, FlashAttentionConfig)
        assert cfg.backend == ("any" if cfg.platform == "xla" else backend)
        assert cfg.fwd_params.q_blocksize > 0
        assert cfg.fwd_params.kv_blocksize > 0
        assert cfg.bwd_params.q_blocksize > 0
        assert cfg.bwd_params.kv_blocksize > 0
    assert chosen.platform in expected_platforms
    assert chosen.backend == ("any" if chosen.platform == "xla" else backend)


@pytest.mark.parametrize("backend", ["cpu", "gpu", "tpu"])
@pytest.mark.parametrize(
    ("variant", "expected_tiles"),
    [
        ("reference", {(1024, 512)}),
        ("w4a4", {(512, 512), (1024, 512), (2048, 512)}),
        ("bf16", {(512, 256), (512, 512), (1024, 256), (1024, 512)}),
    ],
)
def test_fused_mlp_candidates_preserve_variant_configuration(monkeypatch, backend, variant, expected_tiles):
    kernel = FusedMlp(variant=variant)
    inv = Invocation(op_id=kernel.op_id, args=(), kwargs={})
    chosen, candidates = _choose_candidates(monkeypatch, kernel, inv, backend)

    expected = {
        FusedMlpConfig(tile_i=tile_i, tile_m=tile_m, prefill_threshold=256, platform="auto", backend="any")
        for tile_i, tile_m in expected_tiles
    }
    assert len(candidates) == len(expected)
    assert set(candidates) == expected
    assert chosen in expected
