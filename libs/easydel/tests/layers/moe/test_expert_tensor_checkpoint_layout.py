"""ExpertTensor shards experts, so gate/up must not use physical TP packing."""

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import pytest
from easydel.layers.layouts import _moe

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("source", ["separate", "fused", "per_expert"])
@pytest.mark.parametrize("expert_tensor", [False, True])
def test_gate_up_packing_uses_expert_contraction_layout(monkeypatch, source, expert_tensor):
    monkeypatch.setattr(_moe, "tensor_parallel_size", lambda config, arr=None: 4)
    config = SimpleNamespace(use_expert_tensor_mode=expert_tensor)
    gate = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4)
    up = gate + 1000
    layout = _moe.FusedExpertLayout(
        source_is_fused=source == "fused",
        source_per_expert=2 if source == "per_expert" else None,
    )
    rule = layout.reform_param(config=config)["gate_up_proj.weight$"]
    args = (gate, up)
    if source == "fused":
        args = (torch.cat((gate, up), dim=1),)
    elif source == "per_expert":
        args = (gate[0], up[0], gate[1], up[1])
    fused = rule["fuser"](torch, *args)
    logical = torch.cat((gate.transpose(-1, -2), up.transpose(-1, -2)), dim=-1)
    if expert_tensor:
        torch.testing.assert_close(fused, logical, rtol=0, atol=0)
    else:
        from easydel.layers.layouts._torch_packing import torch_interleave_segments_for_tp

        expected = torch_interleave_segments_for_tp(
            torch, (gate.transpose(-1, -2), up.transpose(-1, -2)), tp_size=4, dim=2
        )
        torch.testing.assert_close(fused, expected, rtol=0, atol=0)
    recovered = rule["inverse_fuser"](torch, fused)
    for actual, expected in zip(recovered, args, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
