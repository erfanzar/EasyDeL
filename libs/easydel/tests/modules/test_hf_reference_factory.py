# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""The HF reference factory must not hand uninitialized router biases to parity tests.

Hub remote code for DeepSeek-V3 / Kimi-VL allocates ``e_score_correction_bias`` with ``torch.empty`` and never
initializes it. Deep into a long test session the allocator returns garbage, which turned the scheduled MPMD loss
into NaN. The fixture here poisons that bias explicitly to stand in for recycled memory.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from .test_utils.model_factory import create_hf_model


class _Router(nn.Module):
    def __init__(self, n: int, fill: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n, 4))
        self.e_score_correction_bias = nn.Parameter(torch.full((n,), fill))


class _PoisonedRemoteModel(nn.Module):
    """Stand-in for remote code whose router bias comes straight from ``torch.empty``."""

    fill = float("nan")

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = _Router(4, self.fill)
        self.register_buffer("causal_mask", torch.full((2, 2), float("-inf")))


def test_router_bias_is_zeroed_and_other_weights_untouched():
    model = create_hf_model(_PoisonedRemoteModel, SimpleNamespace(model_type="poisoned"))

    assert torch.equal(model.gate.e_score_correction_bias, torch.zeros(4))
    assert torch.equal(model.gate.weight, torch.ones(4, 4))
    # -inf mask buffers are legitimate and must not be rejected or rewritten.
    assert torch.isinf(model.causal_mask).all()


def test_non_finite_parameters_elsewhere_fail_loudly():
    class _Broken(_PoisonedRemoteModel):
        def __init__(self, config):
            super().__init__(config)
            self.proj = nn.Parameter(torch.full((3,), float("nan")))

    with pytest.raises(AssertionError, match="non-finite"):
        create_hf_model(_Broken, SimpleNamespace(model_type="broken"))
