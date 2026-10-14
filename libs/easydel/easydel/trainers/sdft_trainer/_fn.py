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
"""Self-distillation fine-tuning trainer aliases."""

from __future__ import annotations

import typing as tp


def _zero_reward_func(completions: tp.Sequence[object], **_: object) -> list[float]:
    """Default reward hook for SDFT when users only need self-distillation.

    SDFT can reuse generation-oriented trainer plumbing even when no external
    reward model is involved. Returning one zero per completion keeps the batch
    shape valid while leaving the self-distillation loss to drive training.
    """
    return [0.0] * len(completions)
