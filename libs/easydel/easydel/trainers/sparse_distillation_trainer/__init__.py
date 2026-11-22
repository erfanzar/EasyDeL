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

"""Sparse (gray-box) knowledge distillation trainer module.

In sparse distillation only the teacher's *top-k* logits / log-probabilities
are made available (e.g. emitted by an inference API). The student is
trained against the teacher's renormalized top-k distribution via a partial
KL term, optionally combined with a hard-label cross-entropy term.

Public symbols:
    - :class:`SparseDistillationConfig`: Hyperparameters for sparse
      distillation training.
    - :class:`SparseDistillationTrainer`: Trainer applying the partial-KL
      objective.
"""

from .sparse_distillation_config import SparseDistillationConfig
from .sparse_distillation_trainer import SparseDistillationTrainer

__all__ = ("SparseDistillationConfig", "SparseDistillationTrainer")
