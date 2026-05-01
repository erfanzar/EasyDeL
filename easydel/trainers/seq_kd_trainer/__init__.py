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

"""Sequence-level Knowledge Distillation (SeqKD) trainer module.

SeqKD is a black-box distillation method (Kim & Rush, 2016): a teacher
generates text completions from prompts and the student is trained on
those completions with standard causal-LM cross-entropy loss. Because
no teacher logits or hidden states are used, this trainer also supports
API-based teachers via a callable.

Public symbols:
    - :class:`SeqKDConfig`: Hyperparameters for prompt/completion lengths
      and teacher sampling.
    - :class:`SeqKDTrainer`: Trainer implementing teacher-then-CE
      distillation.
"""

from .seq_kd_config import SeqKDConfig
from .seq_kd_trainer import SeqKDTrainer

__all__ = ("SeqKDConfig", "SeqKDTrainer")
