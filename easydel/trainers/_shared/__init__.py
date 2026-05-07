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

"""Cross-trainer helpers shared by preference / RLHF trainers.

Exposes the small set of utilities that DPO, CPO, KTO, ORPO, and BCO
all need but cannot tidily inherit because their config dataclasses
have their own MRO constraints:

* :func:`normalize_logprob_vocab_chunk_size` -- coerces the
  ``logprob_vocab_chunk_size`` config field to a positive int or
  ``None``, used by every preference config's ``__post_init__``.
* :func:`apply_paired_truncation` -- truncates parallel sequence
  tensors (``input_ids`` / ``attention_mask`` / ``loss_mask``) along
  axis 1 with a configurable keep-end / keep-start mode.
* :func:`gather_multimodal_kwargs` -- collects optional vision-tower
  inputs (``pixel_values``, ``pixel_attention_mask``, ``image_sizes``)
  and the MoE auxiliary-loss flag from a batch dict.
"""

from .preference_config_helpers import normalize_logprob_vocab_chunk_size
from .preference_forward_helpers import (
    apply_paired_truncation,
    gather_multimodal_kwargs,
)

__all__ = [
    "apply_paired_truncation",
    "gather_multimodal_kwargs",
    "normalize_logprob_vocab_chunk_size",
]
