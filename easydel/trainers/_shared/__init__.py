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

"""Cross-trainer helpers shared by preference / RLHF trainers."""

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
