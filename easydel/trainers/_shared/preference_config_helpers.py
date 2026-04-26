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

"""Tiny helpers shared by preference-trainer config dataclasses (DPO/CPO/KTO/ORPO/BCO).

A full ``PreferenceOptimizationConfigMixin`` would require non-trivial dataclass
MRO surgery (each config is registered via ``@Registry.register`` and needs
field defaults to come in a specific order), so we keep field declarations
local to each config but centralise the literally-identical normalisation
logic that previously lived in five copies of ``__post_init__``.
"""

from __future__ import annotations


def normalize_logprob_vocab_chunk_size(value: int | None) -> int | None:
    """Coerce a vocab-chunk-size config value to a positive ``int`` or ``None``.

    The five preference configs all share the same rule: ``None`` and
    non-positive values disable chunking; positive values are kept as-is.
    Centralising it here means a future change to chunk-size semantics only
    touches one file.
    """
    if value is None:
        return None
    coerced = int(value)
    return coerced if coerced > 0 else None


__all__ = ["normalize_logprob_vocab_chunk_size"]
