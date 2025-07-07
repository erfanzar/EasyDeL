# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class SequenceStatus(IntEnum):
    """Sequence lifecycle states."""

    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    ERROR = 3


@dataclass(frozen=True)
class SequenceMetadata:
    """Immutable sequence metadata."""

    seq_id: int
    block_size: int
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1


@dataclass
class SequenceState:
    """Mutable sequence state with efficient token management."""

    metadata: SequenceMetadata
    prompt_tokens: np.ndarray
    output_tokens: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    def __post_init__(self):
        self.prompt_tokens = np.asarray(self.prompt_tokens, dtype=np.int32)
        self._token_cache: np.ndarray | None = None
        self._cache_valid = False

    @property
    def all_tokens(self) -> np.ndarray:
        """Cached concatenation of prompt and output tokens."""
        if not self._cache_valid:
            if self.output_tokens:
                self._token_cache = np.concatenate([self.prompt_tokens, np.array(self.output_tokens, dtype=np.int32)])
            else:
                self._token_cache = self.prompt_tokens
            self._cache_valid = True
        return self._token_cache

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.metadata.block_size - 1) // self.metadata.block_size

    @property
    def last_block_usage(self) -> int:
        usage = self.num_tokens % self.metadata.block_size
        return self.metadata.block_size if usage == 0 else usage

    def append_token(self, token: int) -> None:
        """Append token and invalidate cache."""
        self.output_tokens.append(token)
        self._cache_valid = False

    def get_block_tokens(self, block_idx: int) -> np.ndarray:
        """Get tokens for a specific block."""
        start = block_idx * self.metadata.block_size
        end = min(start + self.metadata.block_size, self.num_tokens)
        return self.all_tokens[start:end]

    def fork(self) -> SequenceState:
        """Create a copy with shared prompt tokens."""
        return SequenceState(
            metadata=self.metadata,
            prompt_tokens=self.prompt_tokens,  # Shared reference
            output_tokens=self.output_tokens.copy(),
            block_table=self.block_table.copy(),
            status=self.status,
        )

    def __repr__(self) -> str:
        """Provides a concise summary of the sequence's state."""
        return (
            f"SequenceState("
            f"id={self.metadata.seq_id}, "
            f"status={self.status.name}, "
            f"tokens={self.num_tokens} (prompt={len(self.prompt_tokens)}, output={len(self.output_tokens)}), "
            f"blocks={len(self.block_table)}"
            f")"
        )
