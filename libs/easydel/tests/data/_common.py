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

"""Pure-Python fixtures shared by the data tests (no array computation)."""

from __future__ import annotations

from easydel.data.core.protocols import ShardedDataSource, ShardInfo


class ListSource(ShardedDataSource[dict]):
    """In-memory sharded source; optionally advertises exact ``ShardInfo`` row counts."""

    def __init__(self, shards: dict[str, list[dict]], with_info: bool = False):
        self._shards = shards
        self._with_info = with_info

    @property
    def shard_names(self):
        return list(self._shards)

    def num_shards(self) -> int:
        return len(self._shards)

    def open_shard(self, shard_name: str):
        return iter([dict(row) for row in self._shards[shard_name]])

    def get_shard_info(self, shard_name: str):
        if not self._with_info:
            return None
        return ShardInfo(
            shard_id=list(self._shards).index(shard_name),
            shard_name=shard_name,
            num_rows=len(self._shards[shard_name]),
        )


class FakeTokenizer:
    """Deterministic char-ordinal tokenizer that records how it was called."""

    def __init__(self):
        self.calls = 0
        self.kwargs_seen: list[dict] = []

    def __getstate__(self):
        """Exclude call-recording instrumentation from the effective tokenizer state."""
        return {key: value for key, value in self.__dict__.items() if key not in ("calls", "kwargs_seen")}

    def __setstate__(self, state):
        """Restore tokenizer settings with fresh call-recording instrumentation."""
        self.__init__()
        self.__dict__.update(state)

    def __call__(
        self,
        text,
        max_length=None,
        truncation=True,
        padding=False,
        add_special_tokens=True,
        return_attention_mask=True,
        **kwargs,
    ):
        self.calls += 1
        self.kwargs_seen.append(dict(kwargs))
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        ids = [[ord(c) for c in t][:max_length] for t in texts]
        if padding is True or padding == "longest":
            longest = max(len(i) for i in ids)
            ids = [i + [0] * (longest - len(i)) for i in ids]
        out = {"input_ids": ids}
        if return_attention_mask:
            out["attention_mask"] = [[1] * len(i) for i in ids]
        if single:
            out = {key: value[0] for key, value in out.items()}
        return out
