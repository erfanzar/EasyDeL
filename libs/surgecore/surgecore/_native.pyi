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

"""Typed stubs of the compiled extension module ``surgecore._native``.

One delimited section per Rust module (payload, kv, engine). Edit only between a section's begin and end markers.
"""

import numpy as np
import numpy.typing as npt

__version__: str

# ---- begin section: payload ----

class HostPayloadBuilder:
    """Fills one batch preparer's preallocated staging arrays with a forward window's step inputs.

    ``buffers`` maps staging-array names to writeable, C-contiguous numpy arrays (int32, or float32 for
    ``packed_f32_padded``). They are registered once; every ``build`` writes into them in place with the GIL released.
    """

    def __init__(
        self,
        *,
        buffers: dict[str, npt.NDArray[np.generic]],
        max_num_reqs: int,
        num_reqs_max_model_len: int,
        data_parallel_size: int,
        use_slot_mapping: bool,
        use_request_distribution: bool,
        plain_recurrent_slots: bool,
        recurrent_slot_group_size: int,
        enable_spec_recurrent_commit: bool,
        enable_dp_local_page_path: bool,
        page_size: int,
        max_num_pages_per_req: int,
        slices_per_page: int,
        max_padded_slices: int,
    ) -> None: ...
    def build(
        self,
        num_tokens_static: int,
        scheduled_full: npt.ArrayLike,
        active_mask_full: npt.ArrayLike,
        token_ids: npt.ArrayLike,
        num_computed_tokens: npt.ArrayLike,
        temperature: npt.ArrayLike,
        top_p: npt.ArrayLike,
        top_k: npt.ArrayLike,
        min_p: npt.ArrayLike,
        frequency_penalties: npt.ArrayLike,
        presence_penalties: npt.ArrayLike,
        repetition_penalties: npt.ArrayLike,
        page_table: npt.ArrayLike,
        padded_num_reqs_in: int,
        window_row_indices: npt.ArrayLike | None = None,
        recurrent_slot_indices: npt.ArrayLike | None = None,
        spec_recurrent_commit: npt.ArrayLike | None = None,
        page_table_version: int | None = None,
        cached_page_table_version: int | None = None,
        cached_rows_to_copy: int | None = None,
        num_pages: int = 0,
        metadata_data_parallel_size: int = 1,
    ) -> tuple[int, int, int, bool, int]:
        """Fill the staging arrays for one window.

        Returns:
            ``(num_requests, padded_num_reqs, rows_to_copy, reuse_pages_tables, total_pages)``.

        Raises:
            ValueError: The window does not fit the bucket, the DP layout or the token history.
        """
        ...

def request_distribution_prefix(
    scheduled: npt.ArrayLike, num_computed_tokens: npt.ArrayLike, num_requests: int
) -> tuple[int, int, int]:
    """``(decode_end, prefill_end, num_requests)`` of eSurge's positional request-distribution contract."""
    ...

# ---- end section: payload ----

# ---- begin section: kv ----

class KvCore:
    """One KV page pool with its prefix cache and the page state of every request that touched it.

    ``groups`` holds one ``(kind, page_size, window_or_chunk, type_index)`` per cache group: kind 0 full attention,
    1 sliding window, 2 chunked local attention. Requests are addressed by the slot ``open`` returns.
    """

    def __init__(
        self, *, num_pages: int, groups: list[tuple[int, int, int, int]], max_model_len: int, caching: bool, eagle: bool
    ) -> None: ...
    def open(self, extra: int) -> int: ...
    def num_keys(self, slot: int) -> int: ...
    def keys(self, slot: int, tokens_: list[int] | None) -> list[int]: ...
    def lookup(
        self,
        slot: int,
        tokens_: list[int] | None,
        no_hit: bool,
        max_len: int,
        hint: int | None = None,
        dp: int | None = None,
    ) -> tuple[list[list[int]], int]: ...
    def probe(self, slot: int | None, tokens_: list[int] | None, extra: int, max_len: int) -> int: ...
    def allocate(
        self,
        slot: int,
        computed: int,
        n_new: int,
        n_hit: int,
        hits: list[list[int]] | None,
        lookahead: int,
        hint: int | None = None,
        dp: int | None = None,
    ) -> list[list[int]] | None: ...
    def needed(
        self, slot: int | None, num_tokens: int, hits: list[list[int]] | None = None, max_step: int | None = None
    ) -> int: ...
    def free(self, slot: int) -> bool: ...
    def forget_keys(self, slot: int) -> bool: ...
    def publish(
        self, slot: int, n: int, tokens_: list[int] | None = None, offset: int = 0
    ) -> tuple[int, int] | None: ...
    def page_ids(self, slot: int) -> list[list[int]]: ...
    def common_prefix(self, slot: int | None, num_scheduled: int) -> list[int]: ...
    def release_tail(self, slot: int, keep_tokens: int) -> int: ...
    def reset(self) -> bool: ...
    @property
    def num_pages(self) -> int: ...
    @property
    def num_free(self) -> int: ...
    @property
    def usage(self) -> float: ...
    @property
    def num_cached(self) -> int: ...
    @property
    def caching(self) -> bool: ...
    @property
    def num_open(self) -> int: ...
    def free_order(self) -> list[int]: ...
    def refs(self) -> list[int]: ...
    def check(self) -> None: ...

class SnapshotIndex:
    """Recurrent-snapshot slots: pending until committed; ready slots are recycled least recently used first."""

    def __init__(self, num_slots: int, slot_offset: int = 0) -> None: ...
    @property
    def num_slots(self) -> int: ...
    @property
    def slot_offset(self) -> int: ...
    def reset(self) -> None: ...
    def contains(self, key: int) -> bool: ...
    def is_ready(self, key: int) -> bool: ...
    def lookup(self, key: int) -> int | None: ...
    def reserve(self, key: int, exclude: object = None) -> int | None: ...
    def cancel(self, slot: int) -> None: ...
    def commit(self, slot: int) -> None: ...
    def state(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]: ...

# ---- end section: kv ----

# ---- begin section: engine ----
# ---- end section: engine ----
