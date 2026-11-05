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
# ---- end section: kv ----

# ---- begin section: engine ----
# ---- end section: engine ----
