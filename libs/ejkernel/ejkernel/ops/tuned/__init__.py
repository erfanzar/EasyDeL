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

"""Shipped, pre-tuned kernel configurations, stored as an indexed table.

    from ejkernel.ops.tuned import tuned_choice

    choice = tuned_choice("grouped_matmul",
                          dtypes={"lhs": lhs.dtype, "rhs": rhs.dtype},
                          shape={"m": m, "k": k, "n": n, "e": groups})
    if choice is not None:
        platform, cfg = choice.platform, choice.config

Entries carry the winning **platform** as well as its config, plus the runner-up
and its timing, so which backend is faster for a given shape is a recorded
measurement rather than a static rule.
"""

from ._lookup import current_device_kind, set_tuned_store, tuned_choice, tuned_store
from ._store import (
    DEFAULT_DB_NAME,
    SCHEMA_VERSION,
    TunedEntry,
    TunedStore,
    bucket,
    default_db_path,
    dtype_signature,
    merge,
    open_for_write,
    shape_signature,
    upsert,
)

__all__ = (
    "DEFAULT_DB_NAME",
    "SCHEMA_VERSION",
    "TunedEntry",
    "TunedStore",
    "bucket",
    "current_device_kind",
    "default_db_path",
    "dtype_signature",
    "merge",
    "open_for_write",
    "set_tuned_store",
    "shape_signature",
    "tuned_choice",
    "tuned_store",
    "upsert",
)
