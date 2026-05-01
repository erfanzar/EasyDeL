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

from __future__ import annotations

import typing as tp


def _spec_axis(spec: tp.Any, index: int) -> tp.Any:
    if spec is None:
        return None
    try:
        if index >= len(spec):
            return None
        axis = spec[index]
    except Exception:
        return None
    return None if axis == () else axis


def align_mask_info_to_qkv_specs(
    mask_info: tp.Any,
    *,
    query_spec: tp.Any,
    key_spec: tp.Any | None = None,
    layout: tp.Literal["bthd", "bhtd"] = "bthd",
) -> tp.Any:
    if mask_info is None:
        return None

    if layout == "bhtd":
        sequence_index = 2
        head_index = 1
    else:
        sequence_index = 1
        head_index = 2

    replace = getattr(mask_info, "replace", None)
    if not callable(replace):
        return mask_info

    key_spec = query_spec if key_spec is None else key_spec
    return replace(
        batch_axis_name=_spec_axis(query_spec, 0),
        qheads_axis_name=_spec_axis(query_spec, head_index),
        kvheads_axis_name=_spec_axis(key_spec, head_index),
        sequence_axis_name=_spec_axis(query_spec, sequence_index),
    )
