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

import numpy as np

from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor


class _Metadata:
    version = "v3"
    page_size = 32
    max_num_pages_per_req = 64

    def __init__(self, num_pages: int) -> None:
        self.num_pages = int(num_pages)


def test_kv_prepare_signature_changes_with_static_page_shape():
    old_kv = {"kv": np.zeros((124, 32, 2, 2, 8), dtype=np.float32)}
    new_kv = {"kv": np.zeros((97, 32, 2, 2, 8), dtype=np.float32)}

    old_signature = ModelStepExecutor._kv_prepare_signature(old_kv, _Metadata(num_pages=124))
    new_signature = ModelStepExecutor._kv_prepare_signature(new_kv, _Metadata(num_pages=97))

    assert old_signature != new_signature


def test_kv_prepare_signature_is_stable_for_equivalent_cache_trees():
    left = {
        "k": np.zeros((97, 32, 2, 8), dtype=np.float32),
        "v": np.zeros((97, 32, 2, 8), dtype=np.float32),
    }
    right = {
        "k": np.zeros((97, 32, 2, 8), dtype=np.float32),
        "v": np.zeros((97, 32, 2, 8), dtype=np.float32),
    }

    assert ModelStepExecutor._kv_prepare_signature(left, _Metadata(num_pages=97)) == (
        ModelStepExecutor._kv_prepare_signature(right, _Metadata(num_pages=97))
    )
