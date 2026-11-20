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

"""``auto_logits_chunk_size``: the distillation ``logits_chunk_size`` a tensor-parallel run gets when it sets none."""

import pytest
from easydel.trainers.distillation_trainer.distillation_trainer import AUTO_LOGITS_CHUNK_SIZE, auto_logits_chunk_size

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("seq_len", [1, 1000, 4096, 4097, 8192, 8193, 32000, 131072, 131073])
def test_chunks_cover_the_sequence_with_little_padding(seq_len):
    chunk = auto_logits_chunk_size(seq_len)
    num_chunks = -(-seq_len // chunk)
    assert chunk <= AUTO_LOGITS_CHUNK_SIZE
    assert num_chunks == -(-seq_len // AUTO_LOGITS_CHUNK_SIZE)
    if num_chunks == 1:
        assert chunk == seq_len
    else:
        assert chunk % 128 == 0
        assert num_chunks * chunk - seq_len < 128 * num_chunks
