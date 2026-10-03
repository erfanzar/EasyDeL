# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""CPU-safe config-default checks for GPT-OSS.

The released ``openai/gpt-oss-20b`` and ``openai/gpt-oss-120b`` ``config.json``
files both ship ``router_aux_loss_coef: 0.9`` (the HF class default of 0.001 is
not what the released models use). EasyDeL's other GPT-OSS defaults mirror the
released 120b checkpoint, so the aux-loss default intentionally does too, while
an explicit checkpoint value must still win.
"""

import pytest

pytestmark = pytest.mark.cpu_ok


def test_gpt_oss_router_aux_loss_coef_default_and_override():
    from easydel.modules.gpt_oss import GptOssConfig

    assert GptOssConfig().router_aux_loss_coef == 0.9
    assert GptOssConfig(router_aux_loss_coef=0.001).router_aux_loss_coef == 0.001
