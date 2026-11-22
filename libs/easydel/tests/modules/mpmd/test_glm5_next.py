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

"""Scheduled MPMD runtime coverage for GLM-5-Next's hybrid text model.

Reuse the shared PP mesh and schedule fixtures unchanged, not the SPMD
GLM-5-Next fixture that replaces the mesh with a PP=1 FSDP mesh. The shared
scheduled generation tester currently checks compute_loss on a short prompt;
actual autoregressive HybridCache generation is covered by the SPMD suite.
"""

import easydel as ed
import pytest

from tests.modules.mpmd._scheduler_utils import GENERATION_SCHEDULE_KIND, LOSS_SCHEDULE_KINDS
from tests.modules.spmd.test_glm5_next import _tiny_kwargs
from tests.modules.test_utils import CausalLMTester


@pytest.fixture
def glm5_config(small_model_config):
    """Keep KDA, DSA, mHC and MoE in the shared four-layer PP recipe."""
    config = ed.Glm5NextTextConfig(**_tiny_kwargs(small_model_config))
    config.moe_force_xla_gmm = True
    return config


class TestGlm5Next:
    """Execute registered GLM-5-Next modules through the scheduled tester."""

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_causal_lm(self, glm5_config, small_model_config, mpmd_schedule_kind):
        """Run finite scalar loss under flat, virtual and bidirectional schedules."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="glm5_next_text",
            hf_class=None,
            task=ed.TaskType.CAUSAL_LM,
            config=glm5_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GLM-5-Next MPMD CAUSAL_LM failed: {result.error_message}"
        assert result.extra_info["scheduled_loss"], "expected the scheduled MPMD loss path"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_generation(self, glm5_config, small_model_config, mpmd_schedule_kind):
        """Exercise the shared generation-schedule short-prompt loss contract."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="glm5_next_text",
            hf_class=None,
            task=ed.TaskType.CAUSAL_LM,
            config=glm5_config,
            small_model_config=small_model_config,
            max_new_tokens=4,
        )
        assert result.success, f"GLM-5-Next MPMD generation schedule failed: {result.error_message}"
        assert result.extra_info["scheduled_loss"], "expected the scheduled MPMD loss path"
        assert result.extra_info["scheduled_generation"], "expected the shared generation-schedule path"
