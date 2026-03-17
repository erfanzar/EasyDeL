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

"""Test utilities for EasyDeL model testing.

This package provides modular utilities for testing EasyDeL models against
HuggingFace transformers implementations.
"""

from .comparators import (
    ComparisonResult,
    compare_hidden_states,
    compare_logits,
    compare_loss,
    compare_outputs,
    print_comparison_result,
)
from .input_generators import (
    make_audio_inputs,
    make_classification_inputs,
    make_image_classification_inputs,
    make_input_ids,
    make_qwen_vlm_inputs,
    make_seq2seq_inputs,
    make_text_inputs,
    make_vlm_inputs,
)
from .model_factory import (
    cleanup_models,
    create_base_config,
    create_ed_model,
    create_ed_model_only,
    create_hf_model,
    create_model_pair,
    get_hf_model_from_hub,
    get_module_classes,
    setup_config,
    transfer_weights,
)
from .task_testers import (
    BaseModuleTester,
    BaseTester,
    CausalLMTester,
    EasyDeLOnlyTester,
    EmbeddingTester,
    Seq2SeqTester,
    SequenceClassificationTester,
    TestResult,
    VisionLanguageTester,
)

__all__ = [
    # Task testers
    "BaseModuleTester",
    "BaseTester",
    "CausalLMTester",
    # Comparators
    "ComparisonResult",
    "EasyDeLOnlyTester",
    "EmbeddingTester",
    "Seq2SeqTester",
    "SequenceClassificationTester",
    "TestResult",
    "VisionLanguageTester",
    # Model factory
    "cleanup_models",
    "compare_hidden_states",
    "compare_logits",
    "compare_loss",
    "compare_outputs",
    "create_base_config",
    "create_ed_model",
    "create_ed_model_only",
    "create_hf_model",
    "create_model_pair",
    "get_hf_model_from_hub",
    "get_module_classes",
    # Input generators
    "make_audio_inputs",
    "make_classification_inputs",
    "make_image_classification_inputs",
    "make_input_ids",
    "make_qwen_vlm_inputs",
    "make_seq2seq_inputs",
    "make_text_inputs",
    "make_vlm_inputs",
    "print_comparison_result",
    "setup_config",
    "transfer_weights",
]
