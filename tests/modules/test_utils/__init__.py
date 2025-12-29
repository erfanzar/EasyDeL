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
