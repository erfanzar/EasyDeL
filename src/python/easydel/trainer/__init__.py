from ..utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "training_configurations": [
        "TrainArguments",
        "EasyDeLXRapTureConfig"
    ],
    "causal_language_model_trainer": [
        "create_casual_language_model_evaluation_step",
        "create_casual_language_model_train_step",
        "CausalLanguageModelTrainer",
        "CausalLMTrainerOutput"
    ],
    "vision_causal_language_model_trainer": [
        "VisionCausalLanguageModelStepOutput",
        "VisionCausalLanguageModelTrainer",
        "create_vision_casual_language_model_evaluation_step",
        "create_vision_casual_language_model_train_step",
        "VisionCausalLMTrainerOutput"
    ],
    "direct_oreference_optimization_trainer": [
        "DPOTrainer",
        "create_dpo_eval_function",
        "create_concatenated_forward",
        "create_dpo_train_function",
        "concatenated_dpo_inputs"
    ],
    "supervised_fine_tuning_trainer": [
        "SFTTrainer"
    ],
    "utils": [
        "create_constant_length_dataset",
        "get_formatting_func_from_dataset",
        "conversations_formatting_function",
        "instructions_formatting_function"
    ],
    "odds_ratio_preference_optimization_trainer": [
        "ORPOTrainer",
        "create_orpo_step_function",
        "create_concatenated_forward",
        "odds_ratio_loss",
        "ORPOTrainerOutput"
    ]
}

if TYPE_CHECKING:
    from .training_configurations import (
        TrainArguments,
        EasyDeLXRapTureConfig
    )
    from .causal_language_model_trainer import (
        create_casual_language_model_evaluation_step,
        create_casual_language_model_train_step,
        CausalLanguageModelTrainer,
        CausalLMTrainerOutput
    )
    from .vision_causal_language_model_trainer import (
        VisionCausalLanguageModelStepOutput,
        VisionCausalLanguageModelTrainer,
        create_vision_casual_language_model_evaluation_step,
        create_vision_casual_language_model_train_step,
        VisionCausalLMTrainerOutput
    )
    from .direct_oreference_optimization_trainer import (
        DPOTrainer,
        create_dpo_eval_function,
        create_concatenated_forward,
        create_dpo_train_function,
        concatenated_dpo_inputs
    )
    from .supervised_fine_tuning_trainer import SFTTrainer
    from .utils import (
        create_constant_length_dataset,
        get_formatting_func_from_dataset,
        conversations_formatting_function,
        instructions_formatting_function
    )
    from .odds_ratio_preference_optimization_trainer import (
        ORPOTrainer,
        create_orpo_step_function,
        create_concatenated_forward,
        odds_ratio_loss,
        ORPOTrainerOutput
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
