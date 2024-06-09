from ..utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "causal_language_model_trainer": [
        "CausalLanguageModelTrainer",
        "CausalLMTrainerOutput",
    ],
    "supervised_fine_tuning_trainer": ["SFTTrainer"],
    "vision_causal_language_model_trainer": [
        "VisionCausalLanguageModelStepOutput",
        "VisionCausalLanguageModelTrainer",
        "VisionCausalLMTrainerOutput",
    ],
    "odds_ratio_preference_optimization_trainer": [
        "ORPOTrainer",
        "ORPOTrainerOutput",
    ],
    "direct_preference_optimization_trainer": ["DPOTrainer", "DPOTrainerOutput"],
    "training_configurations": ["TrainArguments", "EasyDeLXRapTureConfig"],
    "utils": [
        "create_constant_length_dataset",
        "get_formatting_func_from_dataset",
        "conversations_formatting_function",
        "instructions_formatting_function",
    ],
}

if TYPE_CHECKING:
    from .training_configurations import (
        TrainArguments as TrainArguments,
        EasyDeLXRapTureConfig as EasyDeLXRapTureConfig,
    )
    from .causal_language_model_trainer import (
        CausalLanguageModelTrainer as CausalLanguageModelTrainer,
        CausalLMTrainerOutput as CausalLMTrainerOutput,
    )
    from .vision_causal_language_model_trainer import (
        VisionCausalLanguageModelTrainer as VisionCausalLanguageModelTrainer,
        VisionCausalLMTrainerOutput as VisionCausalLMTrainerOutput,
    )
    from .direct_preference_optimization_trainer import (
        DPOTrainer as DPOTrainer,
        DPOTrainerOutput as DPOTrainerOutput,
    )
    from .supervised_fine_tuning_trainer import SFTTrainer as SFTTrainer
    from .utils import (
        create_constant_length_dataset as create_constant_length_dataset,
        get_formatting_func_from_dataset as get_formatting_func_from_dataset,
        conversations_formatting_function as conversations_formatting_function,
        instructions_formatting_function as instructions_formatting_function,
    )
    from .odds_ratio_preference_optimization_trainer import (
        ORPOTrainer as ORPOTrainer,
        ORPOTrainerOutput as ORPOTrainerOutput,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
