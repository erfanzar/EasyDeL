from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "modelling_output": ["VisionCausalLMTrainerOutput"],
    "fwd_bwd_functions": [
        "create_vision_casual_language_model_train_step",
        "create_vision_casual_language_model_evaluation_step",
        "VisionCausalLanguageModelStepOutput"
    ],
    "vision_causal_language_model_trainer": ["VisionCausalLanguageModelTrainer"]
}

if TYPE_CHECKING:
    from .modelling_output import VisionCausalLMTrainerOutput
    from .fwd_bwd_functions import (
        create_vision_casual_language_model_train_step,
        create_vision_casual_language_model_evaluation_step,
        VisionCausalLanguageModelStepOutput
    )
    from .vision_causal_language_model_trainer import VisionCausalLanguageModelTrainer
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
