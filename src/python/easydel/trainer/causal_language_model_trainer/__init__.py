from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "causal_language_model_trainer": [
        "CausalLanguageModelTrainer",
        "CausalLMTrainerOutput"
    ],
    "fwd_bwd_functions": [
        "create_casual_language_model_train_step",
        "create_casual_language_model_evaluation_step"
    ]
}

if TYPE_CHECKING:
    from .causal_language_model_trainer import (
        CausalLanguageModelTrainer,
        CausalLMTrainerOutput
    )
    from .fwd_bwd_functions import (
        create_casual_language_model_train_step,
        create_casual_language_model_evaluation_step
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)