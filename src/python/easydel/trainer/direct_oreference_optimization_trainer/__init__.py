from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "modelling_output": ["DPOTrainerOutput"],
    "fwd_bwd_functions": [
        "create_dpo_train_function",
        "create_dpo_eval_function",
        "create_concatenated_forward",
        "get_batch_log_probs",
        "concatenated_inputs"
    ],
    "dpo_trainer": ["DPOTrainer"]
}

if TYPE_CHECKING:
    from .modelling_output import DPOTrainerOutput
    from .fwd_bwd_functions import (
        create_dpo_train_function,
        create_dpo_eval_function,
        create_concatenated_forward,
        get_batch_log_probs,
        concatenated_inputs
    )
    from .dpo_trainer import DPOTrainer
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
