from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "fwd_bwd_functions": [
        "create_orpo_step_function",
        "create_concatenated_forward",
        "odds_ratio_loss"
    ],
    "modelling_output": ["ORPOTrainerOutput"],
    "orpo_trainer": ["ORPOTrainer"]
}

if TYPE_CHECKING:
    from .fwd_bwd_functions import create_orpo_step_function, create_concatenated_forward, odds_ratio_loss
    from .modelling_output import ORPOTrainerOutput
    from .orpo_trainer import ORPOTrainer
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
