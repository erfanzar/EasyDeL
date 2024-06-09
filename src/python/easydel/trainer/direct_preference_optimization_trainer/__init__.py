from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "modelling_output": ["DPOTrainerOutput"],
    "dpo_trainer": ["DPOTrainer"]
}

if TYPE_CHECKING:
    from .modelling_output import DPOTrainerOutput as DPOTrainerOutput
    from .dpo_trainer import DPOTrainer as DPOTrainer
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
