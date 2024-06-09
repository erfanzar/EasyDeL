from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "modelling_output": ["ORPOTrainerOutput"],
    "orpo_trainer": ["ORPOTrainer"],
}

if TYPE_CHECKING:
    from .modelling_output import ORPOTrainerOutput as ORPOTrainerOutput
    from .orpo_trainer import ORPOTrainer as ORPOTrainer
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
