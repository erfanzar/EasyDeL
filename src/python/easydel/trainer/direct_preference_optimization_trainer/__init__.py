from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "dpo_trainer": ["DPOTrainer", "DPOTrainerOutput"],
}

if TYPE_CHECKING:
    from .dpo_trainer import (
        DPOTrainer as DPOTrainer,
        DPOTrainerOutput as DPOTrainerOutput,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
