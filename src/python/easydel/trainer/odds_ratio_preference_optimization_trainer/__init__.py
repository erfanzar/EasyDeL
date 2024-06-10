from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "orpo_trainer": ["ORPOTrainer", "ORPOTrainerOutput"],
}

if TYPE_CHECKING:
    from .orpo_trainer import (
        ORPOTrainer as ORPOTrainer,
        ORPOTrainerOutput as ORPOTrainerOutput,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
