from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "t5_configuration": ["T5Config"],
    "modelling_t5_flax": [
        "FlaxT5ForConditionalGeneration",
        "FlaxT5ForConditionalGenerationModule",
        "FlaxT5Model",
        "FlaxT5Module",
    ],
}

if TYPE_CHECKING:
    from .t5_configuration import T5Config as T5Config
    from .modelling_t5_flax import (
        FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration,
        FlaxT5ForConditionalGenerationModule as FlaxT5ForConditionalGenerationModule,
        FlaxT5Model as FlaxT5Model,
        FlaxT5Module as FlaxT5Module,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
