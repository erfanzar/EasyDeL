from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "arctic_configuration": ["ArcticConfig"],
    "modelling_arctic_flax": [
        "FlaxArcticForCausalLM",
        "FlaxArcticModel",
    ]
}

if TYPE_CHECKING:
    from .arctic_configuration import ArcticConfig
    from .modelling_arctic_flax import (
        FlaxArcticForCausalLM,
        FlaxArcticModel
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
