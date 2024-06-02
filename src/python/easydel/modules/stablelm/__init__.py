from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "stablelm_configuration": ["StableLmConfig"],
    "modelling_stablelm_flax": [
        "FlaxStableLmForCausalLM",
        "FlaxStableLmModel"
    ],
}

if TYPE_CHECKING:
    from .stablelm_configuration import StableLmConfig
    from .modelling_stablelm_flax import (
        FlaxStableLmForCausalLM,
        FlaxStableLmModel
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)