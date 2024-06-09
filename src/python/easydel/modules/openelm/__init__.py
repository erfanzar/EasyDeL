from typing import TYPE_CHECKING
from ...utils.lazy_import import _LazyModule

_import_structure = {
    "openelm_configuration": ["OpenELMConfig"],
    "modelling_openelm_flax": [
        "FlaxOpenELMModel",
        "FlaxOpenELMForCausalLM",
    ],
}
if TYPE_CHECKING:
    from .openelm_configuration import OpenELMConfig as OpenELMConfig
    from .modelling_openelm_flax import (
        FlaxOpenELMForCausalLM as FlaxOpenELMForCausalLM,
        FlaxOpenELMModel as FlaxOpenELMModel,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
