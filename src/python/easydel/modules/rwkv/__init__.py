from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "rwkv_configuration": ["RwkvConfig"],
    "modelling_rwkv_flax": ["FlaxRwkvForCausalLM", "FlaxRwkvModel"],
}

if TYPE_CHECKING:
    from .rwkv_configuration import RwkvConfig as RwkvConfig
    from .modelling_rwkv_flax import (
        FlaxRwkvForCausalLM as FlaxRwkvForCausalLM,
        FlaxRwkvModel as FlaxRwkvModel,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
