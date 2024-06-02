from typing import TYPE_CHECKING
from ...utils.lazy_import import _LazyModule

_import_structure = {
    "mosaic_configuration": ["MptConfig", "MptAttentionConfig"],
    "modelling_mpt_flax": [
        "FlaxMptForCausalLM",
        "FlaxMptForCausalLMModule",
        "FlaxMptModel",
        "FlaxMptModule"
    ]
}
if TYPE_CHECKING:
    from .mosaic_configuration import (
        MptConfig as MptConfig,
        MptAttentionConfig as MptAttentionConfig
    )
    from .modelling_mpt_flax import (
        FlaxMptForCausalLM,
        FlaxMptForCausalLMModule,
        FlaxMptModel,
        FlaxMptModule
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
