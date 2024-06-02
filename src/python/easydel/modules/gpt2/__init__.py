from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "gpt2_configuration": ["GPT2Config"],
    "modelling_gpt2_flax": [
        "FlaxGPT2LMHeadModel",
        "FlaxGPT2LMHeadModule",
        "FlaxGPT2Model",
        "FlaxGPT2Module",
    ],
}

if TYPE_CHECKING:
    from .gpt2_configuration import GPT2Config
    from .modelling_gpt2_flax import (
        FlaxGPT2LMHeadModel,
        FlaxGPT2LMHeadModule,
        FlaxGPT2Model,
        FlaxGPT2Module,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)