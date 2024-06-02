from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "gpt_neo_x_configuration": ["GPTNeoXConfig"],
    "modelling_gpt_neo_x_flax": [
        "FlaxGPTNeoXForCausalLM",
        "FlaxGPTNeoXForCausalLMModule",
        "FlaxGPTNeoXModel",
        "FlaxGPTNeoXModule",
    ],
}

if TYPE_CHECKING:
    from .gpt_neo_x_configuration import GPTNeoXConfig
    from .modelling_gpt_neo_x_flax import (
        FlaxGPTNeoXForCausalLM,
        FlaxGPTNeoXForCausalLMModule,
        FlaxGPTNeoXModel,
        FlaxGPTNeoXModule,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)