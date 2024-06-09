from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "deepseek_configuration": ["DeepseekV2Config"],
    "modeling_deepseek_flax": ["FlaxDeepseekV2ForCausalLM", "FlaxDeepseekV2Model"],
}

if TYPE_CHECKING:
    from .deepseek_configuration import DeepseekV2Config as DeepseekV2Config
    from .modeling_deepseek_flax import (
        FlaxDeepseekV2ForCausalLM as FlaxDeepseekV2ForCausalLM,
        FlaxDeepseekV2Model as FlaxDeepseekV2Model,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
