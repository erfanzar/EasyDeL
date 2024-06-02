from ...utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "configuration_qwen2_moe": ["Qwen2MoeConfig"],
    "modeling_qwen2_moe_flax": [
        "FlaxQwen2MoeForCausalLM",
        "FlaxQwen2MoeModel"
    ]
}

if TYPE_CHECKING:
    from .configuration_qwen2_moe import Qwen2MoeConfig
    from .modeling_qwen2_moe_flax import (
        FlaxQwen2MoeForCausalLM,
        FlaxQwen2MoeModel
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)