from ..utils.lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "easydel_transform": [
        "huggingface_to_easydel",
        "easystate_to_huggingface_model",
        "easystate_to_torch"
    ]
}

if TYPE_CHECKING:
    from .easydel_transform import (
        huggingface_to_easydel,
        easystate_to_huggingface_model,
        easystate_to_torch
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
