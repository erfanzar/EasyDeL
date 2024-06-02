from .lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "checker": [
        "package_checker",
        "is_jax_available",
        "is_torch_available",
        "is_flax_available",
        "is_tensorflow_available"
    ],
    "utils": [
        "get_mesh",
        "Timers",
        "Timer",
        "prefix_print",
        "RNG"
    ]
}

if TYPE_CHECKING:
    from .checker import (
        package_checker,
        is_jax_available,
        is_torch_available,
        is_flax_available,
        is_tensorflow_available
    )
    from .utils import (
        get_mesh,
        Timers,
        Timer,
        prefix_print,
        RNG
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
