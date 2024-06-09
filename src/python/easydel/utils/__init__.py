from .lazy_import import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "checker": [
        "package_checker",
        "is_jax_available",
        "is_torch_available",
        "is_flax_available",
        "is_tensorflow_available",
    ],
    "helpers": ["get_mesh", "Timers", "Timer", "prefix_print", "RNG"],
}

if TYPE_CHECKING:
    from .checker import (
        package_checker as package_checker,
        is_jax_available as is_jax_available,
        is_torch_available as is_torch_available,
        is_flax_available as is_flax_available,
        is_tensorflow_available as is_tensorflow_available,
    )
    from .helpers import (
        get_mesh as get_mesh,
        Timers as Timers,
        Timer as Timer,
        prefix_print as prefix_print,
        RNG as RNG,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
