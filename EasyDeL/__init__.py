__version__ = '0.0.0'

from .utils.checker import package_checker, is_jax_available, is_torch_available, is_flax_available, \
    is_tensorflow_available

if is_torch_available():
    from .modules import CAdamW
if is_jax_available():
    ...
if is_tensorflow_available():
    ...
if is_flax_available():
    ...
__all__ = __version__, 'package_checker', 'is_jax_available', 'is_torch_available', 'is_flax_available', \
          'is_tensorflow_available'
