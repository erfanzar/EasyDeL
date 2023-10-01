from importlib.util import find_spec


def package_checker(name: str):
    return True if find_spec(name) is not None else False


def is_torch_available():
    return package_checker('torch')


def is_jax_available():
    return package_checker('jax')


def is_flax_available():
    return package_checker('flax')


def is_tensorflow_available():
    return package_checker('tensorflow')
