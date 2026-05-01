# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy-import infrastructure used by EasyDeL packages.

Provides :class:`LazyModule`, a ``ModuleType`` subclass that defers
sub-module imports until the first attribute access. EasyDeL's top-level
``__init__`` swaps itself for a ``LazyModule`` so that ``import easydel`` is
fast and individual heavy sub-modules (modeling, trainers, kernels, etc.)
are only imported when actually used. The implementation is adapted from
HuggingFace's lazy-module pattern.

Also provides :class:`DummyObject`, a metaclass for placeholder classes that
stand in for symbols whose backend dependencies are missing, and the
:func:`is_package_available` import-checking helper.
"""

import importlib
import os
import typing as tp
from itertools import chain
from types import ModuleType

BACKENDS_T = frozenset[str]
IMPORT_STRUCTURE_T = dict[BACKENDS_T, dict[str, set[str]]]


class LazyModule(ModuleType):
    """Module subclass that defers sub-module imports until first access.

    Used as the top-level ``__init__`` module for EasyDeL packages so that
    ``import easydel`` is fast and only imports sub-packages on demand.
    Adapted from the HuggingFace ``transformers`` lazy module pattern.

    Attributes:
        _modules: Set of known sub-module names.
        _class_to_module: Mapping from exported class name to its sub-module.
        _import_structure: Flattened mapping of module -> list of exported names.
        _objects: Extra objects injected at construction time.
    """

    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: IMPORT_STRUCTURE_T,
        module_spec: importlib.machinery.ModuleSpec | None = None,
        extra_objects: dict[str, object] | None = None,
    ):
        """Initialize a lazy module skeleton.

        Two ``import_structure`` shapes are accepted:

        * Mapping of ``frozenset(backend_name, ...)`` to ``{module: {symbol, ...}}``
          (the new HF style with backend-gated symbols).
        * Mapping of ``module: [symbol, ...]`` (the legacy flat shape).

        Args:
            name: The fully-qualified module name being installed (e.g.
                ``"easydel"``).
            module_file: The ``__file__`` of the underlying ``__init__.py``.
            import_structure: Description of which symbols come from which
                sub-module, optionally gated on backend frozensets.
            module_spec: Optional ``ModuleSpec`` to attach (preserves package
                semantics).
            extra_objects: Mapping of symbol name to a concrete value that
                should be returned directly without importing a sub-module.
        """
        super().__init__(name)

        self._object_missing_backend = {}
        if any(isinstance(key, frozenset) for key in import_structure.keys()):
            self._modules = set()
            self._class_to_module = {}
            self.__all__ = []

            _import_structure = {}

            for _backends, module in import_structure.items():
                self._modules = self._modules.union(set(module.keys()))

                for key, values in module.items():
                    for value in values:
                        self._class_to_module[value] = key
                    _import_structure.setdefault(key, []).extend(values)

                # Needed for autocompletion in an IDE
                self.__all__.extend(list(module.keys()) + list(chain(*module.values())))

            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = _import_structure

        # This can be removed once every exportable object has a `export()` export.
        else:
            self._modules = set(import_structure.keys())
            self._class_to_module = {}
            for key, values in import_structure.items():
                for value in values:
                    self._class_to_module[value] = key
            # Needed for autocompletion in an IDE
            self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = import_structure

    def __dir__(self):
        """Return ``dir()`` augmented with all lazy-export names.

        Returns:
            The default ``ModuleType.__dir__`` result extended with every name
            declared in ``__all__`` so IDE auto-completion sees lazy exports.
        """
        result = super().__dir__()
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> tp.Any:
        """Resolve a lazy attribute by importing its backing sub-module.

        Args:
            name: Attribute being accessed on the lazy module.

        Returns:
            The resolved object, which is also cached on ``self`` so further
            accesses don't reimport.

        Raises:
            AttributeError: If ``name`` is not registered as a sub-module,
                a class export, an extra object, or a missing-backend stub.
        """
        if name in self._objects:
            return self._objects[name]
        if name in self._object_missing_backend.keys():
            missing_backends = self._object_missing_backend[name]

            class Placeholder(metaclass=DummyObject):
                """Stub class returned when required backends are missing.

                Tagged with ``_backends`` so callers can introspect which
                optional dependencies they would need to install before this
                symbol becomes usable.
                """

                _backends = missing_backends

            Placeholder.__name__ = name
            Placeholder.__module__ = self.__spec__

            value = Placeholder
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        elif name in self._modules:
            value = self._get_module(name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        """Import and return a child sub-module by relative name.

        Args:
            module_name: The sub-module name relative to this package.

        Returns:
            The imported module.

        Raises:
            RuntimeError: If the import raises any exception (with the
                original error chained as ``__cause__`` for full traceback).
        """
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        """Support ``pickle`` round-trips by rebuilding the lazy module.

        Returns:
            A ``(class, args)`` tuple suitable for ``pickle`` to call.
        """
        return (self.__class__, (self._name, self.__file__, self._import_structure))


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattribute__(cls, key):
        """Forward dunder lookups to the standard machinery, swallow others.

        Args:
            key: Attribute name being accessed on the placeholder class.

        Returns:
            The result of ``type.__getattribute__`` for private/dunder names
            (and ``_from_config``); ``None`` is returned implicitly for any
            other attribute, which keeps ``hasattr`` checks falsey without
            raising at the access site.
        """
        if key.startswith("_") and key != "_from_config":
            return super().__getattribute__(key)


def is_package_available(package_name: str) -> bool:
    """
    Checks if a package is available in the current Python environment.

    Args:
        package_name: The name of the package to check (e.g., "numpy").

    Returns:
        True if the package is available, False otherwise.
    """
    return importlib.util.find_spec(package_name.replace("-", "_")) is not None
