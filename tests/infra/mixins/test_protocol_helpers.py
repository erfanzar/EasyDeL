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

"""Tests for ``easydel.infra.mixins.protocol`` helpers and abstract surface.

The bulk of this 3,662-line module is the ``BaseModuleProtocol`` class with
many ``@tp.overload`` __call__/mesh_call signatures (purely declarative for
type checkers). The genuinely testable surface:

* ``return_type_adjuster`` -- decorator factory
* ``get_module_repr`` -- module-to-string for ParallelLinear, Dropout, Embed,
  RMSNorm-style classes with ``eps``, and fallback
* ``prettify_module`` / ``printify_module`` -- recursive structure printer
  with ``EasyDeL-`` prefix and ``"EasyDeL-Partitions"`` fallback
* ``BaseModuleProtocol`` -- abstract surface check (cannot instantiate; has
  the documented module-level attribute names)
"""

from __future__ import annotations

from abc import ABCMeta

import pytest

from easydel.infra.mixins.protocol import (
    BaseModuleProtocol,
    get_module_repr,
    prettify_module,
    printify_module,
    return_type_adjuster,
)


def test_return_type_adjuster_returns_decorator():
    decorator = return_type_adjuster(int)
    assert callable(decorator)


def test_return_type_adjuster_passes_through_value_at_runtime():
    """The cast is purely for type-checkers; runtime behavior is identity."""

    @return_type_adjuster(str)
    def make_thing() -> object:
        return 42

    result = make_thing()
    assert result == 42


def test_return_type_adjuster_forwards_args_and_kwargs():
    """The decorator's wrapper must forward both positional and keyword args."""

    @return_type_adjuster(dict)
    def builder(a, b, *, c=10):
        return {"a": a, "b": b, "c": c}

    result = builder(1, 2, c=3)
    assert result == {"a": 1, "b": 2, "c": 3}


class _FakeWeight:
    def __init__(self, shape):
        self.shape = shape


def test_get_module_repr_dropout_shows_rate(monkeypatch):
    """Dropout modules render as 'Dropout(p=X)'."""
    import spectrax as spx_module

    nn_dropout_cls = spx_module.nn.Dropout
    instance = nn_dropout_cls.__new__(nn_dropout_cls)
    instance.rate = 0.25
    repr_str = get_module_repr(instance)
    assert repr_str == "Dropout(p=0.25)"


def test_get_module_repr_falls_back_to_class_name():
    """Modules that match no known type return just the class name."""

    class CustomThing:
        pass

    obj = CustomThing()


    assert get_module_repr(obj) == "CustomThing"


def test_get_module_repr_module_with_eps_shows_shape_and_eps():
    """Norm-style modules with ``eps`` use a shape+eps format."""

    class MyNorm:
        eps = 1e-6
        weight = _FakeWeight((128,))

    repr_str = get_module_repr(MyNorm())
    assert "MyNorm" in repr_str
    assert "(128,)" in repr_str
    assert "eps=1e-06" in repr_str


def test_get_module_repr_module_with_eps_no_weight():
    class NormNoWeight:
        eps = 0.001

    repr_str = get_module_repr(NormNoWeight())
    assert "NormNoWeight" in repr_str
    assert "eps=0.001" in repr_str

    assert repr_str.endswith(", eps=0.001)")


class _LeafModule:
    """Minimal module-like class without children -- exercise the leaf path."""

    def iter_children(self):
        return iter(())

    @property
    def __dict__(self):
        return {}


def test_prettify_module_leaf_returns_class_name_only():
    leaf = _LeafModule()
    output = prettify_module(leaf)

    assert output.strip() == "_LeafModule"


def test_prettify_module_max_depth_zero_returns_only_root():
    """``max_depth=0`` allows only the root level; all children would be below."""
    leaf = _LeafModule()
    output = prettify_module(leaf, max_depth=0)
    assert output.strip() == "_LeafModule"


def test_prettify_module_max_depth_below_zero_returns_empty():
    """When ``depth > max_depth``, return empty string per the function's contract."""
    leaf = _LeafModule()

    assert prettify_module(leaf, depth=1, max_depth=0) == ""


def test_printify_module_prefixes_easydel():
    """The wrapper prepends ``EasyDeL-`` to the prettified module."""
    leaf = _LeafModule()
    output = printify_module(leaf)
    assert output.startswith("EasyDeL-")
    assert "_LeafModule" in output


def test_printify_module_falls_back_for_partition_objects():
    """Objects that raise AttributeError during pretty-printing yield ``EasyDeL-Partitions``."""

    class PartitionLike:

        pass

    output = printify_module(PartitionLike())
    assert output == "EasyDeL-Partitions"


def test_printify_module_does_not_silently_swallow_other_errors():
    """Only AttributeError triggers the fallback -- other errors should propagate.

    The except clause in printify_module is narrow (``except AttributeError``).
    A class that raises a different error during prettify must NOT be silently
    converted to ``EasyDeL-Partitions``.
    """

    class RaisesValueError:
        def iter_children(self):
            raise ValueError("intentional")

        @property
        def __dict__(self):
            return {}

    with pytest.raises(ValueError):
        printify_module(RaisesValueError())


def test_base_module_protocol_uses_abcmeta():
    assert isinstance(BaseModuleProtocol, ABCMeta)


def test_base_module_protocol_declares_documented_class_attrs():
    """The class-level annotations include the configuration / model-type slots."""
    annotations = BaseModuleProtocol.__annotations__
    assert "config_class" in annotations
    assert "config" in annotations
    assert "base_model_prefix" in annotations


def test_base_module_protocol_class_attr_defaults():
    """``_model_task`` and ``_model_type`` default to None at class level."""
    assert BaseModuleProtocol._model_task is None
    assert BaseModuleProtocol._model_type is None


def test_base_module_protocol_has_call_method():
    """The class declares a ``__call__`` (overloaded). Just ensure it's present."""
    assert "__call__" in BaseModuleProtocol.__dict__


def test_base_module_protocol_subclass_inherits_class_attrs():
    """Subclasses can rely on default values for ``_model_task``/``_model_type``."""

    class FakeChild(BaseModuleProtocol):


        pass

    assert FakeChild._model_task is None
    assert FakeChild._model_type is None
