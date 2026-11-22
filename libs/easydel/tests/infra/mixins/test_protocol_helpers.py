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
* ``BaseModuleProtocol`` -- abstract surface check (cannot instantiate; has
  the documented module-level attribute names)
"""

from __future__ import annotations

from abc import ABCMeta

from easydel.infra.mixins.protocol import BaseModuleProtocol, return_type_adjuster


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


class _LeafModule:
    """Minimal module-like class without children -- exercise the leaf path."""

    def iter_children(self):
        return iter(())

    @property
    def __dict__(self):
        return {}


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
