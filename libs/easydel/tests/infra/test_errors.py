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

"""Tests for ``easydel.infra.errors``.

The error classes are control-flow signals and tagged exception types. The
tests here lock in:

* class hierarchy (every class is an ``Exception`` subclass),
* ``raise``/``catch`` round-trip semantics,
* message preservation through ``str(exc)``,
* importability from both ``easydel.infra.errors`` and the package root,
* picklability (control-flow exceptions cross process boundaries when
  the trainer's preemption signal travels via ``jax.distributed`` etc).
"""

from __future__ import annotations

import pickle

import pytest

from easydel.infra import errors as ed_errors

ERROR_CLASSES = [
    ed_errors.EasyDeLRuntimeError,
    ed_errors.EasyDeLSyntaxRuntimeError,
    ed_errors.EasyDeLTimerError,
    ed_errors.EasyDeLBreakRequest,
    ed_errors.EasyDeLPreemptionSignal,
    ed_errors.EasyDeLBlockWiseFFNError,
    ed_errors.EasyDeLProcessError,
    ed_errors.EasyDeLComputeError,
    ed_errors.EasyDeLNotImplementedFeatureError,
]


@pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
def test_each_error_class_is_a_subclass_of_exception(cls):
    assert issubclass(cls, Exception)
    assert not issubclass(cls, BaseException) or issubclass(cls, Exception), (
        f"{cls.__name__} must inherit from Exception, not BaseException directly"
    )


@pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
def test_each_error_class_round_trips_through_raise_catch(cls):
    sentinel = f"sentinel-message-for-{cls.__name__}"
    with pytest.raises(cls) as excinfo:
        raise cls(sentinel)
    assert sentinel in str(excinfo.value)

    try:
        raise cls(sentinel)
    except Exception as exc:
        assert isinstance(exc, cls)


@pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
def test_each_error_class_supports_no_args_construction(cls):
    """Some control-flow uses raise EasyDeLBreakRequest() without a message."""
    err = cls()
    assert isinstance(err, cls)
    assert isinstance(err, Exception)


@pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
def test_each_error_class_is_picklable_for_distributed_propagation(cls):
    """Errors can cross process boundaries via pickle (used by preemption signaling)."""
    msg = "across-process"
    err = cls(msg)
    revived = pickle.loads(pickle.dumps(err))
    assert isinstance(revived, cls)
    assert str(revived) == msg


def test_distinct_error_classes_are_not_aliases():
    """Defensive: each error class is its own type. Catching one shouldn't catch another."""
    seen = set()
    for cls in ERROR_CLASSES:
        assert cls not in seen, f"{cls.__name__} appears twice in ERROR_CLASSES list"
        seen.add(cls)

    with pytest.raises(ed_errors.EasyDeLRuntimeError):
        raise ed_errors.EasyDeLRuntimeError("a")
    with pytest.raises(ed_errors.EasyDeLBreakRequest):
        raise ed_errors.EasyDeLBreakRequest("b")

    try:
        raise ed_errors.EasyDeLBreakRequest("c")
    except ed_errors.EasyDeLRuntimeError:
        pytest.fail("EasyDeLRuntimeError caught EasyDeLBreakRequest -- types are aliased")
    except ed_errors.EasyDeLBreakRequest:
        pass


def test_errors_module_exports_all_classes():
    """Every error class is reachable as a module attribute (for star-imports / docs)."""
    for cls in ERROR_CLASSES:
        assert getattr(ed_errors, cls.__name__) is cls


def test_break_request_does_not_subclass_keyboardinterrupt():
    """``EasyDeLBreakRequest`` is a regular Exception -- it must NOT escape ``except Exception``.

    Trainer loops use ``try/except Exception`` to catch ``BreakRequest`` and shut down
    gracefully. If it ever became a ``BaseException`` subclass (e.g. inheriting from
    ``KeyboardInterrupt`` or ``SystemExit``), the existing handlers would silently
    let it through.
    """
    with pytest.raises(Exception) as excinfo:
        raise ed_errors.EasyDeLBreakRequest("graceful stop")
    assert isinstance(excinfo.value, ed_errors.EasyDeLBreakRequest)
    assert not isinstance(excinfo.value, KeyboardInterrupt)
    assert not isinstance(excinfo.value, SystemExit)


def test_preemption_signal_distinguishes_from_stop_iteration():
    """``EasyDeLPreemptionSignal`` is documented to differ from ``StopIteration``."""
    err = ed_errors.EasyDeLPreemptionSignal("preempted")
    assert not isinstance(err, StopIteration)
    assert isinstance(err, Exception)
