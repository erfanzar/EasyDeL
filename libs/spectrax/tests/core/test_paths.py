# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.paths`."""

from __future__ import annotations

import pytest

from spectrax.core.paths import is_prefix, join, path_to_str, str_to_path


def test_path_to_str_empty():
    """Empty tuple renders to an empty string."""
    assert path_to_str(()) == ""


def test_path_to_str_plain_string():
    """Plain string components are emitted verbatim."""
    assert path_to_str(("a", "b", "c")) == "a.b.c"


def test_path_to_str_integer_component():
    """Integer components render as bare digits."""
    assert path_to_str(("layers", 0, "weight")) == "layers.0.weight"


def test_path_to_str_all_digit_string_is_quoted():
    """A string component that looks like a digit is quoted with ``#``."""
    assert path_to_str(("foo", "123")) == "foo.#123"


def test_path_to_str_dotted_string_is_escaped():
    """A string containing a dot is escaped so the codec is invertible."""
    assert path_to_str(("ns", "a.b")) == "ns.#a\\.b"


def test_path_to_str_hash_prefixed_string_is_re_quoted():
    """A string that already starts with ``#`` is quoted again."""
    assert path_to_str(("foo", "#bar")) == "foo.##bar"


def test_path_to_str_rejects_invalid_component_type():
    """Non str/int components trigger ``TypeError``."""
    with pytest.raises(TypeError):
        path_to_str((1.5,))


def test_str_to_path_empty():
    """Empty string parses to the empty tuple."""
    assert str_to_path("") == ()


def test_str_to_path_plain():
    """Plain dotted string decodes to a string tuple."""
    assert str_to_path("a.b.c") == ("a", "b", "c")


def test_str_to_path_digit_becomes_int():
    """All-digit segments decode to ``int``."""
    assert str_to_path("layers.0.weight") == ("layers", 0, "weight")


def test_str_to_path_quoted_digit_stays_string():
    """A ``#``-quoted digit segment decodes to a string."""
    assert str_to_path("foo.#123") == ("foo", "123")


def test_str_to_path_escaped_dot():
    """Backslash-escaped dots are preserved in the decoded string."""
    assert str_to_path("ns.#a\\.b") == ("ns", "a.b")


def test_roundtrip_with_mixed_components():
    """Arbitrary mixed paths round-trip exactly."""
    original = ("encoder", 0, "layers", "blk.1", "12", "w")
    encoded = path_to_str(original)
    assert str_to_path(encoded) == original


def test_roundtrip_integer_chain():
    """Pure-integer paths round-trip to integers."""
    original = (0, 1, 2, 3)
    assert str_to_path(path_to_str(original)) == original


def test_is_prefix_true():
    """A shorter prefix is recognized."""
    assert is_prefix(("a", "b"), ("a", "b", "c", "d"))


def test_is_prefix_equal():
    """A path is a prefix of itself."""
    assert is_prefix(("a", "b"), ("a", "b"))


def test_is_prefix_empty_is_prefix_of_anything():
    """The empty path is a prefix of every path."""
    assert is_prefix((), ("a",))
    assert is_prefix((), ())


def test_is_prefix_false_mismatch():
    """Mismatched middle components are not prefixes."""
    assert not is_prefix(("a", "x"), ("a", "b", "c"))


def test_is_prefix_false_too_long():
    """A longer ``prefix`` cannot be a prefix of a shorter path."""
    assert not is_prefix(("a", "b", "c"), ("a", "b"))


def test_join_single():
    """Joining a single tuple returns an equivalent path."""
    assert join(("a", "b")) == ("a", "b")


def test_join_multiple():
    """Multiple iterables concatenate in order."""
    assert join(("a",), ("b", "c"), (1,)) == ("a", "b", "c", 1)


def test_join_no_args():
    """Calling :func:`join` with no args yields the empty tuple."""
    assert join() == ()


def test_join_accepts_any_iterable():
    """:func:`join` accepts arbitrary iterables, not just tuples."""
    assert join(["a", "b"], ("c",)) == ("a", "b", "c")
