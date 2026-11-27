# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""W&B-style human-readable random run name generator.

Standard library only so training jobs can generate readable run names without
pulling extra dependencies or touching the process-wide ``random`` state.
"""

from __future__ import annotations

import random
import re
import secrets

_ADJECTIVES: tuple[str, ...] = (
    "agate",
    "alpine",
    "amber",
    "astral",
    "azure",
    "bold",
    "bright",
    "brisk",
    "bronze",
    "calm",
    "cedar",
    "cinder",
    "clear",
    "cobalt",
    "copper",
    "coral",
    "crimson",
    "crisp",
    "dawn",
    "deft",
    "drift",
    "ember",
    "emerald",
    "faint",
    "fern",
    "flint",
    "frost",
    "gilded",
    "glacial",
    "golden",
    "granite",
    "harbor",
    "ivory",
    "jade",
    "keen",
    "lively",
    "lucid",
    "lunar",
    "marble",
    "meadow",
    "misty",
    "nimble",
    "noble",
    "ochre",
    "olive",
    "opal",
    "pebble",
    "pine",
    "placid",
    "polar",
    "quartz",
    "quiet",
    "radiant",
    "rapid",
    "river",
    "ruby",
    "russet",
    "saffron",
    "sage",
    "scarlet",
    "serene",
    "silver",
    "slate",
    "solar",
    "steady",
    "still",
    "stone",
    "stormy",
    "sunny",
    "swift",
    "teal",
    "timber",
    "topaz",
    "velvet",
    "verdant",
    "vivid",
    "warm",
    "wild",
    "willow",
    "zenith",
)

_NOUNS: tuple[str, ...] = (
    "albatross",
    "anchor",
    "antelope",
    "aurora",
    "badger",
    "beacon",
    "bison",
    "breeze",
    "brook",
    "canyon",
    "caribou",
    "cascade",
    "chamois",
    "citadel",
    "comet",
    "compass",
    "condor",
    "coyote",
    "crane",
    "crest",
    "cypress",
    "dolphin",
    "dune",
    "eagle",
    "eclipse",
    "falcon",
    "fjord",
    "fox",
    "gazelle",
    "gecko",
    "glacier",
    "grove",
    "harrier",
    "hawk",
    "heron",
    "horizon",
    "ibex",
    "island",
    "jaguar",
    "kestrel",
    "lagoon",
    "lantern",
    "lark",
    "leopard",
    "lynx",
    "marmot",
    "merlin",
    "mesa",
    "meteor",
    "nebula",
    "newt",
    "orca",
    "oriole",
    "osprey",
    "otter",
    "owl",
    "panther",
    "peak",
    "pelican",
    "peregrine",
    "phoenix",
    "pika",
    "plover",
    "puma",
    "quail",
    "rapids",
    "raven",
    "reef",
    "ridge",
    "robin",
    "salmon",
    "sequoia",
    "sparrow",
    "spire",
    "starling",
    "summit",
    "swift",
    "tern",
    "thistle",
    "tiger",
    "tundra",
    "valley",
    "vortex",
    "walrus",
    "wren",
    "zephyr",
)

_PREFIX_UNSAFE = re.compile(r"[^a-z0-9-]+")
_COLLAPSE_DASH = re.compile(r"-{2,}")
_MAX_GENERATED_LEN = 64


def _sanitize_prefix(prefix: str) -> str:
    lowered = prefix.strip().lower().replace("_", "-").replace("/", "-").replace(".", "-")
    cleaned = _PREFIX_UNSAFE.sub("-", lowered)
    return _COLLAPSE_DASH.sub("-", cleaned).strip("-")


def generate_run_name(*, prefix: str | None = None, seed: int | None = None) -> str:
    """Generate a readable W&B-style run name such as ``amber-falcon-42``.

    When ``seed`` is ``None`` (the default), selection uses :mod:`secrets` so
    it is unaffected by ``random.seed(...)`` calls in training scripts.

    Args:
        prefix: Optional context prefix (e.g. model or trainer slug) prepended
            and sanitized to keep the total length at most 64 characters.
        seed: Optional integer seed for deterministic name generation in tests.

    Returns:
        A lowercase hyphen-separated run name slug.

    Raises:
        TypeError: If ``prefix`` is not a string or ``seed`` is not an integer.
    """

    if prefix is not None and not isinstance(prefix, str):
        raise TypeError("prefix must be a string or None")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")

    if seed is None:
        adjective = _ADJECTIVES[secrets.randbelow(len(_ADJECTIVES))]
        noun = _NOUNS[secrets.randbelow(len(_NOUNS))]
        number = secrets.randbelow(99) + 1
    else:
        rng = random.Random(seed)
        adjective = rng.choice(_ADJECTIVES)
        noun = rng.choice(_NOUNS)
        number = rng.randint(1, 99)

    base = f"{adjective}-{noun}-{number}"
    if prefix is None:
        return base
    clean_prefix = _sanitize_prefix(prefix)
    if not clean_prefix:
        return base
    budget = _MAX_GENERATED_LEN - len(base) - 1
    if budget < 1:
        return base
    trimmed = clean_prefix[:budget].rstrip("-")
    if not trimmed:
        return base
    return f"{trimmed}-{base}"


__all__ = ("generate_run_name",)
