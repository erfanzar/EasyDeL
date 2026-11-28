# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Aligned plain-text tables for records a person reads.

Eyvan's read commands emitted the durable record as JSON: a workload is sixty
keys, a cell is thirty, and a list of either is a screenful per row. That is the
right answer for something parsing and the wrong one for somebody asking what is
running, and the two were the same answer because there was only one.

So every read command renders columns by default and keeps ``--json`` for the
record. This is the renderer they share, which is what stops each command
inventing its own alignment and its own word for absent.

**Plain text, deliberately.** Rich is already a dependency and is used where
colour carries meaning -- a state, a warning. A table of identifiers does not
need it, and plain columns survive being piped into ``grep`` and pasted into an
issue, which is most of what happens to them.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from eyvan._internal.json_types import JsonObject, is_json_object

#: What an absent value renders as. One spelling everywhere: a column of blanks
#: reads as a rendering bug, and a column mixing "-", "None" and "" reads as
#: three different kinds of absence.
ABSENT = "-"

#: Ceiling on any one column, so a bucket URI or a stack trace cannot push every
#: other column off the right edge of a terminal.
MAX_WIDTH = 44


@dataclass(frozen=True, slots=True)
class Column:
    """One column: its heading, and how to get its value from a record.

    Attributes:
        heading: Shown uppercase; kept short because it sets the column's floor.
        read: Pulls the value out of one record. Returning ``None`` renders
            :data:`ABSENT` rather than an empty cell.
        align: ``">"`` for numbers, so digits line up on the decimal point.
    """

    heading: str
    read: tp.Callable[[JsonObject], object]
    align: str = "<"


def cell(value: object) -> str:
    """Render one value, truncating anything long enough to break the layout."""

    if value is None or value == "":
        return ABSENT
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        text = f"{value:.2f}"
    elif isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value) or ABSENT
    elif is_json_object(value):
        text = ", ".join(f"{key}={item}" for key, item in value.items()) or ABSENT
    else:
        text = str(value)
    # Truncated with an ellipsis rather than wrapped: a wrapped identifier
    # cannot be double-clicked, and these columns are mostly identifiers.
    return text if len(text) <= MAX_WIDTH else text[: MAX_WIDTH - 1] + "…"


def render(records: tp.Sequence[object], columns: tp.Sequence[Column], *, empty: str) -> tuple[str, ...]:
    """Render records as aligned columns, or one line saying there are none.

    ``empty`` is a sentence rather than a blank table, because "nothing matched
    your filter" and "this fleet holds nothing" are different facts and the
    caller is the only one that knows which it is.
    """

    rows = [record for record in records if is_json_object(record)]
    if not rows:
        return (empty,)
    table = [[cell(column.read(row)) for column in columns] for row in rows]
    headings = [column.heading.upper() for column in columns]
    widths = [max(len(headings[index]), *(len(row[index]) for row in table)) for index in range(len(columns))]
    lines = ["  ".join(headings[index].ljust(widths[index]) for index in range(len(columns))).rstrip()]
    for row in table:
        rendered = [
            row[index].rjust(widths[index]) if columns[index].align == ">" else row[index].ljust(widths[index])
            for index in range(len(columns))
        ]
        lines.append("  ".join(rendered).rstrip())
    return tuple(lines)


def field(*path: str) -> tp.Callable[[JsonObject], object]:
    """Read a nested value by key path, tolerating a record that lacks it.

    Records differ by version and by kind, and a reader that raises on a missing
    key would make one unexpected row fail the whole listing.
    """

    def read(record: JsonObject) -> object:
        value: object = record
        for key in path:
            if not is_json_object(value):
                return None
            value = value.get(key)
        return value

    return read


#: Keys worth showing first when describing one record, in this order. A record
#: has thirty keys and about four of them say what it is; the rest are the
#: document, which ``--json`` returns.
_IDENTIFYING = ("id", "name", "state", "status", "org_id", "project_id", "fleet", "role", "scope", "amount")


def describe(headline: str, record: object, *, fields: tp.Sequence[str] | None = None) -> tuple[str, ...]:
    """Render one record as a result somebody just caused.

    A mutation used to answer with the stored record, which is complete and
    unreadable: creating an organization printed thirty keys, and the one fact
    the caller wanted -- that it worked, and what it is called -- was somewhere
    inside. This says what happened, then the few fields that identify the thing
    it happened to.

    Args:
        headline: What just happened, as a sentence. The caller knows this and
            the record does not.
        record: The server's response. Anything that is not an object renders as
            the headline alone rather than inventing detail.
        fields: Keys to show, in order. Defaults to whichever of
            :data:`_IDENTIFYING` the record carries, so a caller that has
            nothing special to say does not have to choose.

    Returns:
        Lines to print, headline first.
    """

    lines = [headline]
    if not is_json_object(record):
        return tuple(lines)
    keys = tuple(fields) if fields is not None else tuple(key for key in _IDENTIFYING if key in record)
    width = max((len(key) for key in keys), default=0)
    for key in keys:
        value = record.get(key)
        # Nested structure is the document, not the identity of the thing; a
        # dict rendered inline here is how the wall of JSON grows back.
        if value is None or value == "" or isinstance(value, (dict, list, tuple)):
            continue
        lines.append(f"  {key.replace('_', ' '):<{width}}  {cell(value)}")
    return tuple(lines)


__all__ = ("ABSENT", "MAX_WIDTH", "Column", "cell", "describe", "field", "render")
