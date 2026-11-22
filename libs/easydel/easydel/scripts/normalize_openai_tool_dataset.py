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

"""Normalize an OpenAI-style tool-calling dataset for use with structured chat templates.

Many public tool-calling datasets ship with tool-call arguments and tool
parameter schemas stored as *stringified* JSON (i.e. a JSON string containing
escaped JSON) rather than as native JSON objects.  This is fine for models
that receive raw text, but breaks chat templates — like Qwen 3.5's — that
expect to iterate over the ``parameters`` dict or pretty-print function
arguments during template rendering.

This script reads a HuggingFace dataset row-by-row, decodes every
stringified field back into proper JSON objects via the
``normalize_openai_tool_dataset_example`` converter (shared with the
Lambda→Hermes conversion script), writes the result as JSONL + Parquet,
and optionally pushes both files plus a ``metadata.json`` summary to a
HuggingFace Hub repository.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


@dataclass
class NormalizationSummary:
    """Summary emitted after normalizing a tool-calling dataset split.

    ``drop_incomplete_tool_call_rows`` records whether incomplete-call filtering
    was enabled. Rows with empty tool lists are retained. The writer does not
    collapse adjacent assistant messages or strip system sections, and records
    those behaviors separately from the filtering policy.
    """

    source_dataset: str
    config_name: str
    split: str
    rows_seen: int = 0
    rows_written: int = 0
    decoded_json_strings: int = 0
    drop_incomplete_tool_call_rows: bool = False
    collapsed_adjacent_assistant_messages: bool = False
    empty_tool_rows_kept: bool = True
    stripped_system_tool_sections: bool = False
    qwen35_chat_template_reference: dict[str, Any] = field(default_factory=dict)


def _compact_json(value: Any) -> str:
    """Serialize metadata as stable, compact JSON."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _maybe_decode_json_string(value: Any) -> tuple[Any, int]:
    """Decode ``value`` when it is a string holding a JSON object or array.

    Only the value handed in is considered; containers are returned untouched. Callers point this at the specific
    schema fields that are known to be stringified, so free text (message ``content``) is never rewritten.

    Args:
        value: Candidate field value.

    Returns:
        ``(decoded, 1)`` when ``value`` was a JSON-looking string that parsed, else ``(value, 0)``.
    """

    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")):
            try:
                return json.loads(text), 1
            except json.JSONDecodeError:
                return value, 0
    return value, 0


def _decode_key(mapping: Any, key: str) -> tuple[Any, int]:
    """Return ``mapping`` with ``mapping[key]`` JSON-decoded; non-mappings and missing keys pass through."""

    if not isinstance(mapping, dict) or key not in mapping:
        return mapping, 0
    decoded, count = _maybe_decode_json_string(mapping[key])
    if count == 0:
        return mapping, 0
    updated = dict(mapping)
    updated[key] = decoded
    return updated, count


def _map_items(items: Any, fn: Callable[[Any], tuple[Any, int]]) -> tuple[Any, int]:
    """Apply ``fn`` to every element of a list field, decoding the list first when it is stringified."""

    items, total = _maybe_decode_json_string(items)
    if not isinstance(items, list):
        return items, total
    mapped = []
    for item in items:
        new_item, count = fn(item)
        mapped.append(new_item)
        total += count
    return mapped, total


def _normalize_function_holder(holder: Any, field_name: str) -> tuple[Any, int]:
    """Decode ``field_name`` on a tool call / tool spec, both flat and under the OpenAI ``function`` wrapper."""

    holder, total = _decode_key(holder, field_name)
    if isinstance(holder, dict) and isinstance(holder.get("function"), dict):
        function, count = _decode_key(holder["function"], field_name)
        if count:
            holder = {**holder, "function": function}
            total += count
    return holder, total


def _normalize_message(message: Any) -> tuple[Any, int]:
    """Decode the stringified tool-call ``arguments`` of one chat message (never its ``content``)."""

    if not isinstance(message, dict):
        return message, 0
    total = 0
    normalized = message
    if "tool_calls" in message:
        calls, count = _map_items(message["tool_calls"], lambda call: _normalize_function_holder(call, "arguments"))
        if count:
            normalized = {**normalized, "tool_calls": calls}
            total += count
    if "function_call" in message:
        legacy, count = _normalize_function_holder(message["function_call"], "arguments")
        if count:
            normalized = {**normalized, "function_call": legacy}
            total += count
    return normalized, total


def normalize_openai_tool_dataset_example(example: JsonObject) -> JsonObject:
    """Normalize one OpenAI-style tool-calling row.

    Only the schema fields that datasets ship as stringified JSON are decoded:
    ``messages[*].tool_calls[*].function.arguments`` (and the flat / legacy
    ``function_call`` forms), ``tools[*].function.parameters`` and the legacy
    ``functions[*].parameters``. A stringified ``messages`` / ``tools`` /
    ``tool_calls`` list is decoded too. Everything else, notably message
    ``content`` (for example a tool result of ``"[]"``), is left exactly as is.

    Args:
        example: Source dataset row.

    Returns:
        A shallow-normalized copy of the row with those fields decoded into
        native Python lists/dicts. The private ``"_decoded_json_strings"`` field
        records how many fields were decoded and is consumed by the writer.
    """

    normalized = dict(example)
    decoded_count = 0
    if "messages" in normalized:
        normalized["messages"], count = _map_items(normalized["messages"], _normalize_message)
        decoded_count += count
    for key in ("tools", "functions"):
        if key in normalized:
            normalized[key], count = _map_items(
                normalized[key], lambda tool: _normalize_function_holder(tool, "parameters")
            )
            decoded_count += count
    normalized["_decoded_json_strings"] = decoded_count
    return normalized


def _iter_source_rows(
    source_dataset: str,
    config_name: str | None,
    split: str,
    *,
    token: str | None,
    streaming: bool,
    max_rows: int | None,
) -> Iterator[JsonObject]:
    """Yield rows from a HuggingFace dataset split."""

    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "token": token, "streaming": streaming}
    dataset = (
        load_dataset(source_dataset, config_name, **kwargs) if config_name else load_dataset(source_dataset, **kwargs)
    )
    for idx, row in enumerate(dataset):
        if max_rows is not None and idx >= max_rows:
            break
        yield dict(row)


def _has_incomplete_tool_calls(example: JsonObject) -> bool:
    """Check that tool calls have names, object arguments and matching results.

    Modern OpenAI calls are matched to tool results by ``id`` / ``tool_call_id``;
    legacy ``function_call`` messages are matched to function results by name.
    Missing or ambiguous pending identifiers, malformed arguments, orphan results and
    unresolved calls at the end of a conversation make a row incomplete. A row
    without any calls or results (including an empty tool list) is complete.
    """
    messages = example.get("messages", [])
    if not isinstance(messages, list):
        return True
    pending: set[str] = set()
    legacy_pending: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            return True
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            return True
        for call in calls:
            if not isinstance(call, dict):
                return True
            function = call.get("function", call)
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id or call_id in pending:
                return True
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                return True
            if not function["name"].strip() or not isinstance(function.get("arguments"), dict):
                return True
            pending.add(call_id)
        if message.get("function_call") is not None:
            function = message["function_call"]
            if legacy_pending is not None or not isinstance(function, dict):
                return True
            name = function.get("name")
            if not isinstance(name, str) or not name.strip() or not isinstance(function.get("arguments"), dict):
                return True
            legacy_pending = name
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                return True
            pending.remove(call_id)
        elif message.get("role") == "function":
            if legacy_pending is None or message.get("name") != legacy_pending:
                return True
            legacy_pending = None
    return bool(pending) or legacy_pending is not None


def _write_jsonl_dataset(
    rows: Iterable[JsonObject],
    *,
    output_path: Path,
    converter: Callable[[JsonObject], JsonObject],
    source_dataset: str,
    config_name: str,
    split: str,
    drop_incomplete_tool_call_rows: bool = True,
) -> NormalizationSummary:
    """Write normalized rows, optionally filtering incomplete tool conversations."""

    summary = NormalizationSummary(
        source_dataset=source_dataset,
        config_name=config_name,
        split=split,
        drop_incomplete_tool_call_rows=drop_incomplete_tool_call_rows,
    )
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            summary.rows_seen += 1
            normalized = converter(row)
            decoded = normalized.pop("_decoded_json_strings", 0)
            if isinstance(decoded, int):
                summary.decoded_json_strings += decoded
            if drop_incomplete_tool_call_rows and _has_incomplete_tool_calls(normalized):
                continue
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            summary.rows_written += 1
    return summary


def _load_json_dataset(jsonl_path: Path, split: str):
    """Load a JSONL file through ``datasets`` so it can be written as parquet."""

    from datasets import Dataset, load_dataset

    # Filtering may remove every row. The JSON loader cannot infer a schema
    # from an empty file, but an empty Arrow table is still valid Parquet.
    if jsonl_path.stat().st_size == 0:
        return Dataset.from_dict({}, split=split)
    return load_dataset("json", data_files={split: str(jsonl_path)}, split=split)


def _write_summary(path: Path, summary: NormalizationSummary) -> None:
    """Write the normalization summary sidecar."""

    path.write_text(json.dumps(summary.__dict__, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for dataset normalization.

    Supports source selection (``--source-dataset``, ``--config-name``,
    ``--split``), output location (``--out``), optional streaming from HF
    (``--streaming``), row caps for dry runs (``--max-rows``), and Hub
    push options (``--push-to-hub``, ``--repo-id``, ``--private``).

    Args:
        argv: Optional list of CLI arguments. Uses ``sys.argv`` when ``None``.

    Returns:
        ``argparse.Namespace`` populated from the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Normalize an OpenAI-style messages/tools dataset by decoding stringified "
            "tool-call arguments and tool parameter schemas into JSON objects."
        )
    )
    parser.add_argument("--source-dataset", default="erfanzar/Reasoning-and-calling")
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True, help="Output directory for jsonl, parquet, and metadata.json")
    parser.add_argument("--token", default=None, help="HF token for private dataset access or hub push.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for dry runs or spot checks.")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream source rows from HF instead of materializing the full dataset first.",
    )
    parser.add_argument(
        "--drop-incomplete-tool-call-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows with malformed tool calls or calls without matching results; empty-tool rows are kept.",
    )
    parser.add_argument("--repo-id", default=None, help="Optional HF dataset repo to push the normalized data to.")
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Push the normalized dataset and metadata.json to the Hub when --repo-id is set.",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional privacy flag forwarded to datasets.push_to_hub.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full normalization pipeline.

    Steps:
        1. Stream or load rows from the source HF dataset.
        2. Apply ``normalize_openai_tool_dataset_example`` to each row,
           decoding stringified JSON fields into native objects.
        3. Write the result as ``{split}.jsonl`` and ``{split}.parquet``
           under ``--out``.
        4. Emit a ``metadata.json`` summary alongside the data files.
        5. If ``--push-to-hub`` is set, upload the dataset and metadata
           to the Hub repository specified by ``--repo-id``.

    Args:
        argv: Optional list of CLI arguments. Uses ``sys.argv`` when ``None``.

    Raises:
        SystemExit: If ``--push-to-hub`` is provided without ``--repo-id``.
    """
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / f"{args.split}.jsonl"
    parquet_path = out_dir / f"{args.split}.parquet"
    rows = _iter_source_rows(
        args.source_dataset,
        args.config_name,
        args.split,
        token=args.token,
        streaming=args.streaming,
        max_rows=args.max_rows,
    )
    summary = _write_jsonl_dataset(
        rows,
        output_path=jsonl_path,
        converter=normalize_openai_tool_dataset_example,
        source_dataset=args.source_dataset,
        config_name=args.config_name,
        split=args.split,
        drop_incomplete_tool_call_rows=args.drop_incomplete_tool_call_rows,
    )
    summary.qwen35_chat_template_reference["note"] = (
        "Stringified tool-call arguments and tool parameter schemas were decoded into JSON objects "
        "so the dataset can be used directly with the Qwen 3.5 chat template."
    )
    dataset = _load_json_dataset(jsonl_path, args.split)
    dataset.to_parquet(str(parquet_path))

    metadata_path = out_dir / "metadata.json"
    _write_summary(metadata_path, summary)

    if args.push_to_hub:
        if not args.repo_id:
            raise SystemExit("--push-to-hub requires --repo-id")
        push_kwargs: dict[str, Any] = {"token": args.token}
        if args.private is not None:
            push_kwargs["private"] = args.private
        dataset.push_to_hub(args.repo_id, config_name=args.config_name, split=args.split, **push_kwargs)

        from huggingface_hub import HfApi

        api = HfApi(token=args.token)
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo=metadata_path.name,
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print(_compact_json(summary.__dict__))


if __name__ == "__main__":
    main()
