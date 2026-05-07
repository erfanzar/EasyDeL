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
    """Summary emitted after normalizing a tool-calling dataset split."""

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
    """Decode stringified JSON leaves while preserving ordinary strings."""

    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")):
            try:
                return json.loads(text), 1
            except json.JSONDecodeError:
                return value, 0
        return value, 0
    if isinstance(value, list):
        decoded_count = 0
        decoded_items = []
        for item in value:
            decoded, count = _maybe_decode_json_string(item)
            decoded_items.append(decoded)
            decoded_count += count
        return decoded_items, decoded_count
    if isinstance(value, dict):
        decoded_count = 0
        decoded_items: dict[str, Any] = {}
        for key, item in value.items():
            decoded, count = _maybe_decode_json_string(item)
            decoded_items[str(key)] = decoded
            decoded_count += count
        return decoded_items, decoded_count
    return value, 0


def normalize_openai_tool_dataset_example(example: JsonObject) -> JsonObject:
    """Normalize one OpenAI-style tool-calling row.

    Args:
        example: Source dataset row. The row may contain JSON fields encoded as
            strings, commonly under ``messages[*].tool_calls[*].function.arguments``
            or ``tools[*].function.parameters``.

    Returns:
        A shallow-normalized row with stringified JSON leaves decoded into
        native Python lists/dicts. The private ``"_decoded_json_strings"`` field
        records how many leaves were decoded and is consumed by the writer.
    """

    normalized, decoded_count = _maybe_decode_json_string(dict(example))
    if not isinstance(normalized, dict):
        normalized = dict(example)
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
    dataset = load_dataset(source_dataset, config_name, **kwargs) if config_name else load_dataset(source_dataset, **kwargs)
    for idx, row in enumerate(dataset):
        if max_rows is not None and idx >= max_rows:
            break
        yield dict(row)


def _write_jsonl_dataset(
    rows: Iterable[JsonObject],
    *,
    output_path: Path,
    converter: Callable[[JsonObject], JsonObject],
    source_dataset: str,
    config_name: str,
    split: str,
) -> NormalizationSummary:
    """Write normalized rows as JSONL and return a conversion summary."""

    summary = NormalizationSummary(source_dataset=source_dataset, config_name=config_name, split=split)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            summary.rows_seen += 1
            normalized = converter(row)
            decoded = normalized.pop("_decoded_json_strings", 0)
            if isinstance(decoded, int):
                summary.decoded_json_strings += decoded
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            summary.rows_written += 1
    return summary


def _load_json_dataset(jsonl_path: Path, split: str):
    """Load a JSONL file through ``datasets`` so it can be written as parquet."""

    from datasets import load_dataset

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
    )
    summary.drop_incomplete_tool_call_rows = True
    summary.collapsed_adjacent_assistant_messages = False
    summary.empty_tool_rows_kept = True
    summary.stripped_system_tool_sections = False
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
