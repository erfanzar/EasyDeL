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
from pathlib import Path
from typing import Any

from easydel.scripts.convert_lambda_hermes_to_openai import (
    _iter_source_rows,
    _load_json_dataset,
    _write_jsonl_dataset,
    _write_summary,
    normalize_openai_tool_dataset_example,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for dataset normalization.

    Supports source selection (``--source-dataset``, ``--config-name``,
    ``--split``), output location (``--out``), optional streaming from HF
    (``--streaming``), row caps for dry runs (``--max-rows``), and Hub
    push options (``--push-to-hub``, ``--repo-id``, ``--private``).
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

    from easydel.scripts.convert_lambda_hermes_to_openai import _compact_json

    print(_compact_json(summary.__dict__))


if __name__ == "__main__":
    main()
