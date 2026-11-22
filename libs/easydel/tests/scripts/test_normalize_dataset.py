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
"""Regression tests for ``normalize_openai_tool_dataset`` (no array computation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from easydel.scripts import normalize_openai_tool_dataset as norm

pytestmark = pytest.mark.cpu_ok


def _row():
    return {
        "messages": [
            {"role": "user", "content": '{"a": 1}'},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": '{"x": [1, 2]}'}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "[]"},
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": '{"type": "object"}'}}],
    }


def test_only_schema_fields_are_decoded_and_message_content_is_preserved():
    out = norm.normalize_openai_tool_dataset_example(_row())

    assert out["messages"][0]["content"] == '{"a": 1}'
    assert out["messages"][2]["content"] == "[]"
    assert out["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"x": [1, 2]}
    assert out["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert out["_decoded_json_strings"] == 2


def test_normalizer_does_not_mutate_input_row():
    row = _row()
    norm.normalize_openai_tool_dataset_example(row)
    assert row["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"x": [1, 2]}'


def test_metadata_reports_only_behavior_the_pipeline_performed(tmp_path, monkeypatch):
    class _Dataset:
        def to_parquet(self, path):
            Path(path).write_text("", encoding="utf-8")

    incomplete = _row()
    incomplete["messages"].pop()
    empty = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
    monkeypatch.setattr(norm, "_iter_source_rows", lambda *a, **k: iter([_row(), incomplete, empty]))
    monkeypatch.setattr(norm, "_load_json_dataset", lambda *a, **k: _Dataset())

    norm.main(["--out", str(tmp_path), "--source-dataset", "x/y"])

    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["rows_seen"] == 3
    assert meta["rows_written"] == 2
    assert meta["drop_incomplete_tool_call_rows"] is True
    assert meta["empty_tool_rows_kept"] is True
    written = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert written[1] == empty
    assert meta["collapsed_adjacent_assistant_messages"] is False
    assert meta["stripped_system_tool_sections"] is False
    # ``[]`` tool content survives the round trip as a string in the written JSONL.
    first = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["messages"][2]["content"] == "[]"


@pytest.mark.parametrize("filter_rows", [True, False])
def test_filtering_policy_is_controllable_and_recorded(tmp_path, monkeypatch, filter_rows):
    class _Dataset:
        def to_parquet(self, path):
            Path(path).write_text("", encoding="utf-8")

    incomplete = _row()
    incomplete["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{broken"
    monkeypatch.setattr(norm, "_iter_source_rows", lambda *a, **k: iter([incomplete, _row()]))
    monkeypatch.setattr(norm, "_load_json_dataset", lambda *a, **k: _Dataset())
    flag = "--drop-incomplete-tool-call-rows" if filter_rows else "--no-drop-incomplete-tool-call-rows"

    norm.main(["--out", str(tmp_path), flag])

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["drop_incomplete_tool_call_rows"] is filter_rows
    written = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert len(written) == (1 if filter_rows else 2)
    if not filter_rows:
        assert written[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == "{broken"


def test_all_incomplete_rows_still_produce_empty_parquet_and_metadata(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    incomplete = _row()
    incomplete["messages"].pop()
    monkeypatch.setattr(norm, "_iter_source_rows", lambda *a, **k: iter([incomplete]))

    norm.main(["--out", str(tmp_path)])

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["rows_seen"] == 1
    assert meta["rows_written"] == 0
    assert (tmp_path / "train.jsonl").read_text() == ""
    assert pq.ParquetFile(tmp_path / "train.parquet").metadata.num_rows == 0


def test_legacy_function_call_and_stringified_containers_are_supported(tmp_path):
    row = {
        "messages": json.dumps(
            [
                {"role": "assistant", "function_call": {"name": "f", "arguments": "{}"}},
                {"role": "function", "name": "f", "content": "[]"},
            ]
        ),
        "functions": json.dumps([{"name": "f", "parameters": '{"type": "object"}'}]),
    }
    out = tmp_path / "out.jsonl"
    summary = norm._write_jsonl_dataset(
        [row],
        output_path=out,
        converter=norm.normalize_openai_tool_dataset_example,
        source_dataset="test",
        config_name="default",
        split="train",
    )

    assert summary.rows_written == 1
    written = json.loads(out.read_text())
    assert written["messages"][0]["function_call"]["arguments"] == {}
    assert written["messages"][1]["content"] == "[]"
    assert written["functions"][0]["parameters"] == {"type": "object"}
