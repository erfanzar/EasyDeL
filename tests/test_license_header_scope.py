# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Header maintenance must not relabel vendored dependencies or scratch files."""

import importlib.util
from pathlib import Path

import pytest


def test_default_header_scan_is_limited_to_first_party_sources(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "license_headers.py"
    spec = importlib.util.spec_from_file_location("license_headers_scope", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    included = ("eyvan/app.py", "tests/test_app.py", "scripts/tool.py", "examples/demo.py", "dashboard/src/App.tsx")
    excluded = (".pytest_cache/check/site-packages/dependency.py", "dashboard/node_modules/lib/index.ts", "dist/app.py")
    for name in (*included, *excluded):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original third-party notice\n")
    assert {str(path.relative_to(tmp_path)) for path in module.default_targets(tmp_path)} == set(included)
    for name in excluded:
        assert (tmp_path / name).read_text() == "# original third-party notice\n"


def test_existing_header_preserves_following_directives_and_comments(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "license_headers.py"
    spec = importlib.util.spec_from_file_location("license_headers_preservation", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, body in (
        ("suite.tsx", "// @vitest-environment jsdom\n\nexport {};\n"),
        ("block.tsx", "/** @vitest-environment jsdom */\n\nexport {};\n"),
        ("module.py", "# Important module comment\n# mypy: ignore-errors\n\nvalue = 1\n"),
    ):
        path = tmp_path / name
        original = module.header_for(path) + "\n" + body
        path.write_text(original)
        assert module.apply(path) is False
        assert path.read_text() == original


@pytest.mark.parametrize("separator", ["", "\n"])
@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("suite.tsx", "// @vitest-environment jsdom\n\nexport {};\n"),
        ("block.tsx", "/** @vitest-environment jsdom */\n\nexport {};\n"),
        ("module.py", "# Important module comment\n# mypy: ignore-errors\n\nvalue = 1\n"),
    ],
)
def test_repaired_header_preserves_following_metadata(tmp_path, separator, name, body):
    script = Path(__file__).resolve().parents[1] / "scripts" / "license_headers.py"
    spec = importlib.util.spec_from_file_location("license_headers_repair", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / name
    canonical = module.header_for(path)
    path.write_text(canonical.replace("Chavoshi", "Chavashi") + separator + body)

    assert module.apply(path) is True
    assert path.read_text() == canonical + "\n" + body
    assert module.apply(path) is False


@pytest.mark.parametrize("closing", [" */\n", "\n */\n", " *\n */\n"])
@pytest.mark.parametrize(
    "body", ["// @vitest-environment jsdom\n\nexport {};\n", "/** @vitest-environment jsdom */\n\nexport {};\n"]
)
def test_repaired_block_notice_consumes_its_closing_delimiter(tmp_path, closing, body):
    script = Path(__file__).resolve().parents[1] / "scripts" / "license_headers.py"
    spec = importlib.util.spec_from_file_location("license_headers_block_repair", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "suite.tsx"
    path.write_text(
        "/*\n"
        " * Copyright (C) 2026 Erfan Zare Chavoshi\n"
        " * This file is part of Eyvan.\n"
        " *\n"
        " * SPDX-License-Identifier: AGPL-3.0-or-later\n" + closing + "\n" + body
    )

    assert module.apply(path) is True
    assert path.read_text() == module.header_for(path) + "\n" + body
    assert module.apply(path) is False
