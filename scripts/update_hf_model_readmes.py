# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, list_models
from huggingface_hub.errors import EntryNotFoundError


def _load_readme_generator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "easydel" / "utils" / "readme_generator.py"
    spec = importlib.util.spec_from_file_location("easydel_readme_generator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import readme generator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_text(path: str | os.PathLike) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_json(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def _infer_task_from_config(config: dict[str, Any]) -> str:
    architectures = config.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = []
    model_type = str(config.get("model_type") or "").lower()

    for arch in architectures:
        arch_str = str(arch)
        if "ForCausalLM" in arch_str:
            return "causal-language-model"
        if "ForSeq2SeqLM" in arch_str:
            return "sequence-to-sequence"
        if "ForSpeechSeq2Seq" in arch_str or "Whisper" in arch_str:
            return "speech-sequence-to-sequence"
        if "ForImageTextToText" in arch_str:
            return "image-text-to-text"
        if "ForZeroShotImageClassification" in arch_str:
            return "zero-shot-image-classification"
        if "ForSequenceClassification" in arch_str:
            return "sequence-classification"

    if "whisper" in model_type:
        return "speech-sequence-to-sequence"
    if any(k in model_type for k in ("clip", "siglip", "vision", "vit")):
        return "vision-module"
    if "diffusion" in model_type:
        return "diffusion-language-model"
    return "causal-language-model"


def _get_attn_mechanism(config: dict[str, Any]) -> str:
    for key in ("attn_mechanism", "attn_mechanism_str", "attention_mechanism"):
        value = config.get(key)
        if value:
            return str(value)
    return "auto"


def _collect_repo_ids(args: argparse.Namespace) -> list[str]:
    repo_ids: list[str] = []

    for repo_id in args.repo_id or []:
        repo_ids.append(repo_id)

    if args.repos_file:
        text = _read_text(args.repos_file)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            repo_ids.append(line)

    if args.author:
        for model in list_models(author=args.author, token=args.token):
            repo_ids.append(model.modelId)

    if args.match:
        repo_ids = [rid for rid in repo_ids if args.match in rid]

    # de-dup while keeping order
    seen: set[str] = set()
    unique: list[str] = []
    for rid in repo_ids:
        if rid in seen:
            continue
        seen.add(rid)
        unique.append(rid)
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate and push HF model-card README.md files from each repo's config.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-id", action="append", default=[], help="Model repo id (repeatable).")
    parser.add_argument("--repos-file", help="Path to a file with one repo id per line.")
    parser.add_argument("--author", help="Update all model repos owned by this HF user/org.")
    parser.add_argument("--match", default=None, help="Only process repo ids containing this substring.")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var / hf auth login).")
    parser.add_argument("--revision", default=None, help="Repo revision to read from (default: main).")
    parser.add_argument("--commit-message", default="Update README.md (EasyDeL auto-generated)", help="Commit message.")
    parser.add_argument("--dry-run", action="store_true", help="Generate but do not push changes.")
    parser.add_argument("--output-dir", default=None, help="Optional directory to write generated READMEs for review.")
    parser.add_argument("--template-dir", default=None, help="Optional custom Jinja template directory.")
    parser.add_argument("--template-name", default=None, help="Optional template filename inside template_dir.")
    args = parser.parse_args(argv)

    repo_ids = _collect_repo_ids(args)
    if not repo_ids:
        raise SystemExit("No repos selected. Use --repo-id, --repos-file, or --author.")

    gen_mod = _load_readme_generator()
    ModelInfo = gen_mod.ModelInfo
    ReadmeGenerator = gen_mod.ReadmeGenerator
    generator = ReadmeGenerator(template_dir=args.template_dir, template_name=args.template_name)

    api = HfApi(token=args.token)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    updated = 0
    skipped = 0
    failed = 0

    for repo_id in repo_ids:
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                repo_type="model",
                token=args.token,
                revision=args.revision,
            )
            config = _load_json(config_path)

            architectures = config.get("architectures") or []
            if not isinstance(architectures, list):
                architectures = []

            model_type = config.get("model_type")
            model_task = _infer_task_from_config(config)
            attn_mechanism = _get_attn_mechanism(config)
            if attn_mechanism == "vanilla":
                attn_mechanism = "blocksparse"
            model_name = repo_id.split("/")[-1]
            model_class = str(architectures[0]) if architectures else str(model_type or "EasyDeLModel")

            model_info = ModelInfo(
                name=model_name,
                type=model_class,
                repo_id=repo_id,
                model_type=str(model_type) if model_type else None,
                model_task=model_task,
                attn_mechanism=attn_mechanism,
            )
            readme = generator.generate_readme(model_info)

            existing = None
            try:
                existing_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="README.md",
                    repo_type="model",
                    token=args.token,
                    revision=args.revision,
                )
                existing = _read_text(existing_path)
            except EntryNotFoundError:
                existing = None

            if existing is not None and existing.strip() == readme.strip():
                print(f"[skip] {repo_id} (no changes)")
                skipped += 1
                continue

            if out_dir is not None:
                safe_name = repo_id.replace("/", "__")
                (out_dir / f"{safe_name}.README.md").write_text(readme, encoding="utf-8")

            if args.dry_run:
                print(f"[dry-run] {repo_id} (generated)")
                updated += 1
                continue

            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_in_repo="README.md",
                path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
                commit_message=args.commit_message,
            )
            print(f"[ok] {repo_id}")
            updated += 1

        except Exception as e:
            failed += 1
            print(f"[error] {repo_id}: {e}", file=sys.stderr)

    print(f"done: updated={updated} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
