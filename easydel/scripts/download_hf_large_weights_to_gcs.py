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
"""
How to use

Download large non-PyTorch artifacts from one or more Hugging Face repos into a local directory
(including a gcsfuse mount).

Example:

  python scripts/download_hf_large_weights_to_gcs.py \\
    --repo-id org/repo \\
    --out-root /mnt/gcs/weights \\
    --min-size-mb 500 \\
    --token $HF_TOKEN

Notes:
- By default, PyTorch weights (.bin/.safetensors/.pt) are excluded.
- This script is size-based, so it is not a good fit for directory-style weights like Zarr
  (many small chunk files). For Zarr/whole-repo downloads, use:
  `python scripts/download_hf_repo_chunked_to_gcs.py ...`
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from eformer.aparser import DataClassArgumentParser
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_EXCLUDE_FILE_GLOBS = (
    "*.md",
    "*.txt",
    "*.rst",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.svg",
    "*.json",
    "*.yaml",
    "*.yml",
    "*.toml",
)

PYTORCH_WEIGHT_GLOBS = (
    "*.bin",
    "*.bin.*",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.safetensors.*",
)


@dataclass(frozen=True)
class RepoFile:
    name: str
    size: int | None


@dataclass
class LargeWeightsArgs:
    out_root: str = field(metadata={"help": "Output root directory (e.g. /mnt/gcs/weights)."})

    repo_id: str = field(default_factory=list, metadata={"action": "append", "help": "Model repo id (repeatable)."})
    repos_file: Optional[str] = field(default=None, metadata={"help": "File with one repo id per line."})  # noqa: UP045
    collection: str = field(
        default_factory=list,
        metadata={
            "action": "append",
            "help": "HF collection URL or 'owner/slug' (repeatable). Example: https://huggingface.co/collections/Qwen/qwen3",
        },
    )

    revision: Optional[str] = field(default=None, metadata={"help": "Repo revision to download from (default: main)."})  # noqa: UP045
    token: Optional[str] = field(default=None, metadata={"help": "HF token (or use HF_TOKEN env / hf auth login)."})  # noqa: UP045
    cache_dir: Optional[str] = field(  # noqa: UP045
        default=None,
        metadata={
            "help": "HF cache dir. Set this to a non-root disk/mount to avoid filling up / (e.g. /mnt/gcs/hf-cache)."
        },
    )

    min_size_mb: int = field(default=500, metadata={"help": "Only download files at/above this size (MiB)."})
    include: str = field(
        default_factory=list,
        metadata={
            "action": "append",
            "help": "Glob to include (repeatable). Example: --include '*.gguf'. If omitted, selects by size + excludes.",
        },
    )
    exclude: str = field(
        default_factory=list,
        metadata={"action": "append", "help": "Glob to exclude (repeatable). Example: --exclude '*.json'."},
    )
    include_pytorch: bool = field(
        default=False,
        metadata={"help": "Also allow PyTorch weights (.bin/.safetensors/.pt). Default: excluded."},
    )
    match_repo: str = field(
        default_factory=list,
        metadata={"action": "append", "help": "Only process repos whose id contains this substring (repeatable)."},
    )

    dry_run: bool = field(default=False, metadata={"help": "Print what would be downloaded, but do nothing."})
    continue_on_error: bool = field(
        default=False, metadata={"help": "Continue with remaining repos even if one download fails."}
    )
    enable_hf_transfer: bool = field(
        default=False,
        metadata={"help": "Enable hf_transfer accelerated HF downloads (requires `pip install hf_transfer`)."},
    )


def _read_models_file(path: str | os.PathLike) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    repo_ids: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        repo_ids.append(line)
    return repo_ids


def _parse_collection(value: str) -> tuple[str, str]:
    value = value.strip()
    if not value:
        raise ValueError("Empty collection value")

    if value.startswith("http://") or value.startswith("https://"):
        # https://huggingface.co/collections/<owner>/<slug>
        match = re.search(r"/collections/([^/]+)/([^/?#]+)", value)
        if not match:
            raise ValueError(f"Could not parse collection URL: {value!r}")
        return match.group(1), match.group(2)

    if "/" not in value:
        raise ValueError(f"Expected collection as 'owner/slug' or URL, got: {value!r}")
    owner, slug = value.split("/", 1)
    return owner, slug


def _fetch_collection_repo_ids(owner: str, slug: str, *, timeout_s: int = 30) -> list[str]:
    url = f"https://huggingface.co/api/collections/{owner}/{slug}"
    data = requests.get(url, timeout=timeout_s).json()
    repo_ids: list[str] = []
    for item in data.get("items", []):
        if item.get("repoType") != "model":
            continue
        rid = item.get("id")
        if isinstance(rid, str) and rid:
            repo_ids.append(rid)
    return repo_ids


def _matches_any_glob(name: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in globs)


def _should_keep_file(
    filename: str,
    size: int | None,
    *,
    min_size_bytes: int,
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
) -> bool:
    if include_globs and not _matches_any_glob(filename, include_globs):
        return False
    if exclude_globs and _matches_any_glob(filename, exclude_globs):
        return False
    if size is None:
        # If size is unknown, only keep when user explicitly included it.
        return bool(include_globs)
    return size >= min_size_bytes


def _repo_out_dir(out_root: Path, repo_id: str) -> Path:
    if "/" in repo_id:
        owner, name = repo_id.split("/", 1)
        return out_root / owner / name
    return out_root / repo_id


def main(argv: list[str] | None = None) -> int:
    parser = DataClassArgumentParser(
        LargeWeightsArgs,
        description="Download large (non-PyTorch) weight files from Hugging Face into a GCS (or local) directory.",
    )
    (args,) = parser.parse_args_into_dataclasses(args=argv, look_for_args_file=False)

    if args.enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        try:
            import hf_transfer  # noqa: F401 # type:ignore
        except Exception:
            print("Warning: `hf_transfer` is not installed. Run: pip install -U hf_transfer", file=sys.stderr)

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    repo_ids: list[str] = []
    repo_ids.extend([rid for rid in args.repo_id if rid])
    if args.repos_file:
        repo_ids.extend(_read_models_file(args.repos_file))

    for collection_value in args.collection:
        owner, slug = _parse_collection(collection_value)
        repo_ids.extend(_fetch_collection_repo_ids(owner, slug))

    # de-dup while keeping order
    seen: set[str] = set()
    unique: list[str] = []
    for rid in repo_ids:
        if rid in seen:
            continue
        seen.add(rid)
        unique.append(rid)
    repo_ids = unique

    if args.match_repo:
        repo_ids = [rid for rid in repo_ids if all(s in rid for s in args.match_repo)]

    if not repo_ids:
        raise SystemExit("No repos selected. Use --repo-id/--repos-file/--collection (and optional --match-repo).")

    include_globs = tuple(args.include)
    exclude_globs = tuple(args.exclude)
    if not args.include_pytorch:
        exclude_globs = exclude_globs + PYTORCH_WEIGHT_GLOBS
    exclude_globs = exclude_globs + DEFAULT_EXCLUDE_FILE_GLOBS

    min_size_bytes = int(args.min_size_mb) * 1024 * 1024

    api = HfApi(token=args.token)

    ok = 0
    skipped = 0
    failed = 0

    for idx, repo_id in enumerate(repo_ids, start=1):
        try:
            info = api.model_info(
                repo_id,
                revision=args.revision,
                files_metadata=True,
                token=args.token,
            )
            siblings = getattr(info, "siblings", None) or []

            files: list[RepoFile] = []
            for sibling in siblings:
                name = getattr(sibling, "rfilename", None)
                if not isinstance(name, str) or not name:
                    continue
                size = getattr(sibling, "size", None)
                size_int = int(size) if isinstance(size, (int, float)) else None
                files.append(RepoFile(name=name, size=size_int))

            selected = [
                f
                for f in files
                if _should_keep_file(
                    f.name,
                    f.size,
                    min_size_bytes=min_size_bytes,
                    include_globs=include_globs,
                    exclude_globs=exclude_globs,
                )
            ]
            if not selected:
                print(f"[{idx}/{len(repo_ids)}] [skip] {repo_id} (no matching files)")
                skipped += 1
                continue

            dest_dir = _repo_out_dir(out_root, repo_id)
            dest_dir.mkdir(parents=True, exist_ok=True)

            total_bytes = sum(f.size or 0 for f in selected)
            total_gb = total_bytes / (1024**3)
            print(f"[{idx}/{len(repo_ids)}] {repo_id}: {len(selected)} file(s), ~{total_gb:.2f} GiB -> {dest_dir}")

            for f in selected:
                size_mb = (f.size or 0) / (1024**2)
                print(f"  - {f.name} ({size_mb:.1f} MiB)")

            if args.dry_run:
                ok += 1
                continue

            for f in selected:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f.name,
                    repo_type="model",
                    revision=args.revision,
                    token=args.token,
                    cache_dir=args.cache_dir,
                    local_dir=str(dest_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

            ok += 1

        except Exception as e:
            print(f"[{idx}/{len(repo_ids)}] [fail] {repo_id}: {e}", file=sys.stderr)
            failed += 1
            if not args.continue_on_error:
                return 2

    print(f"done: ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
