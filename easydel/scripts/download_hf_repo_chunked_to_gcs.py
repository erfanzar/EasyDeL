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

Chunked downloader for Hugging Face repos with many files (e.g. Zarr).
It repeatedly:
  download ~N GiB to a local staging directory -> sync to destination -> delete staging

Write directly to GCS (no gcsfuse required):

  python scripts/download_hf_repo_chunked_to_gcs.py \\
    --repo-id owner/repo \\
    --out-root gs://my-bucket/easydel-weights \\
    --only-zarr \\
    --chunk-gb 10 \\
    --download-workers 16 \\
    --staging-dir /tmp/easydel-hf-stage \\
    --token $HF_TOKEN

Write to a mounted gcsfuse path:

  python scripts/mount_gcsfuse.sh gs://my-bucket/easydel /mnt/gcs
  python scripts/download_hf_repo_chunked_to_gcs.py --repo-id owner/repo --out-root /mnt/gcs/easydel-weights --only-zarr

Tips:
- Keep `--staging-dir` on a disk with enough free space for `--chunk-gb`.
- Use `--dry-run` first to preview.
"""

from __future__ import annotations

import fnmatch
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from eformer.aparser import DataClassArgumentParser
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

GiB = 1024**3


@dataclass(frozen=True)
class DownloadItem:
    path: str
    size: int


@dataclass
class ChunkedDownloadArgs:
    out_root: str = field(
        metadata={"help": "Destination root: a local path (including gcsfuse mount) or a gs://bucket/prefix URI."}
    )

    repo_id: str = field(default_factory=list, metadata={"action": "append", "help": "HF repo id (repeatable)."})
    repos_file: Optional[str] = field(default=None, metadata={"help": "File with one repo id per line."})  # noqa: UP045
    repo_type: str = field(default="model", metadata={"help": "HF repo type (model|dataset|space)."})
    revision: Optional[str] = field(default=None, metadata={"help": "HF revision/branch/tag/commit (default: main)."})  # noqa: UP045
    token: Optional[str] = field(  # noqa: UP045
        default=None,
        metadata={"help": "HF token (or use HF_TOKEN env / `huggingface-cli login`)."},
    )
    staging_dir: str = field(
        default="/tmp/easydel-hf-stage",
        metadata={"help": "Local staging directory (will be created/emptied per repo batch)."},
    )
    chunk_gb: float = field(default=10.0, metadata={"help": "Target batch size in GiB."})
    download_workers: int = field(
        default=8,
        metadata={"help": "Parallel download threads per batch (I/O-bound). Set 1 to disable."},
    )

    path_in_repo: Optional[str] = field(  # noqa: UP045
        default=None,
        metadata={"help": "Optional subfolder in the repo to download (e.g. 'weights/model.zarr')."},
    )
    only_zarr: bool = field(
        default=False,
        metadata={"help": "Only download files under '*.zarr/' directories (matches paths containing '.zarr/')."},
    )
    include: str = field(default_factory=list, metadata={"action": "append", "help": "Glob to include (repeatable)."})
    exclude: str = field(default_factory=list, metadata={"action": "append", "help": "Glob to exclude (repeatable)."})

    skip_existing: bool = field(default=False, metadata={"help": "Skip files that already exist (local out only)."})
    force_download: bool = field(default=False, metadata={"help": "Re-download even if staging file exists."})
    local_files_only: bool = field(default=False, metadata={"help": "Do not download from HF Hub."})
    dry_run: bool = field(default=False, metadata={"help": "Print actions but do nothing."})
    continue_on_error: bool = field(
        default=False, metadata={"help": "Continue with remaining files if one download fails."}
    )
    keep_staging: bool = field(
        default=False, metadata={"help": "Do not delete staging payload after each sync (useful for debugging)."}
    )

    gsutil_parallel: bool = field(default=True, metadata={"help": "Use `gsutil -m` when syncing to gs:// destinations."})
    enable_hf_transfer: bool = field(
        default=False,
        metadata={"help": "Enable hf_transfer accelerated HF downloads (requires `pip install hf_transfer`)."},
    )


def _read_repos_file(path: str | os.PathLike) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    repo_ids: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        repo_ids.append(line)
    return repo_ids


def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(":", "_")


def _repo_out_dir_local(out_root: Path, repo_id: str) -> Path:
    if "/" in repo_id:
        owner, name = repo_id.split("/", 1)
        return out_root / owner / name
    return out_root / repo_id


def _repo_out_dir_gs(out_root: str, repo_id: str) -> str:
    out_root = out_root.rstrip("/")
    if "/" in repo_id:
        owner, name = repo_id.split("/", 1)
        return f"{out_root}/{owner}/{name}"
    return f"{out_root}/{repo_id}"


def _matches_any_glob(name: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in globs)


def _should_keep_path(
    path: str,
    *,
    only_zarr: bool,
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
) -> bool:
    if only_zarr and ".zarr/" not in path:
        return False
    if include_globs and not _matches_any_glob(path, include_globs):
        return False
    if exclude_globs and _matches_any_glob(path, exclude_globs):
        return False
    return True


def _iter_repo_files(
    api: HfApi,
    repo_id: str,
    *,
    repo_type: str,
    revision: str | None,
    token: str | None,
    path_in_repo: str | None,
) -> Iterable[DownloadItem]:
    for item in api.list_repo_tree(
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        recursive=True,
        revision=revision,
        repo_type=repo_type,
        token=token,
    ):
        if not isinstance(item, RepoFile):
            continue
        yield DownloadItem(path=item.path, size=int(item.size))


def _warn_if_mnt_gcs_unmounted(out_root: Path) -> None:
    try:
        if str(out_root).startswith("/mnt/gcs") and not os.path.ismount("/mnt/gcs"):
            print(
                "warning: /mnt/gcs does not look mounted. "
                "If you write outputs there without gcsfuse, you'll fill your root disk.\n"
                "tip: scripts/mount_gcsfuse.sh <bucket> /mnt/gcs",
                file=sys.stderr,
            )
    except Exception:
        return


def _run(cmd: list[str], *, dry_run: bool) -> None:
    printable = shlex.join(cmd)
    if dry_run:
        print(f"[dry-run] {printable}")
        return
    subprocess.run(cmd, check=True)


def _sync_payload(payload_dir: Path, dest: str | Path, *, dry_run: bool, gsutil_parallel: bool) -> None:
    if isinstance(dest, str) and dest.startswith("gs://"):
        cmd = ["gsutil"]
        if gsutil_parallel:
            cmd.append("-m")
        cmd += ["rsync", "-r", str(payload_dir), dest]
        _run(cmd, dry_run=dry_run)
        return

    dest_path = Path(dest).expanduser().resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", f"{payload_dir}/", f"{dest_path}/"]
    _run(cmd, dry_run=dry_run)


def _download_batch(
    batch: list[DownloadItem],
    *,
    repo_id: str,
    repo_type: str,
    revision: str | None,
    token: str | None,
    staging_payload_dir: Path,
    download_workers: int,
    force_download: bool,
    local_files_only: bool,
    dry_run: bool,
    continue_on_error: bool,
) -> tuple[int, int]:
    if staging_payload_dir.exists():
        shutil.rmtree(staging_payload_dir)
    staging_payload_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0

    if dry_run:
        for item in batch:
            print(f"[dry-run] download {repo_id}:{item.path}")
        return len(batch), 0

    def _download_one(item: DownloadItem) -> None:
        hf_hub_download(
            repo_id=repo_id,
            filename=item.path,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=staging_payload_dir,
            force_download=force_download,
            local_files_only=local_files_only,
        )

    max_workers = max(int(download_workers), 1)
    if max_workers == 1 or len(batch) <= 1:
        for item in batch:
            try:
                _download_one(item)
                ok += 1
            except Exception as e:
                failed += 1
                print(f"[fail] {repo_id}:{item.path}: {e}", file=sys.stderr)
                if not continue_on_error:
                    raise
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_download_one, item): item for item in batch}
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    fut.result()
                    ok += 1
                except Exception as e:
                    failed += 1
                    print(f"[fail] {repo_id}:{item.path}: {e}", file=sys.stderr)
                    if not continue_on_error:
                        for pending in futures:
                            pending.cancel()
                        raise

    # `local_dir` downloads create a ".cache/huggingface" folder for metadata; don't sync it to GCS.
    shutil.rmtree(staging_payload_dir / ".cache", ignore_errors=True)

    return ok, failed


def main(argv: list[str] | None = None) -> int:
    parser = DataClassArgumentParser(
        ChunkedDownloadArgs,
        description=(
            "Chunked Hugging Face repo downloader for huge directory-style weights (e.g. .zarr). "
            "Downloads ~N GiB to a local staging dir, syncs to GCS/local output, deletes staging, repeats."
        ),
    )
    (args,) = parser.parse_args_into_dataclasses(args=argv, look_for_args_file=False)

    if args.enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        try:
            import hf_transfer  # noqa: F401 #type:ignore
        except Exception:
            print("warning: `hf_transfer` is not installed. Run: pip install -U hf_transfer", file=sys.stderr)

    repo_ids: list[str] = []
    repo_ids.extend([rid for rid in args.repo_id if rid])
    if args.repos_file:
        repo_ids.extend(_read_repos_file(args.repos_file))
    # de-dup while keeping order
    seen: set[str] = set()
    unique: list[str] = []
    for rid in repo_ids:
        if rid in seen:
            continue
        seen.add(rid)
        unique.append(rid)
    repo_ids = unique
    if not repo_ids:
        raise SystemExit("No repos selected. Use --repo-id/--repos-file.")

    chunk_bytes = int(max(args.chunk_gb, 0.1) * GiB)
    staging_root = Path(args.staging_dir).expanduser().resolve()
    staging_root.mkdir(parents=True, exist_ok=True)

    out_root_is_gs = isinstance(args.out_root, str) and args.out_root.startswith("gs://")
    out_root_local: Path | None = None
    if not out_root_is_gs:
        out_root_local = Path(args.out_root).expanduser().resolve()
        out_root_local.mkdir(parents=True, exist_ok=True)
        _warn_if_mnt_gcs_unmounted(out_root_local)

    include_globs = tuple(args.include)
    exclude_globs = tuple(args.exclude)

    api = HfApi(token=args.token)

    total_failed = 0
    for repo_idx, repo_id in enumerate(repo_ids, start=1):
        repo_tag = f"[{repo_idx}/{len(repo_ids)}] {repo_id}"
        if out_root_is_gs:
            dest_repo_root: str | Path = _repo_out_dir_gs(args.out_root, repo_id)
        else:
            dest_repo_root = _repo_out_dir_local(out_root_local or Path(args.out_root), repo_id)

        print(f"{repo_tag} -> {dest_repo_root}")

        repo_stage_root = staging_root / _sanitize_repo_id(repo_id)
        payload_dir = repo_stage_root / "payload"

        if repo_stage_root.exists() and not args.keep_staging:
            shutil.rmtree(repo_stage_root)
        repo_stage_root.mkdir(parents=True, exist_ok=True)

        batch: list[DownloadItem] = []
        batch_bytes = 0
        batch_num = 1
        downloaded_ok = 0
        repo_failed = 0

        def flush_batch(
            *,
            _repo_id: str = repo_id,
            _payload_dir: Path = payload_dir,
            _dest_repo_root: str | Path = dest_repo_root,
        ) -> None:
            nonlocal batch, batch_bytes, batch_num, downloaded_ok, repo_failed, total_failed
            if not batch:
                return

            size_gib = batch_bytes / GiB
            print(f"  batch {batch_num}: {len(batch)} file(s), ~{size_gib:.2f} GiB")
            ok, failed = _download_batch(
                batch,
                repo_id=_repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                token=args.token,
                staging_payload_dir=_payload_dir,
                download_workers=args.download_workers,
                force_download=args.force_download,
                local_files_only=args.local_files_only,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )
            downloaded_ok += ok
            repo_failed += failed
            total_failed += failed

            _sync_payload(_payload_dir, _dest_repo_root, dry_run=args.dry_run, gsutil_parallel=args.gsutil_parallel)

            if not args.keep_staging and _payload_dir.exists():
                shutil.rmtree(_payload_dir)

            batch = []
            batch_bytes = 0
            batch_num += 1

        try:
            for item in _iter_repo_files(
                api,
                repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                token=args.token,
                path_in_repo=args.path_in_repo,
            ):
                if not _should_keep_path(
                    item.path,
                    only_zarr=args.only_zarr,
                    include_globs=include_globs,
                    exclude_globs=exclude_globs,
                ):
                    continue

                if args.skip_existing and not out_root_is_gs:
                    assert isinstance(dest_repo_root, Path)
                    dest_file = dest_repo_root / item.path
                    try:
                        if dest_file.exists() and dest_file.stat().st_size == item.size:
                            continue
                    except OSError:
                        pass

                if batch and (batch_bytes + item.size) > chunk_bytes:
                    flush_batch()

                batch.append(item)
                batch_bytes += item.size

            flush_batch()

        except Exception as e:
            print(f"{repo_tag} [fatal]: {e}", file=sys.stderr)
            if not args.continue_on_error:
                return 2

        print(f"  done: downloaded_ok={downloaded_ok} failed={repo_failed}")

        if not args.keep_staging and repo_stage_root.exists():
            shutil.rmtree(repo_stage_root, ignore_errors=True)

    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
