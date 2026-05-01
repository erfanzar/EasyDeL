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
"""CLI script: chunked downloader for HuggingFace repos with many files.

Repeatedly downloads roughly ``--chunk-gb`` GiB into a local staging
directory, syncs it to the destination (local path or GCS bucket), then
deletes the staging payload before pulling the next batch. Designed for
directory-style weights such as Zarr where naive ``hf_hub_download``
would fill up local disks.

Exports a ``main`` entry point and a ``ChunkedDownloadArgs`` dataclass
that encodes the CLI flags.

Example - write directly to GCS (no gcsfuse required):

  python scripts/download_hf_repo_chunked_to_gcs.py \\
    --repo-id owner/repo \\
    --out-root gs://my-bucket/easydel-weights \\
    --only-zarr \\
    --chunk-gb 10 \\
    --download-workers 16 \\
    --staging-dir /tmp/easydel-hf-stage \\
    --token $HF_TOKEN

Example - write to a mounted gcsfuse path:

  python scripts/mount_gcsfuse.sh gs://my-bucket/easydel /mnt/gcs
  python scripts/download_hf_repo_chunked_to_gcs.py \\
    --repo-id owner/repo --out-root /mnt/gcs/easydel-weights --only-zarr

Tips:
    - Keep ``--staging-dir`` on a disk with enough free space for
      ``--chunk-gb``.
    - Use ``--dry-run`` first to preview the actions.
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

from eformer.aparser import DataClassArgumentParser
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

GiB = 1024**3


@dataclass(frozen=True)
class DownloadItem:
    """One ``(path, size)`` pair representing a single file to be downloaded.

    Built from the HF API's ``RepoFile`` listing and consumed by the
    chunked-download loop in :func:`main`. Frozen because the
    download orchestrator hashes items into sets to detect
    duplicates and to track progress per file.

    Attributes:
        path (str): File path relative to the repo root, suitable
            for passing to :func:`huggingface_hub.hf_hub_download`
            as ``filename``.
        size (int): File size in bytes; used to assemble batches
            up to ``--chunk-gb``.
    """

    path: str
    size: int


@dataclass
class ChunkedDownloadArgs:
    """Argument schema for the chunked HF repo downloader CLI.

    Controls repository selection, file filtering, batching
    granularity, the local staging directory, and the destination
    (local path or ``gs://`` URI). Designed for very large
    directory-style payloads (Zarr trees, etc.) that would
    overflow local disk if downloaded as a single batch — the
    runner downloads roughly ``chunk_gb`` GiB at a time, syncs to
    the destination, and clears staging before the next chunk.

    Attributes:
        out_root (str): Destination root — either a local path
            (including a GCSFuse mount) or a ``gs://bucket/prefix``
            URI. Determines whether the post-download sync uses a
            local move/copy or ``gsutil``.
        repo_id (str): Repeatable list-style flag; each occurrence
            adds one repo to download.
        repos_file (str | None): Optional path to a file listing
            repos one per line.
        repo_type (str): HF repo type (``"model"``, ``"dataset"``,
            ``"space"``).
        revision (str | None): HF revision/branch/tag.
        token (str | None): HF token; falls back to the standard
            HF auth chain.
        staging_dir (str): Local directory used to stage each
            batch before sync. Should be on a disk with at least
            ``chunk_gb`` free.
        chunk_gb (float): Target batch size in GiB; the
            downloader stops adding files once the running total
            crosses this threshold.
        download_workers (int): Number of parallel download
            threads per batch. ``1`` disables threading.
        path_in_repo (str | None): Optional repo-relative subdirectory
            to restrict the download to (e.g. ``"weights/model.zarr"``).
        only_zarr (bool): When ``True``, only files under a
            ``*.zarr/`` directory are downloaded.
        include (str): Repeatable include-glob list — files must
            match at least one when set.
        exclude (str): Repeatable exclude-glob list applied after
            includes.
        skip_existing (bool): For local destinations, skip files
            already present at the destination.
        force_download (bool): Re-download files even if the
            staging entry already exists.
        local_files_only (bool): Disable Hub downloads (use only
            cached files).
        dry_run (bool): Print planned actions without executing.
        continue_on_error (bool): Skip past failed files instead
            of aborting.
        keep_staging (bool): Do not delete staging payload after
            sync (useful for debugging).
        gsutil_parallel (bool): Use ``gsutil -m`` for parallel
            uploads to ``gs://`` destinations.
        enable_hf_transfer (bool): Toggle the
            ``HF_HUB_ENABLE_HF_TRANSFER`` accelerator.
    """

    out_root: str = field(
        metadata={"help": "Destination root: a local path (including gcsfuse mount) or a gs://bucket/prefix URI."}
    )

    repo_id: str = field(default_factory=list, metadata={"action": "append", "help": "HF repo id (repeatable)."})
    repos_file: str | None = field(default=None, metadata={"help": "File with one repo id per line."})
    repo_type: str = field(default="model", metadata={"help": "HF repo type (model|dataset|space)."})
    revision: str | None = field(default=None, metadata={"help": "HF revision/branch/tag/commit (default: main)."})
    token: str | None = field(
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

    path_in_repo: str | None = field(
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
    """Read repository IDs from a text file (one per line, ``#`` comments allowed).

    Args:
        path: Path to the repos file.

    Returns:
        List of repository ID strings.
    """
    text = Path(path).read_text(encoding="utf-8")
    repo_ids: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        repo_ids.append(line)
    return repo_ids


def _sanitize_repo_id(repo_id: str) -> str:
    """Make a repo id safe for use as a local directory name.

    Args:
        repo_id: HuggingFace repo identifier.

    Returns:
        Repo id with ``/`` replaced by ``__`` and ``:`` by ``_``.
    """
    return repo_id.replace("/", "__").replace(":", "_")


def _repo_out_dir_local(out_root: Path, repo_id: str) -> Path:
    """Return the local destination directory for a repository.

    Args:
        out_root: Local output root.
        repo_id: HuggingFace repo identifier.

    Returns:
        ``out_root/owner/name`` when ``repo_id`` is qualified, otherwise
        ``out_root/repo_id``.
    """
    if "/" in repo_id:
        owner, name = repo_id.split("/", 1)
        return out_root / owner / name
    return out_root / repo_id


def _repo_out_dir_gs(out_root: str, repo_id: str) -> str:
    """Return the GCS destination URI for a repository.

    Args:
        out_root: ``gs://bucket/prefix`` root URI.
        repo_id: HuggingFace repo identifier.

    Returns:
        ``out_root/owner/name`` (or ``out_root/repo_id``) URI string.
    """
    out_root = out_root.rstrip("/")
    if "/" in repo_id:
        owner, name = repo_id.split("/", 1)
        return f"{out_root}/{owner}/{name}"
    return f"{out_root}/{repo_id}"


def _matches_any_glob(name: str, globs: tuple[str, ...]) -> bool:
    """Return True when ``name`` matches at least one glob in ``globs``.

    Args:
        name: Filename or path to test.
        globs: Tuple of fnmatch patterns.

    Returns:
        True if any pattern matches ``name``.
    """
    return any(fnmatch.fnmatch(name, pattern) for pattern in globs)


def _should_keep_path(
    path: str,
    *,
    only_zarr: bool,
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
) -> bool:
    """Decide whether a repo-relative path passes the configured filters.

    Args:
        path: Repo-relative file path.
        only_zarr: When True, require ``.zarr/`` to appear in the path.
        include_globs: Optional tuple of include patterns; the path must
            match at least one when set.
        exclude_globs: Tuple of exclude patterns; the path must not
            match any.

    Returns:
        True when the path should be downloaded.
    """
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
    """Yield ``DownloadItem`` entries for every file in a repo subtree.

    Args:
        api: Authenticated ``HfApi`` instance.
        repo_id: HuggingFace repo identifier.
        repo_type: ``model``, ``dataset``, or ``space``.
        revision: Optional revision/branch/tag/commit.
        token: Optional HuggingFace token.
        path_in_repo: Optional subdirectory within the repo to enumerate.

    Yields:
        ``DownloadItem`` for each file (directories and other entries
        are skipped).
    """
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
    """Print a warning when writing under ``/mnt/gcs`` without gcsfuse.

    Helps catch the common footgun of writing data under a directory
    intended to be a gcsfuse mount and accidentally filling the local
    root disk instead.

    Args:
        out_root: Resolved local output directory.
    """
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
    """Execute a subprocess command (or merely print it on dry runs).

    Args:
        cmd: Argv list to execute.
        dry_run: When True, only print the command and return.

    Raises:
        subprocess.CalledProcessError: When ``check=True`` and the
            command exits non-zero.
    """
    printable = shlex.join(cmd)
    if dry_run:
        print(f"[dry-run] {printable}")
        return
    subprocess.run(cmd, check=True)


def _sync_payload(payload_dir: Path, dest: str | Path, *, dry_run: bool, gsutil_parallel: bool) -> None:
    """Copy the staged batch to its final destination (local or GCS).

    Uses ``gsutil rsync`` for ``gs://`` destinations and ``rsync`` for
    local paths.

    Args:
        payload_dir: Local staging directory containing the batch.
        dest: Destination path (local) or ``gs://`` URI.
        dry_run: When True, only print the planned commands.
        gsutil_parallel: When True, pass ``-m`` to ``gsutil`` for
            multi-threaded operations.
    """
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
    """Download a batch of files into ``staging_payload_dir``.

    Args:
        batch: Files belonging to the current size-bounded batch.
        repo_id: Source HuggingFace repo identifier.
        repo_type: ``model``, ``dataset``, or ``space``.
        revision: Optional repo revision/branch/tag.
        token: Optional HuggingFace token.
        staging_payload_dir: Directory cleared and populated by this batch.
        download_workers: Number of concurrent download threads (``>= 1``).
        force_download: Re-download even when staged files already exist.
        local_files_only: When True, do not contact the Hub.
        dry_run: When True, only print the actions.
        continue_on_error: When True, keep going after individual failures.

    Returns:
        Tuple ``(ok_count, failed_count)`` for this batch.

    Raises:
        Exception: Re-raises the first download failure when
            ``continue_on_error`` is False.
    """
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
        """Inline closure: download one file via :func:`hf_hub_download`.

        Captures ``repo_id``, ``repo_type``, ``revision``, ``token``,
        ``staging_payload_dir``, ``force_download``, and
        ``local_files_only`` from :func:`_download_batch`'s scope.
        Used either sequentially (for a single worker) or
        submitted as a future to a thread pool when
        ``download_workers > 1``.

        Args:
            item: File descriptor produced by
                :func:`_iter_repo_files`. Only ``item.path`` is used;
                ``item.size`` is informational.
        """
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
    """Run the chunked HuggingFace repository downloader.

    Downloads files from one or more HF repos in size-limited batches,
    syncs each batch to a local or GCS destination, then cleans up the
    staging area before proceeding to the next batch.

    Args:
        argv: Command-line arguments. Uses ``sys.argv`` when ``None``.

    Returns:
        Exit code: 0 on success, 2 if any downloads failed.
    """
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
            """Inline closure: download the buffered batch, sync it, and reset state.

            Three behaviours combine here:

            1. Download every file in the rolling ``batch`` buffer
               into ``_payload_dir`` via :func:`_download_batch`.
            2. Sync the payload to its final destination
               (``rsync`` for local, ``gsutil rsync`` for ``gs://``).
            3. Reset the rolling counters (``batch``,
               ``batch_bytes``, ``batch_num``) for the next chunk.

            The per-repo references (``repo_id``, ``payload_dir``,
            ``dest_repo_root``) are bound through default arguments
            so the closure stays correct even after the outer loop
            advances to the next repo. ``nonlocal`` is used for
            running counters that the outer ``main`` needs to see.

            Args:
                _repo_id: Captured ``repo_id`` for the current
                    iteration; do not pass at call sites.
                _payload_dir: Captured staging directory.
                _dest_repo_root: Captured destination root.
            """
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
