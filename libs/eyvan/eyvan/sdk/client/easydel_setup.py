# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Built-in setup step that installs EasyDeL from a git branch, tag, or commit."""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from eyvan.records import WorkloadSetup

#: The bash step shipped inline with every job that uses ``EasyDeLSetup``.
SCRIPT_PATH = Path(__file__).with_name("easydel_setup.sh")
DEFAULT_REPO = "erfanzar/EasyDeL"
DEFAULT_BRANCH = "main"
ACCELERATORS = ("tpu", "cuda", "cpu")

_COMMIT = re.compile(r"[0-9a-f]{40}")
_OWNER_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+-]*")
_RESOLVE_TIMEOUT_S = 20.0
_MAX_ADVERTISEMENT_BYTES = 16 * 1024 * 1024


def repository_url(repo: str) -> str:
    """Return the https git URL for ``owner/name`` on GitHub, or an https URL as given."""

    if not isinstance(repo, str) or not repo or repo != repo.strip():
        raise ValueError("repo must be a non-empty owner/name or https:// git URL")
    if repo.startswith("https://"):
        if any(character.isspace() or character == "\x00" for character in repo):
            raise ValueError("repo URL must not contain whitespace")
        return repo.rstrip("/")
    if "://" in repo:
        raise ValueError("repo must be owner/name on GitHub or an https:// git URL")
    name = repo.removesuffix(".git")
    if not _OWNER_REPO.fullmatch(name):
        raise ValueError(f"repo {repo!r} is not owner/name")
    return f"https://github.com/{name}.git"


def parse_advertised_refs(body: bytes) -> dict[str, str]:
    """Map ref names to commits from a git smart-HTTP ``info/refs`` advertisement."""

    refs: dict[str, str] = {}
    offset = 0
    while offset < len(body):
        try:
            length = int(body[offset : offset + 4], 16)
        except ValueError as error:
            raise ValueError("the repository did not answer with a git ref advertisement") from error
        if length == 0:  # flush packet
            offset += 4
            continue
        if length < 4 or offset + length > len(body):
            raise ValueError("the repository's ref advertisement is truncated")
        line = body[offset + 4 : offset + length].split(b"\0", 1)[0].rstrip(b"\n").decode("ascii", "replace")
        offset += length
        commit, _, name = line.partition(" ")
        if name and _COMMIT.fullmatch(commit):
            refs[name] = commit
    return refs


def _fetch_advertised_refs(repository: str) -> bytes:
    """Download the ref advertisement of an https git repository."""

    url = f"{repository}/info/refs?service=git-upload-pack"
    request = urllib.request.Request(url, headers={"User-Agent": "git/2 eyvan-sdk"})
    try:
        with urllib.request.urlopen(request, timeout=_RESOLVE_TIMEOUT_S) as response:
            if not str(response.geturl()).startswith("https://"):
                raise ValueError(f"{repository} redirected away from https")
            body: bytes = response.read(_MAX_ADVERTISEMENT_BYTES + 1)
    except OSError as error:
        raise ValueError(f"could not list the branches of {repository}: {error}") from error
    if len(body) > _MAX_ADVERTISEMENT_BYTES:
        raise ValueError(f"the ref list of {repository} is larger than {_MAX_ADVERTISEMENT_BYTES} bytes")
    return body


def resolve_commit(repository: str, ref: str) -> str:
    """Return the commit a branch or tag names now; a full commit is returned as is."""

    if _COMMIT.fullmatch(ref):
        return ref
    refs = parse_advertised_refs(_fetch_advertised_refs(repository))
    names = (ref, f"{ref}^{{}}") if ref.startswith("refs/") else ()
    # A peeled tag (``^{}``) names the commit; the bare tag may name a tag object.
    names += (f"refs/heads/{ref}", f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}")
    for name in names:
        if name in refs:
            return refs[name]
    raise ValueError(
        f"{ref!r} is not a branch or tag of {repository}; pass a full 40-character commit to skip the lookup"
    )


@dataclass(frozen=True, slots=True)
class EasyDeLSetup:
    """Install EasyDeL on every host of a job, before it starts.

    ``branch`` is a branch, tag or full commit of ``repo`` (``owner/name`` on
    GitHub, or an ``https://`` git URL). It is resolved to one commit when the
    job is submitted, so every host installs the same code and a host's cached
    install is reused only while that commit is the one asked for. Both the
    monorepo layout (``libs/easydel``, ``libs/eformer``, ...) and a single
    package at the repository root are handled.

    ``accelerator`` picks the extras (``tpu``, ``cuda`` or ``cpu`` for none);
    ``torch_cpu`` also installs CPU-only torch, which EasyDeL uses to read
    Hugging Face checkpoints.
    """

    branch: str = DEFAULT_BRANCH
    repo: str = DEFAULT_REPO
    accelerator: str = "tpu"
    torch_cpu: bool = False

    def __post_init__(self) -> None:
        repository_url(self.repo)
        branch = self.branch
        if not isinstance(branch, str) or not _REF.fullmatch(branch) or ".." in branch or branch.endswith("/"):
            raise ValueError(f"branch {branch!r} is not a valid git branch, tag or commit")
        if self.accelerator not in ACCELERATORS:
            raise ValueError(f"accelerator must be one of {', '.join(ACCELERATORS)}")
        if not isinstance(self.torch_cpu, bool):
            raise ValueError("torch_cpu must be True or False")

    @property
    def repository_url(self) -> str:
        """The https git URL the hosts clone."""

        return repository_url(self.repo)

    def to_workload_setup(self) -> WorkloadSetup:
        """Resolve ``branch`` and return the setup step every host runs."""

        repository = self.repository_url
        commit = resolve_commit(repository, self.branch)
        args: tuple[str, ...] = ("--repo", repository, "--ref", commit, "--accelerator", self.accelerator)
        if self.torch_cpu:
            args += ("--torch-cpu",)
        return WorkloadSetup(script=SCRIPT_PATH.read_text(encoding="utf-8"), args=args)
