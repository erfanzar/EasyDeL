#!/usr/bin/env bash
# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Installs EasyDeL into a job's environment. Eyvan ships this script inline as
# the setup step of ``EasyDeLSetup`` and runs it on every host of the job, as the
# job user, with $EYVAN_PYTHON naming the job's interpreter. The SDK resolves
# the requested branch or tag to one commit before submitting, so every host
# installs the same code and the per-host cache (keyed by this script plus its
# arguments) is reused only for that exact commit.
#
# Three repository layouts are handled, in this order:
#   - a uv workspace with a uv.lock (EasyDeL's foundation layout): the lock is
#     exported for every workspace member, so every package under libs/ is
#     installed from this checkout, and every third-party dependency at the
#     version the lock pins. A plain `pip install .` of that root would take the
#     members from PyPI instead, and `uv sync` would build a separate .venv and
#     remove what it did not install, so neither is used;
#   - libs/*/ packages without a workspace lock: all of them, from this
#     checkout, in one resolver call;
#   - a single package at the repository root.
#
# Never uses sudo, never touches /tmp/libtpu_lockfile, never stops other
# processes and never reads stdin.

set -euo pipefail

REPO=""
REF=""
ACCEL="tpu"
TORCH_CPU=0

die() {
  echo "easydel setup: $*" >&2
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo) [ "$#" -ge 2 ] || die "--repo needs a value"; REPO="$2"; shift 2 ;;
    --ref) [ "$#" -ge 2 ] || die "--ref needs a value"; REF="$2"; shift 2 ;;
    --accelerator) [ "$#" -ge 2 ] || die "--accelerator needs a value"; ACCEL="$2"; shift 2 ;;
    --torch-cpu) TORCH_CPU=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$REPO" ] || die "--repo is required"
case "$REPO" in -*) die "--repo must be a git URL" ;; esac
[[ "$REF" =~ ^[0-9a-f]{40}$ ]] || die "--ref must be a full 40-character commit"
case "$ACCEL" in
  tpu|cuda) EXTRA_NAME="$ACCEL" ;;
  cpu) EXTRA_NAME="" ;;
  *) die "--accelerator must be tpu, cuda or cpu" ;;
esac
PYTHON="${EYVAN_PYTHON:-}"
[ -n "$PYTHON" ] || die "EYVAN_PYTHON is not set; this script runs as an Eyvan setup step"
if ! command -v git >/dev/null 2>&1; then
  echo "easydel setup: git is not installed on this host" >&2
  exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eyvan-easydel.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
SRC="$WORK/src"

echo "easydel setup: $(hostname) installing $REPO at $REF for $ACCEL"
git init -q "$SRC"
git -C "$SRC" remote add origin "$REPO"
if ! git -C "$SRC" fetch -q --depth 1 origin "$REF" 2>/dev/null; then
  # Some servers refuse a fetch by commit id; fetch the branches instead.
  git -C "$SRC" fetch -q --filter=blob:none origin
fi
git -C "$SRC" -c advice.detachedHead=false checkout -q --detach "$REF"
HEAD_COMMIT="$(git -C "$SRC" rev-parse HEAD)"
if [ "$HEAD_COMMIT" != "$REF" ]; then
  echo "easydel setup: checked out $HEAD_COMMIT, expected $REF" >&2
  exit 1
fi
# Relative paths in an exported lock (./libs/easydel) resolve against the checkout.
cd "$SRC"

UV=()
PRIVATE_UV="$WORK/uv/bin/uv"
find_uv() {
  if command -v uv >/dev/null 2>&1 && uv --version >/dev/null 2>&1; then
    UV=(uv)
  elif "$PYTHON" -m uv --version >/dev/null 2>&1; then
    UV=("$PYTHON" -m uv)
  else
    return 1
  fi
}
private_uv() {
  # Into this step's scratch directory, never into the job's environment.
  echo "easydel setup: fetching a current uv for this step"
  if [ "${#UV[@]}" -gt 0 ]; then
    "${UV[@]}" pip install -q --python "$PYTHON" --target "$WORK/uv" uv
  else
    "$PYTHON" -m pip install -q --target "$WORK/uv" uv
  fi
  UV=("$PRIVATE_UV")
}
install_into_job() {
  if [ "${#UV[@]}" -gt 0 ]; then
    "${UV[@]}" pip install --python "$PYTHON" "$@"
  else
    "$PYTHON" -m pip install "$@"
  fi
}

project_name() {
  "$PYTHON" -I -c 'import sys, tomllib
with open(sys.argv[1], "rb") as handle:
    print(tomllib.load(handle)["project"]["name"])' "$1"
}

find_uv || true

if [ "$TORCH_CPU" = 1 ]; then
  echo "easydel setup: installing CPU-only torch"
  install_into_job --index-url https://download.pytorch.org/whl/cpu torch
fi

if [ -f uv.lock ] && grep -q '^\[tool\.uv\.workspace\]' pyproject.toml 2>/dev/null; then
  [ "${#UV[@]}" -gt 0 ] || private_uv
  REQUIREMENTS="$SRC/.eyvan-requirements.txt"
  EXPORT=(export --frozen --all-packages --no-dev --no-editable --no-hashes --no-header --no-emit-project
    -o "$REQUIREMENTS.lock")
  [ -z "$EXTRA_NAME" ] || EXPORT+=(--extra "$EXTRA_NAME")
  if ! "${UV[@]}" "${EXPORT[@]}" >/dev/null; then
    [ "${UV[0]}" != "$PRIVATE_UV" ] || exit 1
    echo "easydel setup: this host's uv cannot read the lock"
    private_uv
    "${UV[@]}" "${EXPORT[@]}" >/dev/null
  fi
  # The workspace root is an empty meta-package; its members are listed by path.
  sed -E '/^\.( |$)/d' "$REQUIREMENTS.lock" > "$REQUIREMENTS"
  echo "easydel setup: installing the workspace lock ($(grep -c '^\./' "$REQUIREMENTS") local package(s))"
  install_into_job -r "$REQUIREMENTS"
else
  PACKAGES=()
  for project in libs/*/pyproject.toml; do
    [ -f "$project" ] || continue
    lib="$(dirname "$project")"
    extra=""
    case "$(basename "$lib")" in easydel|ejkernel) extra="${EXTRA_NAME:+[$EXTRA_NAME]}" ;; esac
    PACKAGES+=("$(project_name "$project")$extra @ file://$SRC/$lib")
  done
  if [ "${#PACKAGES[@]}" -eq 0 ]; then
    [ -f pyproject.toml ] || { echo "easydel setup: no pyproject.toml at the root or under libs/ at $REF" >&2; exit 1; }
    PACKAGES+=("$(project_name pyproject.toml)${EXTRA_NAME:+[$EXTRA_NAME]} @ file://$SRC")
  fi
  echo "easydel setup: installing ${#PACKAGES[@]} package(s)"
  install_into_job "${PACKAGES[@]}"
fi

# Every package under libs/ (or the root package) must now be importable metadata.
"$PYTHON" -I - "$SRC" <<'EOF'
import importlib.metadata as metadata
import pathlib
import sys
import tomllib

root = pathlib.Path(sys.argv[1])
projects = sorted(root.glob("libs/*/pyproject.toml")) or [root / "pyproject.toml"]
missing = []
for project in projects:
    name = tomllib.loads(project.read_text(encoding="utf-8"))["project"]["name"]
    try:
        print(f"easydel setup: {name} {metadata.version(name)}")
    except metadata.PackageNotFoundError:
        missing.append(name)
if missing:
    sys.exit("easydel setup: not installed: " + ", ".join(missing))
EOF
echo "easydel setup: done"
