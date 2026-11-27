#!/usr/bin/env bash
# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Installs EasyDeL into a job's environment using a two-tier layered cache so
# switching branches/commits takes seconds instead of reinstalling every heavy
# third-party dependency and recompiling Rust from scratch:
#   - Shared host UV_CACHE_DIR (download third-party wheels once)
#   - Shared host bare git mirror (sub-second incremental fetch + local checkout)
#   - Layer 1: Shared heavy base venv keyed by locked third-party requirements,
#     --torch-cpu, accelerator, and Python ABI (linked via .pth after Layer 2)
#   - Pre-built Rust wheel cache keyed by crate sources + Python ABI/arch
#   - Layer 2: Thin per-commit overlay containing only local monorepo packages
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

probe_writable_dir() {
  mkdir -p "$1" 2>/dev/null || return 1
  local probe="$1/.eyvan-write-probe.$$"
  ( : > "$probe" ) 2>/dev/null || return 1
  rm -f "$probe" 2>/dev/null || true
  return 0
}

# Resolve persistent shared cache root:
#   1. Sibling/subdir of EYVAN_RUNTIME_ENV_DIR when set by the worker
#   2. /var/lib/eyvan (host-wide shared across all users)
#   3. ~/.eyvan (user-scoped)
#   4. Ephemeral $WORK/cache (never referenced via persistent .pth files)
CACHE_ROOT=""
PERSISTENT_CACHE=0
if [ -n "${EYVAN_RUNTIME_ENV_DIR:-}" ]; then
  _env_base="$(basename "$EYVAN_RUNTIME_ENV_DIR")"
  _env_parent="$(dirname "$EYVAN_RUNTIME_ENV_DIR")"
  if [ "$_env_base" = "runtime-envs" ] && probe_writable_dir "$_env_parent"; then
    CACHE_ROOT="$_env_parent"
    PERSISTENT_CACHE=1
  elif probe_writable_dir "$EYVAN_RUNTIME_ENV_DIR/.shared-cache"; then
    CACHE_ROOT="$EYVAN_RUNTIME_ENV_DIR/.shared-cache"
    PERSISTENT_CACHE=1
  fi
fi
if [ "$PERSISTENT_CACHE" = 0 ] && probe_writable_dir "/var/lib/eyvan"; then
  CACHE_ROOT="/var/lib/eyvan"
  PERSISTENT_CACHE=1
fi
if [ "$PERSISTENT_CACHE" = 0 ] && [ -n "${HOME:-}" ] && probe_writable_dir "${HOME}/.eyvan"; then
  CACHE_ROOT="${HOME}/.eyvan"
  PERSISTENT_CACHE=1
fi
if [ "$PERSISTENT_CACHE" = 0 ]; then
  CACHE_ROOT="$WORK/cache"
  mkdir -p "$CACHE_ROOT"
fi

# Shared uv wheel/download cache so third-party wheels download once per host.
if [ -z "${UV_CACHE_DIR:-}" ]; then
  if probe_writable_dir "$CACHE_ROOT/uv-cache"; then
    export UV_CACHE_DIR="$CACHE_ROOT/uv-cache"
  else
    export UV_CACHE_DIR="$WORK/uv-cache"
    mkdir -p "$UV_CACHE_DIR"
  fi
fi

# Portable exclusive file lock: uses flock on Linux, Python fcntl.flock on macOS.
run_locked() {
  local lockfile="$1"
  shift
  mkdir -p "$(dirname "$lockfile")"
  if command -v flock >/dev/null 2>&1; then
    (
      flock -x 9
      "$@"
    ) 9>"$lockfile"
  else
    "$PYTHON" -I -c '
import fcntl, subprocess, sys
with open(sys.argv[1], "a+") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    raise SystemExit(subprocess.call(sys.argv[2:]))
' "$lockfile" "$@"
  fi
}

checkout_repo() {
  local used_mirror=0
  if [ "$PERSISTENT_CACHE" = 1 ]; then
    local mirror_root="$CACHE_ROOT/git-mirrors"
    if probe_writable_dir "$mirror_root"; then
      local repo_hash mirror_dir mirror_lock
      repo_hash="$("$PYTHON" -I -c 'import hashlib, sys; print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest()[:24])' "$REPO")"
      mirror_dir="$mirror_root/${repo_hash}.git"
      mirror_lock="$mirror_root/${repo_hash}.lock"
      if run_locked "$mirror_lock" bash -euo pipefail -c '
        mirror="$1"; repo="$2"; ref="$3"
        if [ ! -f "$mirror/HEAD" ]; then
          rm -rf "$mirror"
          git init -q --bare "$mirror"
          git -C "$mirror" remote add origin "$repo"
        else
          git -C "$mirror" remote set-url origin "$repo"
        fi
        if ! git -C "$mirror" cat-file -e "${ref}^{commit}" 2>/dev/null; then
          if ! git -C "$mirror" fetch -q --no-tags origin "$ref" 2>/dev/null; then
            git -C "$mirror" fetch -q --no-tags origin "+refs/heads/*:refs/heads/*" "+refs/tags/*:refs/tags/*" 2>/dev/null \
              || git -C "$mirror" fetch -q origin
          fi
        fi
        git -C "$mirror" cat-file -e "${ref}^{commit}" 2>/dev/null
      ' _ "$mirror_dir" "$REPO" "$REF"; then
        git init -q "$SRC"
        printf '%s/objects\n' "$mirror_dir" > "$SRC/.git/objects/info/alternates"
        git -C "$SRC" -c advice.detachedHead=false checkout -q --detach "$REF"
        used_mirror=1
      fi
    fi
  fi
  if [ "$used_mirror" = 0 ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin "$REPO"
    if ! git -C "$SRC" fetch -q --depth 1 origin "$REF" 2>/dev/null; then
      # Some servers refuse a fetch by commit id; fetch the branches instead.
      git -C "$SRC" fetch -q --filter=blob:none origin
    fi
    git -C "$SRC" -c advice.detachedHead=false checkout -q --detach "$REF"
  fi
}

echo "easydel setup: $(hostname) installing $REPO at $REF for $ACCEL"
checkout_repo
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
install_into_target() {
  local target_py="$1"
  shift
  if [ "${#UV[@]}" -gt 0 ]; then
    # Ignore [tool.uv.sources] so workspace entries (`workspace = true`, which
    # default to editable=true) can never install .pth pointers into $WORK.
    "${UV[@]}" pip install --python "$target_py" --no-sources "$@"
  else
    "$target_py" -m pip install "$@"
  fi
}
install_into_job() {
  install_into_target "$PYTHON" "$@"
}

project_name() {
  "$PYTHON" -I -c 'import sys, tomllib
with open(sys.argv[1], "rb") as handle:
    print(tomllib.load(handle)["project"]["name"])' "$1"
}

eyvan_installed() {
  "$PYTHON" -I -c 'import importlib.metadata as m; m.version("eyvan")' >/dev/null 2>&1
}

JOB_VENV=""
JOB_SITE=""
LAYERED_CAPABLE=0
detect_job_venv() {
  [ "$PERSISTENT_CACHE" = 1 ] || return 1
  local candidate_venv site_dir
  candidate_venv="$(dirname "$(dirname "$PYTHON")")"
  [ -f "$candidate_venv/pyvenv.cfg" ] || return 1
  site_dir="$("$PYTHON" -I -c '
import pathlib, sys
ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
p = pathlib.Path(sys.argv[1]) / "lib" / ver / "site-packages"
print(p if p.is_dir() else "")
' "$candidate_venv" 2>/dev/null || true)"
  if [ -n "$site_dir" ] && [ -w "$site_dir" ]; then
    JOB_VENV="$candidate_venv"
    JOB_SITE="$site_dir"
    LAYERED_CAPABLE=1
    return 0
  fi
  return 1
}

ensure_cargo() {
  if [ -n "${HOME:-}" ]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi
  if command -v cargo >/dev/null 2>&1; then
    echo "easydel setup: $(cargo --version)"
    return 0
  fi
  if [ -z "${HOME:-}" ]; then
    export HOME="$WORK/home"
    mkdir -p "$HOME"
  fi
  echo "easydel setup: cargo not found; installing a minimal stable Rust toolchain with rustup"
  if ! command -v curl >/dev/null 2>&1; then
    echo "easydel setup: curl is not installed; cannot bootstrap rustup for Rust extensions" >&2
    exit 1
  fi
  local rustup_init="$WORK/rustup-init.sh"
  local rustup_log="$WORK/rustup.log"
  curl --proto '=https' --tlsv1.2 -fsSL --max-time 120 https://sh.rustup.rs -o "$rustup_init"
  local install_cmd=(sh "$rustup_init" -y --profile minimal --default-toolchain stable --no-modify-path)
  if command -v timeout >/dev/null 2>&1; then
    install_cmd=(timeout 900 "${install_cmd[@]}")
  fi
  if ! "${install_cmd[@]}" >"$rustup_log" 2>&1; then
    echo "easydel setup: rustup install failed; log tail follows" >&2
    tail -n 40 "$rustup_log" >&2 || true
    exit 1
  fi
  export PATH="${HOME}/.cargo/bin:${PATH}"
  echo "easydel setup: $(cargo --version)"
}

ensure_base_env() {
  local req_file="$1"
  local base_hash base_root base_env base_marker base_lock base_site
  base_root="$CACHE_ROOT/base-envs"
  probe_writable_dir "$base_root" || return 1
  base_hash="$("$PYTHON" -I -c '
import hashlib, json, pathlib, platform, sys, sysconfig
lines = []
p = pathlib.Path(sys.argv[1])
if p.is_file():
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            lines.append(s)
payload = {
    "format": "eyvan-base-v1",
    "impl": sys.implementation.name,
    "py": list(sys.version_info[:2]),
    "soabi": sysconfig.get_config_var("SOABI") or "",
    "system": platform.system(),
    "machine": platform.machine(),
    "torch_cpu": sys.argv[2] == "1",
    "accel": sys.argv[3],
    "reqs": sorted(lines),
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24])
' "$req_file" "$TORCH_CPU" "$ACCEL")"
  base_env="$base_root/$base_hash"
  base_marker="$base_env/.eyvan-complete"
  base_lock="$base_root/${base_hash}.lock"

  if [ ! -f "$base_marker" ]; then
    echo "easydel setup: building shared base environment ($base_hash)"
    run_locked "$base_lock" bash -euo pipefail -c '
      base_env="$1"; base_marker="$2"; job_py="$3"; req_file="$4"; torch_cpu="$5"; shift 5
      uv_cmd=("$@")
      [ ! -f "$base_marker" ] || exit 0
      rm -rf "$base_env"
      cleanup() { rm -rf "$base_env"; }
      trap cleanup ERR
      if [ "${#uv_cmd[@]}" -gt 0 ]; then
        if ! "${uv_cmd[@]}" venv --python "$job_py" --system-site-packages "$base_env" >/dev/null 2>&1; then
          rm -rf "$base_env"
          "$job_py" -m venv --system-site-packages "$base_env"
        fi
      else
        "$job_py" -m venv --system-site-packages "$base_env"
      fi
      base_py="$base_env/bin/python"
      if [ ! -x "$base_py" ]; then
        echo "easydel setup: base venv interpreter missing at $base_py" >&2
        exit 1
      fi
      if [ "$torch_cpu" = 1 ]; then
        if [ "${#uv_cmd[@]}" -gt 0 ]; then
          "${uv_cmd[@]}" pip install --python "$base_py" --no-sources --index-url https://download.pytorch.org/whl/cpu torch
        else
          "$base_py" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
        fi
      fi
      if [ -s "$req_file" ]; then
        if [ "${#uv_cmd[@]}" -gt 0 ]; then
          "${uv_cmd[@]}" pip install --python "$base_py" --no-sources --no-deps -r "$req_file"
        else
          "$base_py" -m pip install --no-deps -r "$req_file"
        fi
      fi
      : > "$base_marker"
      trap - ERR
    ' _ "$base_env" "$base_marker" "$PYTHON" "$req_file" "$TORCH_CPU" ${UV[@]+"${UV[@]}"}
  else
    echo "easydel setup: reusing cached base environment ($base_hash)"
  fi

  base_site="$("$PYTHON" -I -c '
import pathlib, sys
ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
p = pathlib.Path(sys.argv[1]) / "lib" / ver / "site-packages"
print(p if p.is_dir() else "")
' "$base_env" 2>/dev/null || true)"
  [ -n "$base_site" ] || return 1

  # Named _eyvan_base_environment.pth so CPython processes it AFTER the job
  # overlay site-packages (always first on sys.path) and BEFORE
  # _eyvan_parent_environment.pth (alphabetical: _eyvan_b < _eyvan_p).
  "$PYTHON" -I -c '
import pathlib, sys
pth = pathlib.Path(sys.argv[1]) / "_eyvan_base_environment.pth"
pth.write_text(f"import site; site.addsitedir({sys.argv[2]!r})\n", encoding="utf-8")
' "$JOB_SITE" "$base_site"
  return 0
}

find_cached_wheel() {
  local entry_dir="$1"
  [ -f "$entry_dir/.eyvan-complete" ] || return 1
  local whl
  for whl in "$entry_dir"/*.whl; do
    if [ -f "$whl" ]; then
      printf '%s\n' "$whl"
      return 0
    fi
  done
  return 1
}

install_rust_package() {
  local rel_dir="$1"
  local clean_dir="${rel_dir#./}"
  local pkg_dir="$SRC/$clean_dir"
  local pkg_name pkg_safe rust_hash wheel_root wheel_entry wheel_lock cached_whl

  pkg_name="$(project_name "$pkg_dir/pyproject.toml")"
  pkg_safe="$("$PYTHON" -I -c 'import re, sys; print(re.sub(r"[^A-Za-z0-9._-]+", "_", sys.argv[1]).lower())' "$pkg_name")"

  if [ "$PERSISTENT_CACHE" = 1 ] && probe_writable_dir "$CACHE_ROOT/wheel-cache"; then
    wheel_root="$CACHE_ROOT/wheel-cache"
    rust_hash="$("$PYTHON" -I -c '
import hashlib, pathlib, platform, sys, sysconfig
src_root = pathlib.Path(sys.argv[1])
pkg_dir = src_root / sys.argv[2]
h = hashlib.sha256()
h.update(b"eyvan-rust-wheel-v1\0")
for part in (
    sys.argv[3],
    sys.implementation.name,
    f"{sys.version_info.major}.{sys.version_info.minor}",
    sysconfig.get_config_var("SOABI") or "",
    platform.system(),
    platform.machine(),
):
    h.update(part.encode("utf-8") + b"\0")
skip = {"target", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
files = []
root_lock = src_root / "Cargo.lock"
if root_lock.is_file():
    files.append(("__workspace__/Cargo.lock", root_lock))
for path in sorted(pkg_dir.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(pkg_dir)
    if any(part in skip for part in rel.parts) or path.suffix in {".pyc", ".pyo"}:
        continue
    files.append((rel.as_posix(), path))
for rel_name, path in files:
    data = path.read_bytes()
    h.update(rel_name.encode("utf-8") + b"\0" + str(len(data)).encode("ascii") + b"\0" + data + b"\0")
print(h.hexdigest()[:24])
' "$SRC" "$clean_dir" "$pkg_safe")"
    wheel_entry="$wheel_root/${pkg_safe}-${rust_hash}"
    wheel_lock="$wheel_root/${pkg_safe}-${rust_hash}.lock"

    if cached_whl="$(find_cached_wheel "$wheel_entry")"; then
      echo "easydel setup: reusing cached wheel for $pkg_name ($rust_hash)"
      install_into_job --no-deps "$cached_whl"
      return 0
    fi

    ensure_cargo
    echo "easydel setup: building and caching wheel for $pkg_name ($rust_hash)"
    run_locked "$wheel_lock" bash -euo pipefail -c '
      wheel_entry="$1"; pkg_dir="$2"; job_py="$3"; build_out="$4"; shift 4
      uv_cmd=("$@")
      if [ -f "$wheel_entry/.eyvan-complete" ]; then
        exit 0
      fi
      rm -rf "$build_out"
      mkdir -p "$build_out"
      built=0
      if [ "${#uv_cmd[@]}" -gt 0 ]; then
        if "${uv_cmd[@]}" build --wheel --python "$job_py" --out-dir "$build_out" --no-create-gitignore "$pkg_dir"; then
          built=1
        fi
      fi
      if [ "$built" = 0 ]; then
        "$job_py" -m pip wheel --no-deps --wheel-dir "$build_out" "$pkg_dir" || true
      fi
      found=""
      for whl in "$build_out"/*.whl; do
        if [ -f "$whl" ]; then
          found="$whl"
          break
        fi
      done
      if [ -n "$found" ]; then
        stage="${wheel_entry}.tmp.$$"
        rm -rf "$stage" "$wheel_entry"
        mkdir -p "$stage"
        cp "$found" "$stage/"
        : > "$stage/.eyvan-complete"
        mv "$stage" "$wheel_entry"
      fi
    ' _ "$wheel_entry" "$pkg_dir" "$PYTHON" "$WORK/whl-$pkg_safe" ${UV[@]+"${UV[@]}"}

    if cached_whl="$(find_cached_wheel "$wheel_entry")"; then
      install_into_job --no-deps "$cached_whl"
      return 0
    fi
  else
    ensure_cargo
  fi

  install_into_job --no-deps "$rel_dir"
}

if [ -n "${HOME:-}" ]; then
  export PATH="${HOME}/.cargo/bin:${PATH}"
fi
find_uv || true
detect_job_venv || true

NEEDS_RUST=0
VERIFY_PROJECTS=()

if [ -f uv.lock ] && grep -q '^\[tool\.uv\.workspace\]' pyproject.toml 2>/dev/null; then
  [ "${#UV[@]}" -gt 0 ] || private_uv
  REQUIREMENTS="$SRC/.eyvan-requirements.txt"
  if [ -f libs/easydel/pyproject.toml ]; then
    EXPORT=(export --frozen --package easydel --no-dev --no-editable --no-hashes --no-header
      -o "$REQUIREMENTS.lock")
  else
    EXPORT=(export --frozen --all-packages --no-dev --no-editable --no-hashes --no-header --no-emit-project
      -o "$REQUIREMENTS.lock")
  fi
  [ -z "$EXTRA_NAME" ] || EXPORT+=(--extra "$EXTRA_NAME")
  if ! "${UV[@]}" "${EXPORT[@]}" >/dev/null; then
    [ "${UV[0]}" != "$PRIVATE_UV" ] || exit 1
    echo "easydel setup: this host's uv cannot read the lock"
    private_uv
    "${UV[@]}" "${EXPORT[@]}" >/dev/null
  fi
  # Drop the empty root meta-package (if emitted). If the job's environment
  # already provides eyvan from the worker, do not overwrite it from the EasyDeL
  # checkout or force server dependencies into the job venv.
  if eyvan_installed && grep -Eq '^\./libs/eyvan([ ;]|$)' "$REQUIREMENTS.lock"; then
    sed -E '/^\.( |$)/d; /^\.\/libs\/eyvan([ ;]|$)/d' "$REQUIREMENTS.lock" > "$REQUIREMENTS"
  else
    sed -E '/^\.( |$)/d' "$REQUIREMENTS.lock" > "$REQUIREMENTS"
  fi

  THIRD_PARTY_REQS="$WORK/third-party-reqs.txt"
  LOCAL_PY_REQS="$WORK/local-py-reqs.txt"
  LOCAL_RUST_DIRS_FILE="$WORK/local-rust-dirs.txt"
  VERIFY_PROJECTS_FILE="$WORK/verify-projects.txt"

  "$PYTHON" -I -c '
import pathlib, sys
reqs = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
src = pathlib.Path(sys.argv[2])
third_party, local_py, local_rust, verify = [], [], [], []
buf = ""
logical = []
for raw in reqs:
    s = raw.strip()
    if not s or s.startswith("#"):
        if buf:
            logical.append(buf)
            buf = ""
        continue
    if s.endswith("\\"):
        buf = (buf + " " + s[:-1].strip()).strip()
    else:
        logical.append((buf + " " + s).strip() if buf else s)
        buf = ""
if buf:
    logical.append(buf)

for line in logical:
    if line.startswith("./"):
        local_dir = line.split(";", 1)[0].strip().split()[0]
        rel = local_dir[2:]
        pkg_root = src / rel
        if (pkg_root / "pyproject.toml").is_file():
            verify.append(str(pkg_root / "pyproject.toml"))
        if (pkg_root / "Cargo.toml").is_file():
            local_rust.append(local_dir)
        else:
            local_py.append(line)
    else:
        third_party.append(line)

pathlib.Path(sys.argv[3]).write_text("\n".join(third_party) + ("\n" if third_party else ""), encoding="utf-8")
pathlib.Path(sys.argv[4]).write_text("\n".join(local_py) + ("\n" if local_py else ""), encoding="utf-8")
pathlib.Path(sys.argv[5]).write_text("\n".join(local_rust) + ("\n" if local_rust else ""), encoding="utf-8")
pathlib.Path(sys.argv[6]).write_text("\n".join(verify) + ("\n" if verify else ""), encoding="utf-8")
' "$REQUIREMENTS" "$SRC" "$THIRD_PARTY_REQS" "$LOCAL_PY_REQS" "$LOCAL_RUST_DIRS_FILE" "$VERIFY_PROJECTS_FILE"

  while IFS= read -r proj; do
    [ -n "$proj" ] && VERIFY_PROJECTS+=("$proj")
  done < "$VERIFY_PROJECTS_FILE"

  LOCAL_RUST_DIRS=()
  while IFS= read -r rdir; do
    [ -n "$rdir" ] && LOCAL_RUST_DIRS+=("$rdir")
  done < "$LOCAL_RUST_DIRS_FILE"

  if [ "$LAYERED_CAPABLE" = 1 ]; then
    if [ -s "$THIRD_PARTY_REQS" ] || [ "$TORCH_CPU" = 1 ]; then
      ensure_base_env "$THIRD_PARTY_REQS"
    fi
    if [ "${#LOCAL_RUST_DIRS[@]}" -gt 0 ]; then
      for rdir in "${LOCAL_RUST_DIRS[@]}"; do
        install_rust_package "$rdir"
      done
    fi
    if [ -s "$LOCAL_PY_REQS" ]; then
      local_py_count="$(wc -l < "$LOCAL_PY_REQS" | tr -d ' ')"
      echo "easydel setup: installing $local_py_count local Python package(s) into thin overlay"
      install_into_job --no-deps -r "$LOCAL_PY_REQS"
    fi
  else
    if [ "$TORCH_CPU" = 1 ]; then
      echo "easydel setup: installing CPU-only torch"
      install_into_job --index-url https://download.pytorch.org/whl/cpu torch
    fi
    if [ "${#LOCAL_RUST_DIRS[@]}" -gt 0 ]; then
      for rdir in "${LOCAL_RUST_DIRS[@]}"; do
        install_rust_package "$rdir"
      done
      # Install remaining third-party + local Python packages without re-building Rust.
      NON_RUST_REQS="$WORK/non-rust-reqs.txt"
      cat "$THIRD_PARTY_REQS" "$LOCAL_PY_REQS" > "$NON_RUST_REQS"
      if [ -s "$NON_RUST_REQS" ]; then
        echo "easydel setup: installing remaining workspace lock entries"
        install_into_job --no-deps -r "$NON_RUST_REQS"
      fi
    else
      echo "easydel setup: installing the workspace lock ($(grep -c '^\./' "$REQUIREMENTS" || true) local package(s))"
      # The exported lock already lists the complete resolved dependency graph;
      # always passing --no-deps installs every pinned entry directly without
      # re-resolving [tool.uv.sources] (which would otherwise rewrite workspace
      # siblings into editable .pth pointers into $WORK, or overwrite the worker's
      # eyvan from PyPI).
      install_into_job --no-deps -r "$REQUIREMENTS"
    fi
  fi
else
  if [ "$TORCH_CPU" = 1 ]; then
    echo "easydel setup: installing CPU-only torch"
    install_into_job --index-url https://download.pytorch.org/whl/cpu torch
  fi
  PACKAGES=()
  HAS_EYVAN=0
  if eyvan_installed; then
    HAS_EYVAN=1
  fi
  for project in libs/*/pyproject.toml; do
    [ -f "$project" ] || continue
    lib="$(dirname "$project")"
    base="$(basename "$lib")"
    if [ "$base" = "eyvan" ] && [ "$HAS_EYVAN" = 1 ]; then
      continue
    fi
    [ -f "$lib/Cargo.toml" ] && NEEDS_RUST=1
    extra=""
    case "$base" in easydel|ejkernel) extra="${EXTRA_NAME:+[$EXTRA_NAME]}" ;; esac
    PACKAGES+=("$(project_name "$project")$extra @ file://$SRC/$lib")
    VERIFY_PROJECTS+=("$SRC/$project")
  done
  if [ "${#PACKAGES[@]}" -eq 0 ]; then
    [ -f pyproject.toml ] || { echo "easydel setup: no pyproject.toml at the root or under libs/ at $REF" >&2; exit 1; }
    [ -f Cargo.toml ] && NEEDS_RUST=1
    PACKAGES+=("$(project_name pyproject.toml)${EXTRA_NAME:+[$EXTRA_NAME]} @ file://$SRC")
    VERIFY_PROJECTS+=("$SRC/pyproject.toml")
  fi
  [ "$NEEDS_RUST" = 0 ] || ensure_cargo
  echo "easydel setup: installing ${#PACKAGES[@]} package(s)"
  install_into_job "${PACKAGES[@]}"
fi

if [ "${#VERIFY_PROJECTS[@]}" -eq 0 ]; then
  if [ -f "$SRC/libs/easydel/pyproject.toml" ]; then
    VERIFY_PROJECTS=("$SRC/libs/easydel/pyproject.toml")
  elif [ -f "$SRC/pyproject.toml" ]; then
    VERIFY_PROJECTS=("$SRC/pyproject.toml")
  fi
fi

# Every installed local package must expose importable metadata and must not
# carry editable references or .pth pointers into the temporary $WORK checkout,
# which the EXIT trap removes immediately after this script completes.
"$PYTHON" -I - "$WORK" "${VERIFY_PROJECTS[@]}" <<'EOF'
import importlib.metadata as metadata
import json
import pathlib
import sys
import tomllib
from urllib.parse import unquote, urlsplit

work_dir = pathlib.Path(sys.argv[1]).resolve()


def _is_under_work(candidate: pathlib.Path) -> bool:
    try:
        candidate.resolve().relative_to(work_dir)
    except ValueError:
        return False
    return True


def _file_url_path(url: str) -> pathlib.Path | None:
    parsed = urlsplit(url)
    if parsed.scheme != "file":
        return None
    return pathlib.Path(unquote(parsed.path))


missing = []
for project in (pathlib.Path(arg) for arg in sys.argv[2:]):
    name = tomllib.loads(project.read_text(encoding="utf-8"))["project"]["name"]
    try:
        print(f"easydel setup: {name} {metadata.version(name)}")
    except metadata.PackageNotFoundError:
        missing.append(name)

work_leaks = []
for dist in metadata.distributions():
    dist_name = dist.metadata.get("Name") or dist.name
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except ValueError:
            direct_url = {}
        dir_info = direct_url.get("dir_info") if isinstance(direct_url, dict) else None
        url = direct_url.get("url") if isinstance(direct_url, dict) else None
        if isinstance(dir_info, dict) and dir_info.get("editable") is True and isinstance(url, str):
            target = _file_url_path(url)
            if target is not None and _is_under_work(target):
                work_leaks.append(f"{dist_name} (editable direct_url.json -> {target})")
                continue
    for rel in dist.files or ():
        if rel.suffix != ".pth":
            continue
        pth_path = pathlib.Path(dist.locate_file(rel))
        try:
            lines = pth_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            entry = raw_line.strip()
            if not entry or entry.startswith("#") or entry.startswith("import "):
                continue
            target = pathlib.Path(entry)
            if not target.is_absolute():
                target = pth_path.parent / target
            if _is_under_work(target):
                work_leaks.append(f"{dist_name} ({rel.name} -> {entry})")
                break
if missing:
    sys.exit("easydel setup: not installed: " + ", ".join(missing))
if work_leaks:
    sys.exit(
        "easydel setup: installed editable reference into temporary checkout that will be deleted: "
        + ", ".join(work_leaks)
    )
EOF
echo "easydel setup: done"
