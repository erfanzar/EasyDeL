#!/usr/bin/env bash
set -Eeuo pipefail

BUCKET_INPUT="${1:-}"
MOUNT_DIR="${2:-/mnt/gcs}"

if [[ -z "${BUCKET_INPUT}" ]]; then
  echo "Usage: $0 <bucket|gs://bucket[/optional/prefix]> [mount-dir]" >&2
  echo "Example: $0 my-bucket /mnt/gcs" >&2
  echo "Example: $0 gs://my-bucket/models /mnt/gcs" >&2
  exit 2
fi

# Accept "gs://bucket[/prefix]" or "bucket[/prefix]"
BUCKET_INPUT="${BUCKET_INPUT#gs://}"
BUCKET_INPUT="${BUCKET_INPUT#/}"
BUCKET_INPUT="${BUCKET_INPUT%/}"
BUCKET_NAME="${BUCKET_INPUT%%/*}"
ONLY_DIR="${BUCKET_INPUT#${BUCKET_NAME}}"
ONLY_DIR="${ONLY_DIR#/}"

if ! command -v gcsfuse >/dev/null 2>&1; then
  cat >&2 <<'EOF'
gcsfuse not found.

Install Cloud Storage FUSE (Ubuntu/Debian):
  export GCSFUSE_REPO="gcsfuse-$(lsb_release -c -s)"
  echo "deb https://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update && sudo apt-get install -y gcsfuse

Docs: https://cloud.google.com/storage/docs/gcsfuse
EOF
  exit 1
fi

mkdir -p "${MOUNT_DIR}"
if command -v mountpoint >/dev/null 2>&1 && mountpoint -q "${MOUNT_DIR}"; then
  echo "${MOUNT_DIR} is already mounted."
else
  GCSFUSE_ARGS=(--implicit-dirs)
  if gcsfuse --help 2>&1 | grep -q -- '--uid'; then
    MOUNT_UID="${GCSFUSE_UID:-${SUDO_UID:-$(id -u)}}"
    MOUNT_GID="${GCSFUSE_GID:-${SUDO_GID:-$(id -g)}}"
    GCSFUSE_ARGS+=(--uid "${MOUNT_UID}" --gid "${MOUNT_GID}")
  fi
  if [[ -n "${ONLY_DIR}" && "${ONLY_DIR}" != "${BUCKET_NAME}" ]]; then
    echo "Mounting gs://${BUCKET_NAME}/${ONLY_DIR} -> ${MOUNT_DIR}"
    gcsfuse "${GCSFUSE_ARGS[@]}" --only-dir "${ONLY_DIR}" "${BUCKET_NAME}" "${MOUNT_DIR}"
  else
    echo "Mounting gs://${BUCKET_NAME} -> ${MOUNT_DIR}"
    gcsfuse "${GCSFUSE_ARGS[@]}" "${BUCKET_NAME}" "${MOUNT_DIR}"
  fi
fi

cat <<EOF

Mounted.

Tip: downloading Hugging Face checkpoints directly into a gcsfuse mount can be slow.
For large models prefer local temp downloads and only write outputs to the mount:
  python scripts/convert_hf_to_easydel.py ... --convert-mode sequential --torch-streaming-cache temp

Suggested Hugging Face cache env (optional):
  export HF_HOME="${MOUNT_DIR}/hf"
  export TRANSFORMERS_CACHE="\$HF_HOME/transformers"
  export HF_DATASETS_CACHE="\$HF_HOME/datasets"
  export HF_HUB_CACHE="\$HF_HOME/hub"
  export HF_HUB_DISABLE_SYMLINKS_WARNING=1
EOF
