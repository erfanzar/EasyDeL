#!/usr/bin/env bash
set -Eeuo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

# ---------- Argument parsing ----------
EASYDEL_BRANCH=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [--branch <branch-or-tag-or-sha>]

Options:
  --branch <ref>   Git branch / tag / commit of EasyDeL to clone onto the TPU
                   hosts and install (default: main). Setup is standalone — it
                   fetches the repo from git on each host, so no local checkout
                   is required. Override the source with EASYDEL_REPO_URL.
  -h, --help       Show this help message.
EOF
}

while (( "$#" )); do
  case "$1" in
    --branch)
      if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
        log_error "--branch requires a value (branch / tag / commit SHA)."
        usage; exit 1
      fi
      EASYDEL_BRANCH="$2"
      shift 2
      ;;
    --branch=*)
      EASYDEL_BRANCH="${1#--branch=}"
      if [ -z "$EASYDEL_BRANCH" ]; then
        log_error "--branch= requires a value."
        usage; exit 1
      fi
      shift
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      log_error "Unknown argument: $1"
      usage; exit 1
      ;;
  esac
done

# Default branch when --branch is not given. The workspace is fetched from git on each TPU
# host (see clone_workspace_on_tpu), so setup is fully standalone — no local checkout needed.
EASYDEL_BRANCH="${EASYDEL_BRANCH:-main}"

metadata_value() {
  local path="$1"
  curl -fsS "http://metadata.google.internal/computeMetadata/v1/${path}" -H "Metadata-Flavor: Google" 2>/dev/null || true
}

# Ensure ~/.local/bin in PATH for current session
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) log_info "Adding $HOME/.local/bin to PATH for the current session."; export PATH="$HOME/.local/bin:$PATH";;
esac

# Persist PATH
log_info "Checking shell configuration for persistent PATH..."
python - << 'PY'
import os, sys
BLUE = '\033[0;34m'; GREEN = '\033[0;32m'; NC = '\033[0m'
def info(m): print(f'{BLUE}[INFO]{NC} (Python) {m}', file=sys.stderr)
def ok(m):   print(f'{GREEN}[SUCCESS]{NC} (Python) {m}', file=sys.stderr)

line = 'export PATH="$HOME/.local/bin:$PATH"'
home = os.path.expanduser('~')
cands = [os.path.join(home, '.zshrc'), os.path.join(home, '.bashrc')]
target = next((c for c in cands if os.path.exists(c)), os.path.join(home, '.bashrc'))
info(f'Checking shell configuration file: {target}')
try:
    content = ''
    if os.path.exists(target):
        with open(target) as f: content = f.read()
    if line in content:
        info('PATH configuration already exists. No changes needed.')
    else:
        with open(target, 'a') as f:
            f.write('\n# Added by script to include local binaries\n' + line + '\n')
        ok(f'Successfully updated {target}. Run "source {target}" or restart your terminal for it to take effect.')
except Exception as e:
    print(f'\033[0;31m[ERROR]\033[0m (Python) Failed to modify shell config: {e}', file=sys.stderr)
PY

# gcloud checks
if ! command -v gcloud >/dev/null 2>&1; then
  log_error "gcloud CLI not found. Please install Google Cloud SDK."
  exit 1
fi
ACTIVE_GCLOUD_ACCOUNT="$(gcloud config get-value account 2>/dev/null | sed -n '1p' || true)"
if [ "${ACTIVE_GCLOUD_ACCOUNT:-}" = "(unset)" ] || [ -z "${ACTIVE_GCLOUD_ACCOUNT:-}" ]; then
  log_error "No active gcloud authentication found. Please run 'gcloud auth login'"
  exit 1
fi

# Detect project/zone similarly to eopod (metadata first, then gcloud config)
log_info "Detecting project and zone..."
PROJECT_ID="$(metadata_value "project/project-id" | sed -n '1p')"
if [ -z "${PROJECT_ID:-}" ]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null | sed -n '1p' || true)"
fi
if [ "${PROJECT_ID:-}" = "(unset)" ] || [ -z "${PROJECT_ID:-}" ]; then
  log_error "Failed to detect project ID from metadata or gcloud config."
  exit 1
fi

ZONE_RAW="$(metadata_value "instance/zone" | sed -n '1p')"
ZONE=""
if [ -n "${ZONE_RAW:-}" ]; then
  ZONE="${ZONE_RAW##*/}"
fi
if [ -z "${ZONE:-}" ]; then
  ZONE="$(gcloud config get-value compute/zone 2>/dev/null | sed -n '1p' || true)"
fi
if [ "${ZONE:-}" = "(unset)" ] || [ -z "${ZONE:-}" ]; then
  log_error "Failed to detect zone from metadata or gcloud config."
  exit 1
fi

log_success "Detected project: $PROJECT_ID"
log_success "Detected zone: $ZONE"

if ! gcloud auth print-access-token --account "$ACTIVE_GCLOUD_ACCOUNT" >/dev/null 2>&1; then
  log_error "Active gcloud account cannot refresh non-interactively: ${ACTIVE_GCLOUD_ACCOUNT}"
  log_info "gsutil may still work via legacy .boto/ADC credentials, but eopod uses 'gcloud compute tpus tpu-vm ssh'."
  log_info "Refresh this gcloud account, then rerun setup:"
  log_info "  gcloud auth login \"${ACTIVE_GCLOUD_ACCOUNT}\" --no-browser"
  log_info "Verify with:"
  log_info "  gcloud compute tpus tpu-vm list --project=\"${PROJECT_ID}\" --zone=\"${ZONE}\""
  exit 1
fi

# TPU selection
TPU_NAME=""
SELF_INTERNAL_IP="$(metadata_value "instance/network-interfaces/0/ip" | sed -n '1p')"
if [ -n "${SELF_INTERNAL_IP:-}" ]; then
  log_info "Trying eopod-style TPU auto-detection using self internal IP: ${SELF_INTERNAL_IP}"
  TPU_NAME="$(
    gcloud compute tpus tpu-vm list \
      --project="$PROJECT_ID" \
      --zone="$ZONE" \
      --flatten="networkEndpoints[]" \
      --filter="networkEndpoints.ipAddress=${SELF_INTERNAL_IP}" \
      --format="value(name.basename())" 2>/dev/null | sed -n '1p' || true
  )"
fi

if [ -n "${TPU_NAME:-}" ]; then
  log_success "Auto-detected TPU from metadata/network endpoints: $TPU_NAME"
else
  log_warning "Could not resolve TPU from current VM metadata; falling back to READY TPU lookup."
  mapfile -t READY_TPUS < <(
    gcloud compute tpus tpu-vm list \
      --project="$PROJECT_ID" \
      --zone="$ZONE" \
      --filter="state:READY" \
      --format="value(name.basename())" 2>/dev/null || true
  )

  if (( ${#READY_TPUS[@]} == 0 )); then
    log_error "Could not auto-detect TPU name (no READY TPUs found in zone ${ZONE})."
    log_info "All TPUs in zone ${ZONE}:"
    gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" --format="table(name.basename(),state,health,acceleratorType)" || true
    exit 1
  elif (( ${#READY_TPUS[@]} == 1 )); then
    TPU_NAME="${READY_TPUS[0]}"
    log_success "Found one READY TPU: $TPU_NAME - using it automatically"
  else
    TPU_NAME="${READY_TPUS[0]}"
    log_warning "Multiple READY TPUs found; auto-selecting the first one: $TPU_NAME"
    gcloud compute tpus tpu-vm list \
      --project="$PROJECT_ID" \
      --zone="$ZONE" \
      --filter="state:READY" \
      --format="table(name.basename(),acceleratorType,health)" || true
  fi
fi

log_success "Selected TPU: $TPU_NAME"

# TPU type
log_info "Getting TPU accelerator type..."
TPU_TYPE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --format="value(acceleratorType)" 2>/dev/null | awk -F'/' '{print $NF}')
if [ -z "${TPU_TYPE:-}" ]; then
  log_warning "Could not determine TPU type, defaulting to v4-8"
  TPU_TYPE="v4-8"
else
  log_success "Detected TPU type: $TPU_TYPE"
fi

# ---------- Bootstrapping ----------
UV="${HOME}/.local/bin/uv"
LOCAL_VENV_PATH="$HOME/orchestrator-venv"
REMOTE_VENV_PATH="$HOME/easy-venv"
# Where each TPU host clones EasyDeL from (overridable via env). Kept separate from any
# developer checkout at ${HOME}/EasyDeL so this script never clobbers local work.
EASYDEL_SRC_DIR="${EASYDEL_SRC_DIR:-$HOME/easydel-src}"
EASYDEL_REPO_URL="${EASYDEL_REPO_URL:-https://github.com/erfanzar/EasyDeL}"
EOPOD_COMMAND_TIMEOUT="${EOPOD_COMMAND_TIMEOUT:-1800}"
EOPOD_CLONE_TIMEOUT="${EOPOD_CLONE_TIMEOUT:-300}"
EOPOD_INSTALL_TIMEOUT="${EOPOD_INSTALL_TIMEOUT:-2400}"
REMOTE_PYTHON_TIMEOUT="${REMOTE_PYTHON_TIMEOUT:-300}"
EASYDEL_VERBOSE="${EASYDEL_VERBOSE:-0}"

log_info "Installing uv locally on orchestrator..."
if ! python3 -m pip install --user -U uv --quiet; then
  if ! /usr/bin/python -m pip install --user -U uv --quiet; then
    log_error "Failed to install uv locally"
    exit 1
  fi
fi
log_success "uv installed locally"

log_info "Creating local orchestrator virtual environment at $LOCAL_VENV_PATH..."
if ! "$UV" venv "$LOCAL_VENV_PATH" --clear --python 3.13.5; then
  log_error "Failed to create local orchestrator virtual environment"
  exit 1
fi
if ! "$UV" venv "$REMOTE_VENV_PATH" --clear --python 3.13.5; then
  log_error "Failed to create local orchestrator/remote virtual environment"
  exit 1
fi

log_success "Local orchestrator virtual environment created"

log_info "Installing eopod in local orchestrator environment..."
if ! "$UV" pip install --python "$LOCAL_VENV_PATH/bin/python" -U eopod --quiet; then
  log_error "Failed to install eopod in local environment"
  exit 1
fi
LOCAL_EOPOD_PATH="$LOCAL_VENV_PATH/bin/eopod"
log_success "eopod installed in local environment"

log_info "Configuring eopod with TPU: $TPU_NAME"
if ! "$LOCAL_EOPOD_PATH" configure --project-id "$PROJECT_ID" --zone "$ZONE" --tpu-name "$TPU_NAME"; then
  log_error "Failed to configure eopod with TPU"
  exit 1
fi
log_success "eopod configured successfully"
log_warning "IMPORTANT: Press Enter during first execution to accept terms (terms may not be displayed)"
echo ""

emit_eopod_payload_output() {
  awk '
    /^Output:$/ { in_output = 1; next }
    /^Duration:/ { in_output = 0; next }
    in_output {
      if ($0 ~ /^__EASYDEL_REMOTE_EXIT_STATUS__:/) next
      if ($0 ~ /^Command completed successfully$/) next
      if ($0 == "") next
      print
    }
  '
}

emit_ray_config_output() {
  awk '
    /Using current machine as head:/ { print; next }
    /Found internal IPs:/ { print; next }
    /Auto-detecting TPU configuration/ { print; next }
    /Auto-detected TPU version:/ { print; next }
    /Auto-detected TPU slice size:/ { print; next }
    /Ray runtime started/ { print "Ray runtime started."; next }
    /Ray cluster configuration completed successfully/ { print; next }
  '
}

run_checked_output() {
  local description="$1"
  shift
  local output
  local status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  if (( status != 0 )) || grep -Eq "Command failed|ModuleNotFoundError|Traceback \\(most recent call last\\)" <<< "$output"; then
    printf '%s\n' "$output"
    log_error "Failed to ${description}"
    return 1
  fi
  printf '%s\n' "$output" | emit_ray_config_output
}

run_on_tpu() {
  local command="$1"
  local description="$2"
  local timeout="${3:-$EOPOD_COMMAND_TIMEOUT}"
  local payload
  local encoded_payload
  local output
  local status
  local remote_status

  payload="set -o pipefail
${command}
status=\$?
echo __EASYDEL_REMOTE_EXIT_STATUS__:\${status}
exit \${status}"
  encoded_payload="$(printf '%s' "$payload" | base64 -w 0)"

  set +e
  output="$("$LOCAL_EOPOD_PATH" run --no-stream --timeout "$timeout" "printf %s ${encoded_payload} | base64 -d | bash" 2>&1)"
  status=$?
  set -e

  remote_status="$(sed -n 's/.*__EASYDEL_REMOTE_EXIT_STATUS__:\([0-9][0-9]*\).*/\1/p' <<< "$output" | tail -n1)"
  if (( status != 0 )) || [ -z "${remote_status:-}" ] || (( remote_status != 0 )); then
    printf '%s\n' "$output"
    log_error "Failed to ${description}"
    return 1
  fi
  printf '%s\n' "$output" | emit_eopod_payload_output
}

run_on_tpu_stream() {
  local command="$1"
  local description="$2"
  local timeout="${3:-$EOPOD_COMMAND_TIMEOUT}"
  local payload
  local encoded_payload
  local remote_script
  local output_file
  local -a pipeline_status
  local status

  payload="set -Ee -o pipefail; trap 'status=\$?; echo __EASYDEL_REMOTE_EXIT_STATUS__:\${status}; exit \${status}' EXIT; ${command}"
  encoded_payload="$(printf '%s' "$payload" | base64 -w 0)"
  remote_script="/tmp/easydel-eopod-step-${RANDOM}-${RANDOM}.sh"
  output_file="$(mktemp -t easydel-eopod-stream.XXXXXX)"

  if ! run_on_tpu "printf %s ${encoded_payload} | base64 -d > '${remote_script}' && chmod 700 '${remote_script}'" "stage remote script for ${description}" 120; then
    rm -f "$output_file"
    return 1
  fi

  set +e
  "$LOCAL_EOPOD_PATH" run --timeout "$timeout" "bash ${remote_script}" 2>&1 \
    | tee "$output_file" \
    | awk '
      /^__EASYDEL_REMOTE_EXIT_STATUS__:/ { next }
      /^Started at:/ { next }
      /^Executing:/ { next }
      /^Executing command on worker/ { next }
      /^Using ssh batch size/ { next }
      /^SSH: Attempting to connect to worker/ { next }
      /^Command completed successfully$/ { next }
      /^Duration:/ { next }
      { print }
    '
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  status="${pipeline_status[0]}"

  run_on_tpu "rm -f '${remote_script}'" "clean remote script for ${description}" 120 || true

  if (( status != 0 )) \
    || ! grep -q '^__EASYDEL_REMOTE_EXIT_STATUS__:' "$output_file" \
    || grep -Eq '^__EASYDEL_REMOTE_EXIT_STATUS__:([1-9][0-9]*)$' "$output_file"; then
    rm -f "$output_file"
    log_error "Failed to ${description}"
    return 1
  fi
  rm -f "$output_file"
}

run_python_on_tpu() {
  local code="$1"
  local description="$2"
  local eopod_timeout="${3:-$EOPOD_COMMAND_TIMEOUT}"
  local python_timeout="${4:-$REMOTE_PYTHON_TIMEOUT}"
  local encoded_code
  encoded_code="$(printf '%s' "$code" | base64 -w 0)"
  run_on_tpu "printf %s ${encoded_code} | base64 -d | timeout ${python_timeout} ${REMOTE_VENV_PATH}/bin/python -" "$description" "$eopod_timeout"
}

run_python_on_tpu_worker0() {
  local code="$1"
  local description="$2"
  local eopod_timeout="${3:-$EOPOD_COMMAND_TIMEOUT}"
  local python_timeout="${4:-$REMOTE_PYTHON_TIMEOUT}"
  local encoded_code
  encoded_code="$(printf '%s' "$code" | base64 -w 0)"
  run_on_tpu "
    host=\$(hostname)
    worker_id=\${TPU_WORKER_ID:-\${TPU_WORKER_INDEX:-}}
    if [ -z \"\${worker_id}\" ] && [[ \"\${host}\" =~ -w-([0-9]+)$ ]]; then
      worker_id=\"\${BASH_REMATCH[1]}\"
    fi
    if [ \"\${worker_id:-}\" = \"0\" ]; then
      printf %s ${encoded_code} | base64 -d | timeout ${python_timeout} ${REMOTE_VENV_PATH}/bin/python -
    fi
  " "$description" "$eopod_timeout"
}

run_python_local() {
  local code="$1"
  local description="$2"
  local python_timeout="${3:-$REMOTE_PYTHON_TIMEOUT}"
  local output
  local status

  set +e
  output="$(printf '%s' "$code" | timeout "$python_timeout" "${REMOTE_VENV_PATH}/bin/python" - 2>&1)"
  status=$?
  set -e

  if (( status != 0 )); then
    printf '%s\n' "$output"
    log_error "Failed to ${description}"
    return 1
  fi
  printf '%s\n' "$output"
}

cleanup_ray_on_tpu() {
  local remote_venv_q
  printf -v remote_venv_q '%q' "$REMOTE_VENV_PATH"

  log_info "Stopping existing Ray runtime on TPU hosts (if any)..."
  run_on_tpu_stream "
    remote_venv=${remote_venv_q}
    host=\$(hostname)
    sudo_cmd=\"\"
    if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
      sudo_cmd=\"sudo -n\"
    fi
    echo \"[\${host}] stopping Ray through known CLIs...\"
    if [ -n \"\${sudo_cmd}\" ]; then
      echo \"[\${host}] passwordless sudo available; stale Ray processes from other users can be cleaned\"
    fi

    ray_bins=\"\${remote_venv}/bin/ray /app/.venv/bin/ray\"
    if found_ray=\$(command -v ray 2>/dev/null); then
      ray_bins=\"\${ray_bins} \${found_ray}\"
    fi
    for ray_bin in \${ray_bins}; do
      [ -x \"\${ray_bin}\" ] || continue
      timeout 60 \"\${ray_bin}\" stop --force >/tmp/easydel-ray-stop.log 2>&1 || true
    done

    list_ray_pids() {
      ps -eo pid=,args= | awk '
        /[r]ay start/ ||
        /[r]ay::/ ||
        /[r]ay.util.client.server/ ||
        /[s]ite-packages\\/ray\\// ||
        /[r]ay\\/core\\/src\\/ray\\// ||
        /[j]axtuner_tmp\\/ray/ ||
        /[e]asydel_tmp\\/ray/ ||
        /[t]mp\\/ray\\/session_/ { print \$1 }
      ' | sort -u
    }

    pids=\$(list_ray_pids | tr '\n' ' ')
    if [ -n \"\${pids}\" ]; then
      echo \"[\${host}] terminating stale Ray pids: \${pids}\"
      \${sudo_cmd} kill \${pids} 2>/dev/null || true
      sleep 5
    fi

    pids=\$(list_ray_pids | tr '\n' ' ')
    if [ -n \"\${pids}\" ]; then
      echo \"[\${host}] force-killing stale Ray pids: \${pids}\"
      \${sudo_cmd} kill -9 \${pids} 2>/dev/null || true
      sleep 2
    fi

    if [ -n \"\${sudo_cmd}\" ]; then
      \${sudo_cmd} fuser -k 6379/tcp 8265/tcp 10001/tcp >/dev/null 2>&1 || true
      \${sudo_cmd} rm -rf /tmp/ray /tmp/jaxtuner_tmp/ray /tmp/easydel_tmp/ray
      \${sudo_cmd} rm -f /tmp/libtpu_lockfile
    else
      fuser -k 6379/tcp 8265/tcp 10001/tcp >/dev/null 2>&1 || true
      rm -rf /tmp/ray /tmp/jaxtuner_tmp/ray /tmp/easydel_tmp/ray >/tmp/easydel-ray-rm.log 2>&1 || true
      rm -f /tmp/libtpu_lockfile >/dev/null 2>&1 || true
    fi

    for attempt in 1 2 3 4 5 6 7 8 9 10; do
      pids=\$(list_ray_pids | tr '\n' ' ')
      [ -z \"\${pids}\" ] && break
      \${sudo_cmd} kill -9 \${pids} 2>/dev/null || true
      sleep 1
    done

    pids=\$(list_ray_pids | tr '\n' ' ')
    if [ -n \"\${pids}\" ]; then
      echo \"[\${host}] Ray processes remain after cleanup:\"
      ps -eo user=,pid=,stat=,etime=,args= | awk '
        /[r]ay start/ ||
        /[r]ay::/ ||
        /[r]ay.util.client.server/ ||
        /[s]ite-packages\\/ray\\// ||
        /[r]ay\\/core\\/src\\/ray\\// ||
        /[j]axtuner_tmp\\/ray/ ||
        /[e]asydel_tmp\\/ray/ ||
        /[t]mp\\/ray\\/session_/ { print }
      '
      if [ -z \"\${sudo_cmd}\" ]; then
        echo \"[\${host}] cleanup needs passwordless sudo or manual termination by the owning user/root\"
      fi
      exit 1
    fi

    if command -v ss >/dev/null 2>&1; then
      ray_ports=\$(\${sudo_cmd} ss -ltnp 2>/dev/null | grep -E ':(6379|8265|10001) ' || true)
      if [ -n \"\${ray_ports}\" ]; then
        echo \"[\${host}] Ray ports are still in use:\"
        printf '%s\n' \"\${ray_ports}\"
        exit 1
      fi
    fi

    echo \"[\${host}] Ray cleanup complete\"
  " "clean existing Ray runtime on TPU hosts" 300
}

configure_ray_on_tpu() {
  local internal_ips_output
  local internal_ips_str
  local head_ip
  local expected_nodes
  local tpu_version
  local tpu_slice_size
  local ray_bin="${REMOTE_VENV_PATH}/bin/ray"
  local ray_tmp_dir="/tmp/easydel_tmp"
  local head_resources
  local worker_resources
  local head_ip_q
  local ray_bin_q
  local ray_tmp_dir_q
  local head_resources_q
  local worker_resources_q

  log_info "Fetching TPU internal IPs for Ray configuration..."
  if ! internal_ips_output="$("$LOCAL_EOPOD_PATH" get-internal-ips 2>&1)"; then
    printf '%s\n' "$internal_ips_output"
    log_error "Failed to fetch TPU internal IPs"
    return 1
  fi
  internal_ips_str="$(tr -d '[:space:]' <<< "$internal_ips_output")"
  if [ -z "${internal_ips_str:-}" ]; then
    log_error "No TPU internal IPs returned by eopod"
    return 1
  fi

  head_ip="${internal_ips_str%%,*}"
  expected_nodes="$(awk -F, '{ print NF }' <<< "$internal_ips_str")"
  RAY_HEAD_IP="$head_ip"
  tpu_version="${TPU_TYPE%%-*}"
  tpu_slice_size="${TPU_TYPE##*-}"
  head_resources="{\"TPU\":4,\"TPU-${tpu_version}\":4,\"TPU-${tpu_version}-${tpu_slice_size}-head\":1,\"TPU-${tpu_version}-${tpu_slice_size}-global-head\":1,\"accelerator_type:TPU-${tpu_version^^}\":1,\"head-node\":1,\"ray-cluster-head\":1}"
  worker_resources="{\"TPU\":4,\"TPU-${tpu_version}\":4,\"accelerator_type:TPU-${tpu_version^^}\":1}"

  printf -v head_ip_q '%q' "$head_ip"
  printf -v ray_bin_q '%q' "$ray_bin"
  printf -v ray_tmp_dir_q '%q' "$ray_tmp_dir"
  printf -v head_resources_q '%q' "$head_resources"
  printf -v worker_resources_q '%q' "$worker_resources"

  log_info "Starting Ray head at ${head_ip} and workers with ${ray_bin}..."
  run_on_tpu_stream "
    head_ip=${head_ip_q}
    ray_bin=${ray_bin_q}
    ray_tmp_dir=${ray_tmp_dir_q}
    head_resources=${head_resources_q}
    worker_resources=${worker_resources_q}
    host=\$(hostname)
    local_ip=\$(hostname -I | awk '{print \$1}')

    export TMPDIR=\"\${ray_tmp_dir}\"
    export RAY_TMPDIR=\"\${ray_tmp_dir}/ray\"
    mkdir -p \"\${TMPDIR}\" \"\${RAY_TMPDIR}\"

    \"\${ray_bin}\" stop --force >/tmp/easydel-ray-stop.log 2>&1 || true

    start_ray() {
      local log_file=\$1
      shift
      if ! \"\${ray_bin}\" start \"\$@\" >\"\${log_file}\" 2>&1; then
        echo \"[\${host}] Ray start failed; log tail follows\"
        tail -n 120 \"\${log_file}\" || true
        exit 1
      fi
    }

    if [ \"\${local_ip}\" = \"\${head_ip}\" ]; then
      echo \"[\${host}] starting Ray head at \${head_ip}\"
      start_ray /tmp/easydel-ray-start.log \
        --head \
        --port=6379 \
        --resources=\"\${head_resources}\" \
        --node-ip-address=\"\${head_ip}\" \
        --dashboard-host=0.0.0.0 \
        --disable-usage-stats
      echo \"[\${host}] Ray head started\"
    else
      echo \"[\${host}] waiting for Ray head at \${head_ip}:6379\"
      for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
        if timeout 1 bash -c \"</dev/tcp/\${head_ip}/6379\" >/dev/null 2>&1; then
          break
        fi
        sleep 2
      done
      timeout 1 bash -c \"</dev/tcp/\${head_ip}/6379\" >/dev/null 2>&1 || {
        echo \"[\${host}] Ray head did not open port 6379\"
        exit 1
      }
      echo \"[\${host}] starting Ray worker \${local_ip} -> \${head_ip}:6379\"
      start_ray /tmp/easydel-ray-start.log \
        --address=\"\${head_ip}:6379\" \
        --resources=\"\${worker_resources}\" \
        --node-ip-address=\"\${local_ip}\" \
        --disable-usage-stats
      echo \"[\${host}] Ray worker started\"
    fi
  " "start Ray runtime on TPU hosts" 300

  log_info "Waiting for Ray cluster at ${head_ip}:6379 to register ${expected_nodes} TPU hosts..."
  if ! "${REMOTE_VENV_PATH}/bin/python" - "$head_ip:6379" "$expected_nodes" <<'PY'; then
import logging
import sys
import time
import warnings

import ray

address = sys.argv[1]
expected_nodes = int(sys.argv[2])
expected_tpus = expected_nodes * 4
deadline = time.monotonic() + 300
last_seen = None
last_error = None

while time.monotonic() < deadline:
    try:
        if not ray.is_initialized():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                ray.init(address=address, ignore_reinit_error=True, logging_level=logging.ERROR)
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        total_tpus = sum(node.get("Resources", {}).get("TPU", 0) for node in alive_nodes)
        current = (len(alive_nodes), int(total_tpus))
        if current != last_seen:
            print(f"Ray readiness: {current[0]}/{expected_nodes} nodes, {current[1]}/{expected_tpus} TPU resources", flush=True)
            last_seen = current
        if len(alive_nodes) >= expected_nodes and total_tpus >= expected_tpus:
            break
    except Exception as exc:
        current_error = repr(exc)
        if current_error != last_error:
            print(f"Ray readiness: waiting for {address}: {current_error}", flush=True)
            last_error = current_error
        ray.shutdown()
    time.sleep(5)
else:
    raise SystemExit(f"Ray cluster did not become ready: last={last_seen}, expected=({expected_nodes}, {expected_tpus})")
PY
    log_error "Ray cluster did not become reachable at ${head_ip}:6379"
    return 1
  fi

  if [ "$EASYDEL_VERBOSE" = "1" ]; then
    timeout 60 "${ray_bin}" status --address="${head_ip}:6379" || true
  fi
}

# Fetch the EasyDeL monorepo on EVERY TPU host, from scratch, directly from git.
# This is intentionally self-contained: the script may be run on its own (e.g. downloaded
# via a link) with NO local checkout, and each TPU worker is a separate host that does not
# share a filesystem with the orchestrator. So every host clones the repo itself rather than
# depending on any local files. The clone goes to a dedicated ${EASYDEL_SRC_DIR} so it never
# touches a developer's working checkout that may live at ${HOME}/EasyDeL.
clone_workspace_on_tpu() {
  # Clone EasyDeL on each TPU host with eopod. The clone is shallow (~13MB pack,
  # ~1s), so doing it per-host is cheap. git's progress bar is redirected to a log
  # file, while concise per-worker status lines keep this step visibly alive.
  # One retry covers a transient network hiccup; the final `test -f` is the
  # success gate the hook reports on.
  local branch_q
  local repo_url_q
  local src_dir_q
  printf -v branch_q '%q' "$EASYDEL_BRANCH"
  printf -v repo_url_q '%q' "$EASYDEL_REPO_URL"
  printf -v src_dir_q '%q' "$EASYDEL_SRC_DIR"

  log_info "Cloning EasyDeL ${EASYDEL_BRANCH} from ${EASYDEL_REPO_URL} on all TPU hosts -> ${EASYDEL_SRC_DIR} (via eopod)..."
  run_on_tpu_stream "
    branch=${branch_q}
    repo_url=${repo_url_q}
    src_dir=${src_dir_q}
    clone_log=/tmp/easydel-clone.log
    host=\$(hostname)
    echo \"[\${host}] cloning \${repo_url}@\${branch} -> \${src_dir}\"
    rm -rf \"\${src_dir}\"
    if git clone --depth 1 --branch \"\${branch}\" \"\${repo_url}\" \"\${src_dir}\" >\"\${clone_log}\" 2>&1; then
      :
    else
      echo \"[\${host}] first clone attempt failed; retrying once\"
      tail -n 40 \"\${clone_log}\" || true
      rm -rf \"\${src_dir}\"
      sleep 3
      if git clone --depth 1 --branch \"\${branch}\" \"\${repo_url}\" \"\${src_dir}\" >>\"\${clone_log}\" 2>&1; then
        :
      else
        status=\$?
        echo \"[\${host}] clone failed; log tail follows\"
        tail -n 120 \"\${clone_log}\" || true
        exit \${status}
      fi
    fi
    test -f \"\${src_dir}/libs/easydel/pyproject.toml\" || {
      echo \"[\${host}] clone completed but libs/easydel/pyproject.toml is missing\"
      exit 1
    }
    rev=\$(git -C \"\${src_dir}\" rev-parse --short HEAD)
    echo \"[\${host}] clone ready @ \${rev}\"
  " "clone EasyDeL ${EASYDEL_BRANCH} on TPU hosts" "$EOPOD_CLONE_TIMEOUT"
}

install_workspace_on_tpu() {
  local remote_python="${REMOTE_VENV_PATH}/bin/python"
  local remote_python_q
  local src_dir_q
  printf -v remote_python_q '%q' "$remote_python"
  printf -v src_dir_q '%q' "$EASYDEL_SRC_DIR"

  log_info "Installing EasyDeL editable workspace packages from ${EASYDEL_SRC_DIR}/libs on TPU hosts..."
  run_on_tpu_stream "
    src_dir=${src_dir_q}
    remote_python=${remote_python_q}
    host=\$(hostname)
    test -f \"\${src_dir}/libs/eformer/pyproject.toml\" || {
      echo \"[\${host}] missing \${src_dir}/libs/eformer/pyproject.toml\"
      exit 1
    }
    echo \"[\${host}] installing EasyDeL editable workspace packages with TPU, profile, and dev extras...\"
    ~/.local/bin/uv pip install \
      --python \"\${remote_python}\" \
      --editable \"\${src_dir}/libs/eformer[dev]\" \
      --editable \"\${src_dir}/libs/spectrax[tpu,dev]\" \
      --editable \"\${src_dir}/libs/ejkernel[tpu,profile,dev]\" \
      --editable \"\${src_dir}/libs/easydel[tpu,torch,lm_eval,profile]\" \
      --editable \"\${src_dir}/libs/eray[dev]\" \
      \"import-linter>=2.0\" \
      ruff \
      pytest \
      pre-commit \
      basedpyright \
      --quiet
    echo \"[\${host}] workspace install complete\"
  " "install EasyDeL workspace packages on TPU hosts" "$EOPOD_INSTALL_TIMEOUT"
}

log_info "Installing uv on TPU hosts..."
if ! run_on_tpu "pip install uv --quiet -U" "install uv on TPU hosts" "$EOPOD_INSTALL_TIMEOUT"; then
  log_error "Failed to install uv on TPU hosts"
  exit 1
fi
log_success "uv installed on TPU hosts"

log_info "Creating virtual environment on TPU hosts at $REMOTE_VENV_PATH..."
if ! run_on_tpu "~/.local/bin/uv venv $REMOTE_VENV_PATH --clear --python 3.13.5" "create virtual environment on TPU hosts"; then
  log_error "Failed to create virtual environment on TPU hosts"
  exit 1
fi
log_success "Virtual environment created on TPU hosts"

log_info "Installing eopod on TPU hosts..."
if ! run_on_tpu "~/.local/bin/uv pip install --python ${REMOTE_VENV_PATH}/bin/python -U eopod --quiet" "install eopod on TPU hosts" "$EOPOD_INSTALL_TIMEOUT"; then
  log_error "Failed to install eopod on TPU hosts"
  exit 1
fi
log_success "eopod installed on TPU hosts"

# Helper to install packages remotely into the TPU venv
install_package_on_tpu() {
  local spec="$1"
  local quoted_spec
  quoted_spec="'${spec}'"
  log_info "Installing ${spec} on TPU hosts..."
  if ! run_on_tpu "~/.local/bin/uv pip install --python ${REMOTE_VENV_PATH}/bin/python ${quoted_spec} --quiet" "install ${spec} on TPU hosts" "$EOPOD_INSTALL_TIMEOUT"; then
    log_error "Failed to install ${spec} on TPU hosts"
    return 1
  fi
  log_success "Successfully installed ${spec} on TPU hosts"
}

echo ""
log_info "Starting package installations on TPU hosts..."

log_info "Uninstalling existing easydel on TPU hosts (if any)..."
run_on_tpu "~/.local/bin/uv pip uninstall --python ${REMOTE_VENV_PATH}/bin/python easydel easydel-foundation eformer ejkernel spectrax-lib || true" "uninstall existing EasyDeL packages on TPU hosts" || true

log_info "Installing EasyDeL workspace (${EASYDEL_REPO_URL}@${EASYDEL_BRANCH}) on TPU hosts"
if ! clone_workspace_on_tpu; then
  log_error "Failed to clone EasyDeL on TPU hosts"
  exit 1
fi
install_workspace_on_tpu
run_python_on_tpu_worker0 "
from importlib import metadata
from importlib import util

packages = [
    ('easydel', 'easydel'),
    ('eformer', 'eformer'),
    ('ejkernel', 'ejkernel'),
    ('spectrax-lib', 'spectrax'),
    ('eray', 'eray'),
]
versions = []
for dist_name, module_name in packages:
    if util.find_spec(module_name) is None:
        raise ModuleNotFoundError(module_name)
    versions.append(f'{module_name} {metadata.version(dist_name)}')

print('; '.join(versions))
required_dists = [
    'xprof',
    'tb-nightly',
    'xprof-nightly',
    'ruff',
    'pytest',
    'pre-commit',
    'basedpyright',
    'import-linter',
]
missing = []
for dist_name in required_dists:
    try:
        metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        missing.append(dist_name)
if missing:
    raise RuntimeError(f'missing profile/dev packages: {missing}')
print('profile/dev tools installed')
" "verify EasyDeL foundation imports on TPU hosts and print versions from worker 0"
install_package_on_tpu "ray[default]==2.54.0"
# Configure Ray (use the actual eopod binary we installed locally, not uv run)
log_info "Configuring Ray..."
export RAY_EXECUTABLE_PATH="${REMOTE_VENV_PATH}/bin/ray"
if ! cleanup_ray_on_tpu; then
  log_error "Failed to clean existing Ray runtime"
  exit 1
fi
if ! configure_ray_on_tpu; then
  log_error "Failed to configure Ray"
  exit 1
fi
log_success "Ray configured successfully"

# ---------- Summary ----------
echo ""
log_success "🎉 TPU setup completed successfully!"
log_info "Project: $PROJECT_ID"
log_info "TPU Name: $TPU_NAME"
log_info "TPU Type: $TPU_TYPE"
log_info "Zone: $ZONE"
log_info "EasyDeL source: ${EASYDEL_REPO_URL}@${EASYDEL_BRANCH} (cloned to ${EASYDEL_SRC_DIR}/libs on each host)"
echo ""
log_info "Final TPU status:"
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" --filter="name:$TPU_NAME" --format="table(name,state,health,acceleratorType)" || true


log_info "Running runtime health check from Ray head..."
run_python_local "
from eformer.executor.ray import TpuAcceleratorConfig, execute
import logging
import ray
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    ray.init(address='$RAY_HEAD_IP:6379', ignore_reinit_error=True, logging_level=logging.ERROR, log_to_driver=False)


@execute(TpuAcceleratorConfig('$TPU_TYPE'))
@ray.remote
def health_check():
    import contextlib
    import io
    import os
    import re
    import socket
    from importlib import metadata

    os.environ.setdefault('ENABLE_DISTRIBUTED_INIT', '0')
    import jax

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import easydel as ed
        import eformer
        import ejkernel
        import spectrax

    host = socket.gethostname()
    worker_id = os.environ.get('TPU_WORKER_ID') or os.environ.get('TPU_WORKER_INDEX') or ''
    if not worker_id:
        match = re.search(r'-w-(\d+)$', host)
        if match:
            worker_id = match.group(1)
    is_worker0 = worker_id == '0'

    def version_text(label, dist_name, module):
        dist_version = metadata.version(dist_name)
        module_version = getattr(module, '__version__', dist_version)
        if module_version == dist_version:
            return f'{label} version: {dist_version}'
        return f'{label} version: {dist_version} (__version__ {module_version})'

    version_line = (
        ' | '.join(
            [
                version_text('EasyDeL', 'easydel', ed),
                version_text('eformer', 'eformer', eformer),
                version_text('ejkernel', 'ejkernel', ejkernel),
                version_text('spectrax', 'spectrax-lib', spectrax),
                f'JAX version: {jax.__version__}',
            ]
        )
    ) if is_worker0 else None
    return {
        'host': host,
        'worker_id': worker_id,
        'is_worker0': is_worker0,
        'version_line': version_line,
        'jax_devices': [dev.coords for dev in jax.local_devices()],
        'device_count': jax.device_count(),
        'local_device_count': jax.local_device_count(),
    }


if __name__ == '__main__':
    status = health_check()
    if not hasattr(status, 'result'):
        raise RuntimeError(status)

    def flatten_reports(value):
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            reports = []
            for item in value:
                reports.extend(flatten_reports(item))
            return reports
        return []

    reports = flatten_reports(status.result)
    if not reports:
        raise RuntimeError(status.result)

    worker0 = next((report for report in reports if report.get('is_worker0')), reports[0])
    total_devices = max(int(report.get('device_count', 0)) for report in reports)
    local_counts = sorted({int(report.get('local_device_count', 0)) for report in reports})
    versions = worker0.get('version_line')
    if versions:
        print(f'Worker 0 versions: {versions}')
    print(f'Workers healthy: {len(reports)}')
    print(f'Cluster JAX devices: {total_devices}')
    print(f'Per-worker local device counts: {local_counts}')
    print(f\"Worker 0 JAX devices: {worker0.get('jax_devices')}\")
" "run runtime health check from Ray head"

log_success "🎉 Runtime health check completed!"
echo ""

# Add eopod alias to shell configuration
log_info "Adding eopod alias to shell configuration..."
python - << 'PY'
import os, sys
BLUE = '\033[0;34m'; GREEN = '\033[0;32m'; NC = '\033[0m'
def info(m): print(f'{BLUE}[INFO]{NC} (Python) {m}', file=sys.stderr)
def ok(m):   print(f'{GREEN}[SUCCESS]{NC} (Python) {m}', file=sys.stderr)

alias_line = f'alias eopod="{os.path.expanduser("~/orchestrator-venv/bin/eopod")}"'
home = os.path.expanduser('~')
cands = [os.path.join(home, '.zshrc'), os.path.join(home, '.bashrc')]
target = next((c for c in cands if os.path.exists(c)), os.path.join(home, '.bashrc'))
info(f'Adding eopod alias to: {target}')
try:
    content = ''
    if os.path.exists(target):
        with open(target) as f: content = f.read()
    if 'alias eopod=' in content:
        info('eopod alias already exists. No changes needed.')
    else:
        with open(target, 'a') as f:
            f.write('\n# Added by TPU setup script for easy eopod access\n' + alias_line + '\n')
        ok(f'Successfully added eopod alias. Run "source {target}" or restart your terminal to use it.')
except Exception as e:
    print(f'\033[0;31m[ERROR]\033[0m (Python) Failed to add alias: {e}', file=sys.stderr)
PY

log_info "Local Orchestrator Environment: $LOCAL_VENV_PATH"
log_info "TPU Hosts Environment: $REMOTE_VENV_PATH"

echo ""

log_info "Next time:"
log_info "  Use eopod directly: eopod run \"${REMOTE_VENV_PATH}/bin/python your_script.py\""
log_info "  Local orchestrator: source ${LOCAL_VENV_PATH}/bin/activate"
log_info "  Run on TPU: ${LOCAL_VENV_PATH}/bin/eopod run \"${REMOTE_VENV_PATH}/bin/python your_script.py\""
