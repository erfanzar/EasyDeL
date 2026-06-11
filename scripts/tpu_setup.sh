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
  --branch <ref>   Accepted for compatibility, but package installation uses
                   this checkout's local libs/ workspace.
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

if [ -n "${EASYDEL_BRANCH:-}" ]; then
  log_warning "--branch ${EASYDEL_BRANCH} was provided, but TPU setup installs from local libs/: ${REPO_ROOT}/libs"
fi

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
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 >/dev/null; then
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
  local payload
  local quoted_payload
  local output
  local status
  local remote_status

  payload="set -o pipefail; ${command}; status=\$?; echo __EASYDEL_REMOTE_EXIT_STATUS__:\${status}; exit \${status}"
  printf -v quoted_payload '%q' "$payload"

  set +e
  output="$("$LOCAL_EOPOD_PATH" run --no-stream "bash -lc ${quoted_payload}" 2>&1)"
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

run_python_on_tpu() {
  local code="$1"
  local description="$2"
  local encoded_code
  encoded_code="$(printf '%s' "$code" | base64 -w 0)"
  run_on_tpu "printf %s ${encoded_code} | base64 -d | ${REMOTE_VENV_PATH}/bin/python -" "$description"
}

install_workspace_on_tpu() {
  local remote_python="${REMOTE_VENV_PATH}/bin/python"

  run_on_tpu "test -f '${REPO_ROOT}/pyproject.toml' && test -f '${REPO_ROOT}/libs/easydel/pyproject.toml' && test -f '${REPO_ROOT}/libs/eformer/pyproject.toml' && test -f '${REPO_ROOT}/libs/ejkernel/pyproject.toml' && test -f '${REPO_ROOT}/libs/spectrax/pyproject.toml' && ~/.local/bin/uv pip install --python '${remote_python}' --editable '${REPO_ROOT}/libs/eformer' --editable '${REPO_ROOT}/libs/spectrax[tpu]' --editable '${REPO_ROOT}/libs/ejkernel[tpu]' --editable '${REPO_ROOT}/libs/easydel[tpu,torch,lm_eval]' --quiet" "install local EasyDeL workspace packages from libs/ on TPU hosts"
}

log_info "Installing uv on TPU hosts..."
if ! run_on_tpu "pip install uv --quiet -U" "install uv on TPU hosts"; then
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
if ! run_on_tpu "~/.local/bin/uv pip install --python ${REMOTE_VENV_PATH}/bin/python -U eopod --quiet" "install eopod on TPU hosts"; then
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
  if ! run_on_tpu "~/.local/bin/uv pip install --python ${REMOTE_VENV_PATH}/bin/python ${quoted_spec} --quiet" "install ${spec} on TPU hosts"; then
    log_error "Failed to install ${spec} on TPU hosts"
    return 1
  fi
  log_success "Successfully installed ${spec} on TPU hosts"
}

echo ""
log_info "Starting package installations on TPU hosts..."

log_info "Uninstalling existing easydel on TPU hosts (if any)..."
run_on_tpu "~/.local/bin/uv pip uninstall --python ${REMOTE_VENV_PATH}/bin/python easydel easydel-foundation eformer ejkernel spectrax-lib || true" "uninstall existing EasyDeL packages on TPU hosts" || true

log_info "Installing EasyDeL workspace from local libs/: ${REPO_ROOT}/libs"
install_workspace_on_tpu
run_python_on_tpu "
from importlib import metadata

import easydel as ed
import eformer
import ejkernel
import spectrax
from eformer.executor.ray import TpuAcceleratorConfig, execute


def version_text(label, dist_name, module):
    dist_version = metadata.version(dist_name)
    module_version = getattr(module, '__version__', dist_version)
    if module_version == dist_version:
        return f'{label} {dist_version}'
    return f'{label} {dist_version} (__version__ {module_version})'


print(
    '; '.join(
        [
            version_text('easydel', 'easydel', ed),
            version_text('eformer', 'eformer', eformer),
            version_text('ejkernel', 'ejkernel', ejkernel),
            version_text('spectrax', 'spectrax-lib', spectrax),
        ]
    )
)
" "verify EasyDeL foundation imports on TPU hosts"
install_package_on_tpu "ray[default]==2.54.0"
# Configure Ray (use the actual eopod binary we installed locally, not uv run)
log_info "Configuring Ray..."
export RAY_EXECUTABLE_PATH="${REMOTE_VENV_PATH}/bin/ray"
if ! run_checked_output "configure Ray" "$LOCAL_EOPOD_PATH" auto-config-ray --self-job --python-path "${REMOTE_VENV_PATH}/bin/python"; then
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
log_info "EasyDeL source: ${REPO_ROOT}/libs"
echo ""
log_info "Final TPU status:"
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE" --filter="name:$TPU_NAME" --format="table(name,state,health,acceleratorType)" || true


run_python_on_tpu "
from eformer.executor.ray import TpuAcceleratorConfig, execute
import ray


@execute(TpuAcceleratorConfig('$TPU_TYPE'))
@ray.remote
def health_check():
    from importlib import metadata

    import easydel as ed
    import eformer
    import ejkernel
    import jax
    import spectrax

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
    )
    return [
        version_line,
        f'JAX devices: {[dev.coords for dev in jax.local_devices()]}',
        f'Device count: {jax.device_count()}',
        f'Local device count: {jax.local_device_count()}',
    ]


if __name__ == '__main__':
    status = health_check()
    if not hasattr(status, 'result'):
        raise RuntimeError(status)
    lines = status.result
    if len(lines) == 1 and isinstance(lines[0], list):
        lines = lines[0]
    for line in lines:
        print(line)
" "run runtime health check on TPU hosts"

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
