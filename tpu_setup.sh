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

# Detect zone (GCE)
log_info "Detecting current zone..."
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | cut -d/ -f4 || true)
: "${ZONE:=}" || true
log_info "Current zone: ${ZONE:-unknown}"

# gcloud checks
if ! command -v gcloud >/dev/null 2>&1; then
  log_error "gcloud CLI not found. Please install Google Cloud SDK."
  exit 1
fi
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 >/dev/null; then
  log_error "No active gcloud authentication found. Please run 'gcloud auth login'"
  exit 1
fi

# TPU selection
log_info "Searching for available TPUs in zone ${ZONE}..."
mapfile -t READY_TPUS < <(gcloud compute tpus tpu-vm list --zone="$ZONE" --filter="state:READY" --format="value(name)" 2>/dev/null || true)

if (( ${#READY_TPUS[@]} == 0 )); then
  log_warning "No READY TPUs found in zone $ZONE"
  echo ""
  log_info "All TPUs in zone $ZONE:"
  gcloud compute tpus tpu-vm list --zone="$ZONE" --format="table(name,state,health,acceleratorType)" || {
    log_error "Failed to list TPUs. Check your permissions."
    exit 1
  }
  echo ""
  read -p "Enter your TPU name: " TPU_NAME < /dev/tty
  if [ -z "${TPU_NAME:-}" ]; then
    log_error "TPU name cannot be empty"
    exit 1
  fi
elif (( ${#READY_TPUS[@]} == 1 )); then
  TPU_NAME="${READY_TPUS[0]}"
  log_success "Found one READY TPU: $TPU_NAME - using it automatically"
else
  log_info "Multiple READY TPUs found:"
  echo ""
  gcloud compute tpus tpu-vm list --zone="$ZONE" --filter="state:READY" --format="table(name,acceleratorType,health)"
  echo ""
  echo "Available READY TPUs:"
  for i in "${!READY_TPUS[@]}"; do
    echo "$((i + 1)). ${READY_TPUS[i]}"
  done
  echo ""
  while true; do
    read -p "Enter TPU name or number (1-${#READY_TPUS[@]}): " input < /dev/tty
    if [[ "$input" =~ ^[0-9]+$ ]] && (( input >= 1 && input <= ${#READY_TPUS[@]} )); then
      TPU_NAME="${READY_TPUS[$((input - 1))]}"
      break
    elif [[ " ${READY_TPUS[*]} " == *" ${input} "* ]]; then
      TPU_NAME="$input"
      break
    else
      log_error "Invalid selection. Please enter a number (1-${#READY_TPUS[@]}) or a valid TPU name."
    fi
  done
fi

log_success "Selected TPU: $TPU_NAME"

# TPU type
log_info "Getting TPU accelerator type..."
TPU_TYPE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format="value(acceleratorType)" 2>/dev/null | awk -F'/' '{print $NF}')
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
if ! "$UV" venv "$LOCAL_VENV_PATH" --clear --python 3.11.6; then
  log_error "Failed to create local orchestrator virtual environment"
  exit 1
fi
if ! "$UV" venv "$REMOTE_VENV_PATH" --clear --python 3.11.6; then
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
if ! "$LOCAL_EOPOD_PATH" configure --tpu-name "$TPU_NAME"; then
  log_error "Failed to configure eopod with TPU"
  exit 1
fi
log_success "eopod configured successfully"
log_warning "IMPORTANT: Press Enter during first execution to accept terms (terms may not be displayed)"
echo ""

log_info "Installing uv on TPU hosts..."
if ! "$LOCAL_EOPOD_PATH" run "pip install uv --quiet -U"; then
  log_error "Failed to install uv on TPU hosts"
  exit 1
fi
log_success "uv installed on TPU hosts"

log_info "Creating virtual environment on TPU hosts at $REMOTE_VENV_PATH..."
if ! "$LOCAL_EOPOD_PATH" run "~/.local/bin/uv venv $REMOTE_VENV_PATH --clear --python 3.11.6"; then
  log_error "Failed to create virtual environment on TPU hosts"
  exit 1
fi
log_success "Virtual environment created on TPU hosts"

log_info "Installing eopod on TPU hosts..."
if ! "$LOCAL_EOPOD_PATH" run "~/.local/bin/uv pip install --python ${REMOTE_VENV_PATH}/bin/python -U eopod --quiet"; then
  log_error "Failed to install eopod on TPU hosts"
  exit 1
fi
log_success "eopod installed on TPU hosts"

# Helper to install packages remotely into the TPU venv
install_package_on_tpu() {
  local spec="$1"
  log_info "Installing ${spec} on TPU hosts..."
  if ! "$LOCAL_EOPOD_PATH" run "~/.local/bin/uv pip install --python ${REMOTE_VENV_PATH}/bin/python ${spec} --quiet"; then
    log_error "Failed to install ${spec} on TPU hosts"
    return 1
  fi
  log_success "Successfully installed ${spec} on TPU hosts"
}

echo ""
log_info "Starting package installations on TPU hosts..."

log_info "Uninstalling existing easydel on TPU hosts (if any)..."
"$LOCAL_EOPOD_PATH" run "~/.local/bin/uv pip uninstall --python ${REMOTE_VENV_PATH}/bin/python easydel" || true

# Use PEP 508 direct URL so extras are preserved:
# 'easydel[extras] @ git+https://...'
install_package_on_tpu "'easydel[tpu,torch,lm_eval] @ git+https://github.com/erfanzar/easydel.git'"
install_package_on_tpu "ray[default]==2.34.0"
# Configure Ray (use the actual eopod binary we installed locally, not uv run)
log_info "Configuring Ray..."
export RAY_EXECUTABLE_PATH="${REMOTE_VENV_PATH}/bin/ray"
if ! "$LOCAL_EOPOD_PATH" auto-config-ray --self-job --python-path "${REMOTE_VENV_PATH}/bin/python"; then
  log_error "Failed to configure Ray"
  exit 1
fi
log_success "Ray configured successfully"

# ---------- Summary ----------
echo ""
log_success "🎉 TPU setup completed successfully!"
log_info "TPU Name: $TPU_NAME"
log_info "TPU Type: $TPU_TYPE"
log_info "Zone: $ZONE"
echo ""
log_info "Final TPU status:"
gcloud compute tpus tpu-vm list --zone="$ZONE" --filter="name:$TPU_NAME" --format="table(name,state,health,acceleratorType)" || true


"${REMOTE_VENV_PATH}/bin/python" -c "
from eformer.executor.ray import TpuAcceleratorConfig, execute
import ray


@execute(TpuAcceleratorConfig('$TPU_TYPE'))
@ray.remote
def health_check():
    import easydel as ed
    import eformer
    import jax

    print(f'EasyDel version: {ed.__version__} | eformer version {eformer.__version__} | JAX version {jax.__version__}')
    print(f'JAX devices: {[dev.coords for dev in jax.local_devices()]}')
    print(f'Device count: {jax.device_count()}')
    print(f'Local device count: {jax.local_device_count()}')


if __name__ == '__main__':
    health_check()
"

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
