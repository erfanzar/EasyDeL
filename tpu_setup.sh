#!/bin/bash

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}
case ":$PATH:" in
  *":$HOME/.local/bin:"*)
    :
    ;;
  *)
    log_info "Adding $HOME/.local/bin to PATH for the current session."
    export PATH="$HOME/.local/bin:$PATH"
    ;;
esac

log_info "Checking shell configuration for persistent PATH..."
python -c "
import os
import sys
BLUE = '\033[0;34m'
GREEN = '\033[0;32m'
NC = '\033[0m'

def log_py_info(msg):
    print(f'{BLUE}[INFO]{NC} (Python) {msg}', file=sys.stderr)

def log_py_success(msg):
    print(f'{GREEN}[SUCCESS]{NC} (Python) {msg}', file=sys.stderr)

line_to_add = 'export PATH=\"\$HOME/.local/bin:\$PATH\"'
home_dir = os.path.expanduser('~')
shell_config_files = [os.path.join(home_dir, '.zshrc'), os.path.join(home_dir, '.bashrc')]

config_file_to_modify = None
for f in shell_config_files:
    if os.path.exists(f):
        config_file_to_modify = f
        break

if not config_file_to_modify:
    config_file_to_modify = os.path.join(home_dir, '.bashrc')
    log_py_info(f'No .zshrc or .bashrc found. Will create {config_file_to_modify}.')

log_py_info(f'Checking shell configuration file: {config_file_to_modify}')

try:
    content = ''
    if os.path.exists(config_file_to_modify):
        with open(config_file_to_modify, 'r') as f:
            content = f.read()

    if line_to_add in content:
        log_py_info('PATH configuration already exists. No changes needed.')
    else:
        log_py_info('Adding PATH configuration to shell config file.')
        with open(config_file_to_modify, 'a') as f:
            f.write(f'\n# Added by script to include local binaries\n{line_to_add}\n')
        log_py_success(f'Successfully updated {config_file_to_modify}. Please run \"source {config_file_to_modify}\" or restart your terminal for it to take effect in other sessions.')
except Exception as e:
    RED = '\033[0;31m'
    print(f'{RED}[ERROR]{NC} (Python) Failed to modify shell config: {e}', file=sys.stderr)
"


log_info "Detecting current zone..."
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | cut -d/ -f4)
log_info "Current zone: $ZONE"

if ! command -v gcloud &>/dev/null; then
    log_error "gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 >/dev/null; then
    log_error "No active gcloud authentication found. Please run 'gcloud auth login'"
    exit 1
fi

log_info "Searching for available TPUs in zone $ZONE..."
READY_TPUS=($(gcloud compute tpus tpu-vm list --zone="$ZONE" --filter="state:READY" --format="value(name)" 2>/dev/null))

case ${#READY_TPUS[@]} in
0)
    log_warning "No READY TPUs found in zone $ZONE"

    echo ""
    log_info "All TPUs in zone $ZONE:"
    gcloud compute tpus tpu-vm list --zone="$ZONE" --format="table(name,state,health,acceleratorType)" 2>/dev/null || {
        log_error "Failed to list TPUs. Check your permissions."
        exit 1
    }

    echo ""
    read -p "Enter your TPU name: " TPU_NAME < /dev/tty

    if [ -z "$TPU_NAME" ]; then
        log_error "TPU name cannot be empty"
        exit 1
    fi
    ;;
1)
    TPU_NAME="${READY_TPUS[0]}"
    log_success "Found one READY TPU: $TPU_NAME - using it automatically"
    ;;
*)
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

        if [[ "$input" =~ ^[0-9]+$ ]] && [ "$input" -ge 1 ] && [ "$input" -le ${#READY_TPUS[@]} ]; then
            TPU_NAME="${READY_TPUS[$((input - 1))]}"
            break
        elif [[ " ${READY_TPUS[@]} " =~ " ${input} " ]]; then
            TPU_NAME="$input"
            break
        else
            log_error "Invalid selection. Please enter a number (1-${#READY_TPUS[@]}) or valid TPU name."
        fi
    done
    ;;
esac

log_success "Selected TPU: $TPU_NAME"

# Get TPU accelerator type
log_info "Getting TPU accelerator type..."
TPU_TYPE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format="value(acceleratorType)" 2>/dev/null | awk -F'/' '{print $NF}')

if [ -z "$TPU_TYPE" ]; then
    log_warning "Could not determine TPU type, defaulting to v4-8"
    TPU_TYPE="v4-8"
else
    log_success "Detected TPU type: $TPU_TYPE"
fi

log_info "Installing eopod..."
if ! pip install eopod --quiet -U; then
    log_error "Failed to install eopod"
    exit 1
fi

EOPOD_PATH="$HOME/.local/bin/eopod"
if [ ! -f "$EOPOD_PATH" ]; then
    log_error "eopod not found at $EOPOD_PATH after installation"
    log_error "This might be a PATH issue. The script tried to fix it, but you may need to 'source ~/.bashrc' or 'source ~/.zshrc' and re-run."
    exit 1
fi
log_info "Configuring eopod with TPU: $TPU_NAME"
if ! "$EOPOD_PATH" configure --tpu-name "$TPU_NAME"; then
    log_error "Failed to configure eopod with TPU"
    exit 1
fi

log_success "eopod configured successfully"
log_warning "IMPORTANT: Press Enter during first execution to accept terms (terms may not be displayed)"
echo ""

# read -p "Press Enter to continue with package installations..." < /dev/tty

VENV_PATH="$HOME/easy-venv"
ENV_EOPOD_PATH="$VENV_PATH/bin/eopod"
export RAY_EXECUTABLE_PATH="$VENV_PATH/bin/ray"
log_info "Setting up virtual environment with uv at $VENV_PATH..."
log_info "Installing uv..."
if ! eopod run "pip install uv --quiet -U"; then
    log_error "Failed to install uv"
    exit 1
fi

log_info "Creating virtual environment..."
if ! eopod run "~/.local/bin/uv venv $VENV_PATH --clear --python 3.11.6"; then
    log_error "Failed to create virtual environment"
    exit 1
fi

install_package() {
    local package="$1"
    local extra_args="$2"

    log_info "Installing $package in virtual environment..."
    if ! eopod run "~/.local/bin/uv pip install --python ${VENV_PATH}/bin/python $package $extra_args --quiet"; then
        log_error "Failed to install $package"
        return 1
    fi
    log_success "Successfully installed $package"
}

echo ""
log_info "Starting package installations in virtual environment..."

~/.local/bin/uv pip install --python "${VENV_PATH}/bin/python" eopod --quiet

log_info "Uninstalling existing easydel..."
eopod run "~/.local/bin/uv pip uninstall  --python ${VENV_PATH}/bin/python easydel" 2>/dev/null || true
install_package "git+https://github.com/erfanzar/easydel.git[tpu,torch,lm_eval]" || exit 1

log_info "Configuring Ray..."
if ! "$ENV_EOPOD_PATH" auto-config-ray --self-job --python-path "$VENV_PATH/bin/python"; then
    log_error "Failed to configure Ray"
    exit 1
fi

log_success "Ray configured successfully"

echo ""
log_success "🎉 TPU setup completed successfully!"
log_info "TPU Name: $TPU_NAME"
log_info "TPU Type: $TPU_TYPE"
log_info "Zone: $ZONE"
echo ""
log_info "Final TPU status:"
gcloud compute tpus tpu-vm list --zone="$ZONE" --filter="name:$TPU_NAME" --format="table(name,state,health,acceleratorType)"
echo ""
echo ""
log_info "Virtual Environment: $VENV_PATH"
log_info "All packages installed and configured in virtual environment."
log_info "Running health check."
log_info "(maybe health check fail but the setup is fine ;/ if ur on large pods like v4-2048 ... )."

"${VENV_PATH}/bin/python" -c "
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

log_success "🎉 looks like the runtime is as healthy as it can be."